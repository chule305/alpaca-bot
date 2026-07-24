"""
Mocked tests for trading_bot.py's control flow
-------------------------------------------------
test_strategy.py proves the DECISION logic (what signal fires and why)
is correct on engineered data. It does NOT cover trading_bot.py's own
control flow -- position-cap tracking, the portfolio risk cap, daily-
loss-state persistence, and check_symbol's gating order -- since that
needs a live or mocked Alpaca connection to exercise. This file covers
exactly that gap, using unittest.mock instead of real API calls.

Run with `py test_trading_bot.py`. No network access is made -- every
Alpaca client call is mocked. (Importing trading_bot.py does read your
real .env and append a line or two to trading_log.txt via its normal
startup logging config -- same as it would on a real run.)
"""

import os
import tempfile
import types
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from alpaca.trading.enums import OrderType

import trading_bot as tb

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def make_fake_enriched(minutes_since_open: float, close: float = 100.0, atr: float = 1.0) -> pd.DataFrame:
    """A one-row stand-in for add_indicators()'s output, with just the columns check_symbol reads."""
    return pd.DataFrame({"close": [close], "atr": [atr], "minutes_since_open": [minutes_since_open]})


# ---------------------------------------------------------------------------
# 1. would_exceed_portfolio_risk_cap -- pure function, no mocking needed
# ---------------------------------------------------------------------------

def test_would_exceed_portfolio_risk_cap():
    # Explicitly forces USE_RISK_BASED_SIZING both ways rather than
    # relying on whatever your .env currently has it set to (it's a
    # user-facing on/off switch, expected to change) -- same pattern as
    # test_compute_stop_and_target in test_strategy.py.
    equity = 10000.0
    per_trade_risk = equity * tb.RISK_PER_TRADE_PCT / 100
    cap = equity * tb.MAX_PORTFOLIO_RISK_PCT / 100
    original = tb.USE_RISK_BASED_SIZING

    try:
        tb.USE_RISK_BASED_SIZING = True
        check("room under the cap does not block a new trade",
              not tb.would_exceed_portfolio_risk_cap(equity, 0.0))
        check("already at (cap - one trade's worth) still allows exactly one more",
              not tb.would_exceed_portfolio_risk_cap(equity, cap - per_trade_risk))
        check("already within one trade's worth of the cap blocks the next one",
              tb.would_exceed_portfolio_risk_cap(equity, cap - per_trade_risk + 0.01))
        check("no equity known (fetch failed this cycle) never blocks -- fail-open, not fail-closed",
              not tb.would_exceed_portfolio_risk_cap(None, cap * 10))

        tb.USE_RISK_BASED_SIZING = False
        check("disabled when USE_RISK_BASED_SIZING is off (can't cleanly estimate flat-$ trade risk)",
              not tb.would_exceed_portfolio_risk_cap(equity, cap * 10))
    finally:
        tb.USE_RISK_BASED_SIZING = original


# ---------------------------------------------------------------------------
# 2. get_current_portfolio_risk_usd -- mocked trading_client.get_orders
# ---------------------------------------------------------------------------

def make_order(symbol, order_type, stop_price):
    return types.SimpleNamespace(symbol=symbol, order_type=order_type, stop_price=stop_price)


def test_get_current_portfolio_risk_usd():
    open_positions = {
        "AAA": {"qty": 10.0, "avg_entry_price": 100.0},   # stop $95 -> $50 at risk
        "BBB": {"qty": 5.0, "avg_entry_price": 50.0},      # stop $48 -> $10 at risk
    }
    fake_orders = [
        make_order("AAA", OrderType.STOP, 95.0),
        make_order("BBB", OrderType.STOP, 48.0),
        make_order("BBB", OrderType.LIMIT, None),  # the take-profit leg -- must be ignored
    ]
    fake_client = MagicMock()
    fake_client.get_orders.return_value = fake_orders
    with patch.object(tb, "trading_client", fake_client):
        risk = tb.get_current_portfolio_risk_usd(open_positions)
    check("sums (entry - stop) * qty across positions using their open stop orders",
          np.isclose(risk, 60.0), f"got {risk}")

    failing_client = MagicMock()
    failing_client.get_orders.side_effect = Exception("API down")
    with patch.object(tb, "trading_client", failing_client):
        risk_on_failure = tb.get_current_portfolio_risk_usd(open_positions)
    check("returns 0.0 (fail-safe, not a crash) if fetching open orders fails", risk_on_failure == 0.0)

    check("returns 0.0 immediately for no open positions, without calling the API",
          tb.get_current_portfolio_risk_usd({}) == 0.0)


# ---------------------------------------------------------------------------
# 3. Daily risk state persists across a simulated restart
# ---------------------------------------------------------------------------

def test_daily_risk_state_persistence():
    tmp_dir = tempfile.mkdtemp()
    state_file = os.path.join(tmp_dir, "daily_risk_state_test.json")

    with patch.object(tb, "DAILY_RISK_STATE_FILE", state_file):
        tb.day_start_equity = 12345.67
        tb.current_trading_day = date(2026, 7, 22)
        tb.daily_loss_breaker_tripped = True
        tb.save_daily_risk_state()

        # Simulate a crash-and-restart: wipe the in-memory state, then reload it.
        tb.day_start_equity = None
        tb.current_trading_day = None
        tb.daily_loss_breaker_tripped = False
        tb.load_daily_risk_state()

    check("day_start_equity survives a save/reload round-trip", tb.day_start_equity == 12345.67)
    check("current_trading_day survives a save/reload round-trip", tb.current_trading_day == date(2026, 7, 22))
    check("daily_loss_breaker_tripped=True survives a save/reload round-trip -- "
          "this is the actual bug this fixes: a restart mid-bad-day must NOT silently un-trip the breaker",
          tb.daily_loss_breaker_tripped is True)


# ---------------------------------------------------------------------------
# 4. check_symbol's gating order -- add_indicators/decide_signal_at mocked
#    to isolate check_symbol's OWN control flow from strategy logic
#    (already covered by test_strategy.py).
# ---------------------------------------------------------------------------

def test_check_symbol_gating():
    df_input = pd.DataFrame({"close": [100.0]})  # content irrelevant, add_indicators is mocked below

    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test breakout")), \
         patch.object(tb, "place_buy_order") as mock_buy:
        notional = tb.check_symbol("AAA", df_input, entries_paused_reason="within X min of close",
                                    at_position_cap=False, current_qty=0.0, equity=10000.0, portfolio_risk_estimate=0.0)
    check("entries_paused_reason blocks a BUY signal", notional == 0.0 and not mock_buy.called)

    blackout_minute = (tb.ENTRY_BLACKOUT_START_MINUTES + tb.ENTRY_BLACKOUT_END_MINUTES) // 2
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(blackout_minute)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test breakout")), \
         patch.object(tb, "place_buy_order") as mock_buy:
        notional = tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0, portfolio_risk_estimate=0.0)
    check("the historically-weak entry blackout window blocks a BUY signal",
          notional == 0.0 and not mock_buy.called)

    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test breakout")), \
         patch.object(tb, "place_buy_order") as mock_buy:
        notional = tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                                    at_position_cap=True, current_qty=0.0, equity=10000.0, portfolio_risk_estimate=0.0)
    check("MAX_CONCURRENT_POSITIONS cap blocks a BUY signal", notional == 0.0 and not mock_buy.called)

    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test breakout")), \
         patch.object(tb, "would_exceed_portfolio_risk_cap", return_value=True), \
         patch.object(tb, "place_buy_order") as mock_buy:
        notional = tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=999999.0)
    check("the portfolio risk cap blocks a BUY signal", notional == 0.0 and not mock_buy.called)

    fake_order = types.SimpleNamespace(id="test-order-id")
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test breakout")), \
         patch.object(tb, "place_buy_order", return_value=(fake_order, 500.0)) as mock_buy:
        notional = tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0, portfolio_risk_estimate=0.0)
    check("an unblocked BUY signal calls place_buy_order and returns its notional",
          mock_buy.called and notional == 500.0)

    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("SELL", "trend_following", "test sell")), \
         patch.object(tb, "place_sell_order", return_value=types.SimpleNamespace(id="sell-id")) as mock_sell:
        notional = tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=5.0, equity=10000.0, portfolio_risk_estimate=0.0)
    check("a SELL signal on a held position calls place_sell_order (and reports 0 notional opened)",
          mock_sell.called and notional == 0.0)


# ---------------------------------------------------------------------------
# SCANNER -- regression cover for the 2026-07-24 silent-empty-watchlist bug,
# where every scan returned nothing for two days and the bot ran the whole
# time on its fallback SYMBOLS list without anything looking broken.
# ---------------------------------------------------------------------------

def test_leveraged_etf_name_detection():
    """
    Real Alpaca asset names seen live on 2026-07-24. The old ticker
    denylist knew none of these symbols; the name pattern is what
    actually catches them.
    """
    leveraged = [
        "Tradr 2X Short NBIS Daily ETF",
        "GraniteShares ETF Trust GraniteShares 2x Long NBIS Daily ETF",
        "Themes ETF Trust Leverage Shares 2X Long NBIS Daily ETF",
        "Tidal Trust II Defiance Daily Target 2X Long HIMS ETF",
        "Direxion Daily Semiconductor Bull 3X Shares",
        "ProShares UltraShort QQQ",
    ]
    ordinary = [
        "Safety Insurance Group, Inc. Common Stock",
        "RINGCENTRAL, INC.",
        "Noodles & Company Class A Common Stock",
        "Tesla, Inc. Common Stock",
        "NVIDIA Corporation Common Stock",
    ]
    for name in leveraged:
        check(f"flags leveraged ETF: {name[:45]}",
              bool(tb.LEVERAGED_ETF_NAME_PATTERN.search(name)))
    for name in ordinary:
        check(f"allows ordinary stock: {name[:45]}",
              not tb.LEVERAGED_ETF_NAME_PATTERN.search(name))


def test_scanner_tops_up_a_thin_watchlist():
    """A scan that legitimately finds only 1-2 names still leaves the bot
    with a usable watchlist, rather than a near-empty one."""
    original = (tb.active_watchlist, tb.last_scan_time)
    try:
        with patch.object(tb, "scan_for_volatile_stocks", return_value=["SAFT", "RNG"]), \
             patch.object(tb, "save_watchlist_state"), \
             patch.object(tb, "SCANNER_MIN_WATCHLIST_SIZE", 5), \
             patch.object(tb, "SYMBOLS", ["TSLA", "NVDA", "COIN", "AMD", "PLTR"]), \
             patch.object(tb, "USE_SCANNER", True):
            tb.last_scan_time = None  # force a refresh
            tb.refresh_watchlist_if_needed(datetime.now(timezone.utc))
            result = tb.active_watchlist
        check("a 2-name scan gets topped up to the 5-name floor", len(result) == 5)
        check("the scanner's own picks come first and are kept",
              result[:2] == ["SAFT", "RNG"])
        check("top-up names come from the fallback list",
              set(result[2:]) <= {"TSLA", "NVDA", "COIN", "AMD", "PLTR"})
    finally:
        tb.active_watchlist, tb.last_scan_time = original


# ---------------------------------------------------------------------------
# DURATION MODE -- the GitHub Actions entry point. Exercised here against a
# SIMULATED open market, because outside trading hours the only path that can
# be run live is the market-closed early exit.
# ---------------------------------------------------------------------------

def test_run_for_duration_cycles_while_market_open():
    """The whole point of this mode: many cycles per job, not one."""
    calls = []
    slept = []

    def fake_cycle():
        calls.append(1)
        return 300.0  # what run_one_cycle returns on a normal 5-min cadence

    def fake_sleep(seconds):
        slept.append(seconds)
        # Advance the loop's clock instead of really waiting.
        fake_sleep.now += seconds
    fake_sleep.now = 0.0

    with patch.object(tb, "run_one_cycle", side_effect=fake_cycle), \
         patch.object(tb, "market_done_for_day", return_value=False), \
         patch.object(tb.time, "sleep", side_effect=fake_sleep), \
         patch.object(tb.time, "monotonic", side_effect=lambda: fake_sleep.now):
        had_error = tb.run_for_duration(30)  # 30 min at a 5-min cadence

    check("runs many cycles in one job rather than a single one", len(calls) == 6)
    check("sleeps the interval the cycle asked for", set(slept) == {300.0})
    check("reports no error on a clean window", had_error is False)


def test_run_for_duration_never_overruns_its_window():
    """Must not sleep past the deadline -- a CI job that overruns gets
    killed mid-cycle and its state never gets committed."""
    def fake_sleep(seconds):
        fake_sleep.now += seconds
    fake_sleep.now = 0.0

    with patch.object(tb, "run_one_cycle", return_value=3600.0), \
         patch.object(tb, "market_done_for_day", return_value=False), \
         patch.object(tb.time, "sleep", side_effect=fake_sleep), \
         patch.object(tb.time, "monotonic", side_effect=lambda: fake_sleep.now):
        tb.run_for_duration(10)  # 10 min window, cycle wants to idle 60 min

    check("a long idle request is clamped to the window, not overrun",
          fake_sleep.now <= 10 * 60 + 1)


def test_run_for_duration_exits_when_session_is_over():
    calls = []
    with patch.object(tb, "run_one_cycle", side_effect=lambda: (calls.append(1), 3605.0)[1]), \
         patch.object(tb, "market_done_for_day", return_value=True), \
         patch.object(tb.time, "sleep") as mock_sleep:
        tb.run_for_duration(150)
    check("stops after one cycle once the session is over", len(calls) == 1)
    check("does not idle away the rest of the window", not mock_sleep.called)


def test_run_for_duration_survives_a_bad_cycle_but_reports_it():
    """One bad cycle must not kill the window -- but the job must still go
    red, or a run that failed every cycle would look green."""
    calls = []

    def flaky():
        calls.append(1)
        raise RuntimeError("Alpaca hiccup")

    def fake_sleep(seconds):
        fake_sleep.now += seconds
    fake_sleep.now = 0.0

    with patch.object(tb, "run_one_cycle", side_effect=flaky), \
         patch.object(tb, "market_done_for_day", return_value=False), \
         patch.object(tb.time, "sleep", side_effect=fake_sleep), \
         patch.object(tb.time, "monotonic", side_effect=lambda: fake_sleep.now):
        had_error = tb.run_for_duration(5)

    check("keeps cycling through repeated failures", len(calls) > 1)
    check("still reports failure so the run goes red", had_error is True)


if __name__ == "__main__":
    tests = [
        test_would_exceed_portfolio_risk_cap,
        test_get_current_portfolio_risk_usd,
        test_daily_risk_state_persistence,
        test_check_symbol_gating,
        test_leveraged_etf_name_detection,
        test_scanner_tops_up_a_thin_watchlist,
        test_run_for_duration_cycles_while_market_open,
        test_run_for_duration_never_overruns_its_window,
        test_run_for_duration_exits_when_session_is_over,
        test_run_for_duration_survives_a_bad_cycle_but_reports_it,
    ]
    for t in tests:
        print(f"\n{t.__name__}")
        t()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    else:
        print("All tests passed.")
