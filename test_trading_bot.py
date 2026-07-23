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
from datetime import date
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


if __name__ == "__main__":
    tests = [
        test_would_exceed_portfolio_risk_cap,
        test_get_current_portfolio_risk_usd,
        test_daily_risk_state_persistence,
        test_check_symbol_gating,
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
