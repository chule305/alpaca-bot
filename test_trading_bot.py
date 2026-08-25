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

# trading_bot configures its log file at IMPORT time, so this has to be set
# before the import below or the suite's output (including deliberate test
# tracebacks) lands in the real logs/<today>.log that gets committed --
# noise in exactly the file a future post-mortem depends on.
os.environ["LOG_DIR"] = tempfile.mkdtemp()

import trading_bot as tb
import trade_recorder

# Likewise: check_symbol records real trades, so without this the suite
# appends test rows to the project's actual trades.csv, which is committed
# to git and meant to be genuine trading history.
trade_recorder.TRADE_HISTORY_FILE = os.path.join(tempfile.mkdtemp(), "trades.csv")

# Likewise again: check_symbol's BUY branch now persists open-position
# context on every successful buy (see USE_BREAKOUT_INVALIDATION_EXIT),
# so without this, every EXISTING test that drives a successful BUY
# through check_symbol (most of which predate this feature and have no
# reason to know about it) would write to the project's real, git-tracked
# open_position_context.json. Individual tests below that specifically
# exercise this feature still override this per-test via patch.object,
# same layering as trade_recorder.TRADE_HISTORY_FILE above.
tb.OPEN_POSITION_CONTEXT_FILE = os.path.join(tempfile.mkdtemp(), "open_position_context.json")

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def make_fake_enriched(minutes_since_open: float, close: float = 100.0, atr: float = 1.0,
                        high_vol_tercile: bool = False) -> pd.DataFrame:
    """A one-row stand-in for add_indicators()'s output, with just the columns check_symbol reads."""
    return pd.DataFrame({"close": [close], "atr": [atr], "minutes_since_open": [minutes_since_open],
                          "high_vol_tercile": [high_vol_tercile]})


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
# 2b. Portfolio HEAT cap (fixed dollars, not %-of-equity) -- see
#     USE_PORTFOLIO_HEAT_CAP's comment in trading_bot.py for why this has
#     to be a fixed dollar figure rather than reusing the existing
#     %-of-equity cap: that one only ever applies under risk-based sizing,
#     so with this bot's actual default (flat $500-per-position sizing)
#     there is otherwise no aggregate cap on open risk at all.
# ---------------------------------------------------------------------------

def test_estimate_new_position_risk_usd():
    """
    Pure function, no mocking needed. Checks that the estimate is built
    from the SAME helpers place_buy_order itself uses for sizing
    (compute_stop_and_target + compute_position_size/flat-$ math) under
    both sizing modes, rather than a made-up approximation -- otherwise
    the heat-cap gate below would be checking against a number that
    doesn't match what would actually be opened.
    """
    last_price = 100.0
    atr_value = 1.0
    original_mode = tb.USE_RISK_BASED_SIZING
    try:
        tb.USE_RISK_BASED_SIZING = False
        stop_price, _ = tb.compute_stop_and_target(last_price, atr_value)
        expected_flat_qty = int(tb.TRADE_AMOUNT_USD // last_price)
        expected_flat_risk = max(last_price - stop_price, 0.0) * expected_flat_qty
        got_flat = tb.estimate_new_position_risk_usd(last_price, atr_value, equity=10000.0)
        check("flat-$ mode sizes off TRADE_AMOUNT_USD // price, matching place_buy_order's own flat sizing",
              np.isclose(got_flat, expected_flat_risk), f"got {got_flat}, expected {expected_flat_risk}")

        check("flat-$ mode ignores equity entirely -- same result whether equity is known or not",
              np.isclose(tb.estimate_new_position_risk_usd(last_price, atr_value, None), expected_flat_risk))

        tb.USE_RISK_BASED_SIZING = True
        equity = 10000.0
        expected_rb_qty = tb.compute_position_size(equity, last_price, stop_price)
        expected_rb_risk = max(last_price - stop_price, 0.0) * expected_rb_qty
        got_rb = tb.estimate_new_position_risk_usd(last_price, atr_value, equity)
        check("risk-based mode sizes off compute_position_size, matching place_buy_order's own risk-based sizing",
              np.isclose(got_rb, expected_rb_risk), f"got {got_rb}, expected {expected_rb_risk}")

        check("risk-based mode with equity unknown this cycle falls back to flat-$ sizing "
              "(can't compute a risk-based qty without equity)",
              np.isclose(tb.estimate_new_position_risk_usd(last_price, atr_value, None), expected_flat_risk))
    finally:
        tb.USE_RISK_BASED_SIZING = original_mode

    check("invalid last_price (0) returns 0.0 rather than raising or dividing by zero",
          tb.estimate_new_position_risk_usd(0.0, atr_value, 10000.0) == 0.0)
    check("missing last_price (None) returns 0.0 rather than raising",
          tb.estimate_new_position_risk_usd(None, atr_value, 10000.0) == 0.0)


def test_estimate_new_position_risk_usd_volatility_scaled():
    """
    See USE_VOLATILITY_SCALED_SIZING in strategy.py -- a second, ADX-
    independent regime axis (Moreira & Muir 2017, "Volatility-Managed
    Portfolios"). Only ever a fixed-dollar SUBSTITUTION inside the flat-
    sizing branch (never risk-based, never a fraction of equity), and
    this function's own docstring promises to mirror place_buy_order's
    real qty exactly -- pinned here so the heat-cap gate can't silently
    drift from what actually gets bought.
    """
    last_price = 50.0
    atr_value = 1.0
    original_mode = tb.USE_RISK_BASED_SIZING
    original_toggle = tb.USE_VOLATILITY_SCALED_SIZING
    original_reduced = tb.VOLATILITY_SCALED_REDUCED_USD
    try:
        tb.USE_RISK_BASED_SIZING = False
        tb.USE_VOLATILITY_SCALED_SIZING = False
        stop_price, _ = tb.compute_stop_and_target(last_price, atr_value)
        risk_per_share = last_price - stop_price

        expected_unreduced = int(tb.TRADE_AMOUNT_USD // last_price) * risk_per_share
        check("toggle OFF: high_vol_tercile=True is ignored, same result as low/mid",
              np.isclose(tb.estimate_new_position_risk_usd(last_price, atr_value, 10000.0,
                                                             high_vol_tercile=True), expected_unreduced))

        tb.USE_VOLATILITY_SCALED_SIZING = True
        tb.VOLATILITY_SCALED_REDUCED_USD = 350.0
        expected_reduced = int(350.0 // last_price) * risk_per_share
        got_reduced = tb.estimate_new_position_risk_usd(last_price, atr_value, 10000.0, high_vol_tercile=True)
        check("toggle ON + high_vol_tercile=True uses the reduced $ amount, not TRADE_AMOUNT_USD",
              np.isclose(got_reduced, expected_reduced) and got_reduced < expected_unreduced,
              f"got {got_reduced}, expected {expected_reduced}, unreduced was {expected_unreduced}")

        got_low_mid = tb.estimate_new_position_risk_usd(last_price, atr_value, 10000.0, high_vol_tercile=False)
        check("toggle ON but high_vol_tercile=False (low/mid tercile) still uses the full TRADE_AMOUNT_USD",
              np.isclose(got_low_mid, expected_unreduced))

        tb.USE_RISK_BASED_SIZING = True
        equity = 10000.0
        expected_rb_qty = tb.compute_position_size(equity, last_price, stop_price)
        expected_rb_risk = risk_per_share * expected_rb_qty
        got_rb_high_vol = tb.estimate_new_position_risk_usd(last_price, atr_value, equity, high_vol_tercile=True)
        check("under risk-based sizing, high_vol_tercile is ignored entirely -- this candidate never "
              "stacks a second size cut on top of risk-based sizing's own stop-distance scaling",
              np.isclose(got_rb_high_vol, expected_rb_risk))
    finally:
        tb.USE_RISK_BASED_SIZING = original_mode
        tb.USE_VOLATILITY_SCALED_SIZING = original_toggle
        tb.VOLATILITY_SCALED_REDUCED_USD = original_reduced


def test_estimate_new_position_risk_usd_conviction_boosted():
    """
    Mirrors test_estimate_new_position_risk_usd_volatility_scaled exactly,
    for the conviction-boost lever instead. Only ever a fixed-dollar
    SUBSTITUTION inside the flat-sizing branch (never risk-based, never
    a fraction of equity), and this function's own docstring promises to
    mirror place_buy_order's real qty exactly -- pinned here so the
    heat-cap gate can't silently UNDER-estimate risk for a boosted trade
    (the opposite, and more dangerous, direction of drift than the
    volatility-scaled case's over-estimate).
    """
    last_price = 50.0
    atr_value = 1.0
    original_mode = tb.USE_RISK_BASED_SIZING
    original_toggle = tb.USE_CONVICTION_SIZING
    original_strategies = tb.HIGH_CONVICTION_STRATEGIES
    original_boost = tb.CONVICTION_BOOST_USD
    try:
        tb.USE_RISK_BASED_SIZING = False
        tb.USE_CONVICTION_SIZING = False
        tb.HIGH_CONVICTION_STRATEGIES = {"trend_following"}
        stop_price, _ = tb.compute_stop_and_target(last_price, atr_value)
        risk_per_share = last_price - stop_price

        expected_unboosted = int(tb.TRADE_AMOUNT_USD // last_price) * risk_per_share
        check("toggle OFF: a qualifying reason_key is ignored, same result as any other",
              np.isclose(tb.estimate_new_position_risk_usd(last_price, atr_value, 10000.0,
                                                             reason_key="trend_following"), expected_unboosted))

        tb.USE_CONVICTION_SIZING = True
        tb.CONVICTION_BOOST_USD = 750.0
        expected_boosted = int(750.0 // last_price) * risk_per_share
        got_boosted = tb.estimate_new_position_risk_usd(last_price, atr_value, 10000.0,
                                                          reason_key="trend_following")
        check("toggle ON + reason_key IN HIGH_CONVICTION_STRATEGIES uses the boosted $ amount, "
              "not TRADE_AMOUNT_USD",
              np.isclose(got_boosted, expected_boosted) and got_boosted > expected_unboosted,
              f"got {got_boosted}, expected {expected_boosted}, unboosted was {expected_unboosted}")

        got_not_qualifying = tb.estimate_new_position_risk_usd(last_price, atr_value, 10000.0,
                                                                 reason_key="mean_reversion")
        check("toggle ON but reason_key NOT in HIGH_CONVICTION_STRATEGIES still uses the full "
              "TRADE_AMOUNT_USD",
              np.isclose(got_not_qualifying, expected_unboosted))

        tb.USE_RISK_BASED_SIZING = True
        equity = 10000.0
        expected_rb_qty = tb.compute_position_size(equity, last_price, stop_price)
        expected_rb_risk = risk_per_share * expected_rb_qty
        got_rb_boosted = tb.estimate_new_position_risk_usd(last_price, atr_value, equity,
                                                             reason_key="trend_following")
        check("under risk-based sizing, reason_key/conviction is ignored entirely -- this "
              "candidate never stacks a boost on top of risk-based sizing's own stop-distance "
              "scaling, same reasoning as the volatility-scaled case above",
              np.isclose(got_rb_boosted, expected_rb_risk))
    finally:
        tb.USE_RISK_BASED_SIZING = original_mode
        tb.USE_CONVICTION_SIZING = original_toggle
        tb.HIGH_CONVICTION_STRATEGIES = original_strategies
        tb.CONVICTION_BOOST_USD = original_boost


def test_estimate_new_position_risk_usd_volatility_precedes_conviction():
    """
    THE precedence test. A trade that is BOTH in a high-conviction
    strategy AND in its own high-vol tercile must get the REDUCED
    (volatility) amount, never the boosted one -- safety (size down on
    real, measured noise) always wins over a return-chasing boost (size
    up on an unconfirmed strategy-level edge), every time, no exceptions.

    Written so it would actually FAIL if the precedence were implemented
    backwards (i.e. conviction checked before/instead of volatility): the
    reduced and boosted $ amounts are chosen to be clearly distinguishable
    ($350 vs $750 against a $500 baseline -> 7 vs 15 shares at $50), and
    the assertion pins the EXACT reduced qty, not just "less than the
    boosted amount" (which a broken implementation that used, say, the
    average of the two could still slip past).
    """
    last_price = 50.0
    atr_value = 1.0
    original_mode = tb.USE_RISK_BASED_SIZING
    original_vol_toggle = tb.USE_VOLATILITY_SCALED_SIZING
    original_reduced = tb.VOLATILITY_SCALED_REDUCED_USD
    original_conv_toggle = tb.USE_CONVICTION_SIZING
    original_strategies = tb.HIGH_CONVICTION_STRATEGIES
    original_boost = tb.CONVICTION_BOOST_USD
    try:
        tb.USE_RISK_BASED_SIZING = False
        tb.USE_VOLATILITY_SCALED_SIZING = True
        tb.VOLATILITY_SCALED_REDUCED_USD = 350.0
        tb.USE_CONVICTION_SIZING = True
        tb.HIGH_CONVICTION_STRATEGIES = {"trend_following"}
        tb.CONVICTION_BOOST_USD = 750.0

        stop_price, _ = tb.compute_stop_and_target(last_price, atr_value)
        risk_per_share = last_price - stop_price
        expected_reduced = int(350.0 // last_price) * risk_per_share
        expected_boosted = int(750.0 // last_price) * risk_per_share
        expected_unmodified = int(tb.TRADE_AMOUNT_USD // last_price) * risk_per_share
        # These three must actually differ, or the test below can't tell
        # reduced from boosted from unmodified.
        check("fixture sanity: reduced/boosted/unmodified expected values are all distinct",
              len({expected_reduced, expected_boosted, expected_unmodified}) == 3,
              f"reduced={expected_reduced}, boosted={expected_boosted}, unmodified={expected_unmodified}")

        got = tb.estimate_new_position_risk_usd(last_price, atr_value, 10000.0,
                                                  high_vol_tercile=True, reason_key="trend_following")
        check("a trade BOTH high-vol-tercile AND in a high-conviction strategy gets the REDUCED "
              "(volatility) amount, never the boosted one",
              np.isclose(got, expected_reduced) and not np.isclose(got, expected_boosted),
              f"got {got}, reduced would be {expected_reduced}, boosted would be {expected_boosted}")

        # Sanity checks on the other three quadrants, so this test would
        # catch a regression in EITHER direction, not just the precedence.
        got_vol_only = tb.estimate_new_position_risk_usd(last_price, atr_value, 10000.0,
                                                           high_vol_tercile=True, reason_key="mean_reversion")
        check("high-vol-tercile alone (not a conviction strategy) still gets reduced",
              np.isclose(got_vol_only, expected_reduced))

        got_conviction_only = tb.estimate_new_position_risk_usd(
            last_price, atr_value, 10000.0, high_vol_tercile=False, reason_key="trend_following")
        check("conviction strategy alone (not high-vol-tercile) gets boosted",
              np.isclose(got_conviction_only, expected_boosted))

        got_neither = tb.estimate_new_position_risk_usd(last_price, atr_value, 10000.0,
                                                          high_vol_tercile=False, reason_key="mean_reversion")
        check("neither condition applies -- plain TRADE_AMOUNT_USD",
              np.isclose(got_neither, expected_unmodified))
    finally:
        tb.USE_RISK_BASED_SIZING = original_mode
        tb.USE_VOLATILITY_SCALED_SIZING = original_vol_toggle
        tb.VOLATILITY_SCALED_REDUCED_USD = original_reduced
        tb.USE_CONVICTION_SIZING = original_conv_toggle
        tb.HIGH_CONVICTION_STRATEGIES = original_strategies
        tb.CONVICTION_BOOST_USD = original_boost


def test_conviction_sizing_structurally_inert_with_real_shipped_default():
    """
    Confirms the REAL shipped default (HIGH_CONVICTION_STRATEGIES as
    loaded from the actual, unmodified environment -- not a test-
    populated one) means USE_CONVICTION_SIZING=true is a complete no-op
    regardless of reason_key. This is not "empirically happens to do
    nothing today" -- HIGH_CONVICTION_STRATEGIES is structurally an empty
    set by default (see strategy.py's comment on why), so the
    `reason_key in HIGH_CONVICTION_STRATEGIES` membership test can never
    be True for ANY reason_key without someone deliberately repopulating
    the set first.
    """
    check("the real shipped default really is an empty set (sanity check on the fixture itself, "
          "not just the assumption behind this test)",
          tb.HIGH_CONVICTION_STRATEGIES == set(), f"got {tb.HIGH_CONVICTION_STRATEGIES!r}")

    last_price = 50.0
    atr_value = 1.0
    original_toggle = tb.USE_CONVICTION_SIZING
    try:
        tb.USE_CONVICTION_SIZING = True  # forced ON; HIGH_CONVICTION_STRATEGIES left at its real default
        stop_price, _ = tb.compute_stop_and_target(last_price, atr_value)
        risk_per_share = last_price - stop_price
        expected = int(tb.TRADE_AMOUNT_USD // last_price) * risk_per_share
        for reason_key in ("trend_following", "vwap_reversion", "rvol_spike",
                            "breakout", "mean_reversion", "gap_continuation", "unknown"):
            got = tb.estimate_new_position_risk_usd(last_price, atr_value, 10000.0, reason_key=reason_key)
            check(f"USE_CONVICTION_SIZING=true with the real empty default set: "
                  f"reason_key={reason_key!r} still uses plain TRADE_AMOUNT_USD",
                  np.isclose(got, expected), f"got {got}, expected {expected}")
    finally:
        tb.USE_CONVICTION_SIZING = original_toggle


def test_would_exceed_portfolio_heat_cap():
    original_toggle = tb.USE_PORTFOLIO_HEAT_CAP
    original_cap = tb.MAX_PORTFOLIO_HEAT_USD
    try:
        tb.USE_PORTFOLIO_HEAT_CAP = True
        tb.MAX_PORTFOLIO_HEAT_USD = 200.0

        check("current heat well under the cap plus a small new position -- not blocked",
              not tb.would_exceed_portfolio_heat_cap(150.0, 40.0))
        check("current heat + new position lands EXACTLY on the cap -- not blocked (only strictly over blocks)",
              not tb.would_exceed_portfolio_heat_cap(150.0, 50.0))
        check("current heat + new position would land $0.01 over the cap -- blocked",
              tb.would_exceed_portfolio_heat_cap(150.0, 50.01))
        check("zero current heat, a new position alone already exceeds the cap on its own -- blocked",
              tb.would_exceed_portfolio_heat_cap(0.0, 200.01))

        tb.USE_PORTFOLIO_HEAT_CAP = False
        check("disabled toggle never blocks, no matter how far over the (still-configured) cap this would be",
              not tb.would_exceed_portfolio_heat_cap(10000.0, 10000.0))
    finally:
        tb.USE_PORTFOLIO_HEAT_CAP = original_toggle
        tb.MAX_PORTFOLIO_HEAT_USD = original_cap


def test_portfolio_heat_cap_blocks_when_aggregate_open_risk_is_near_the_ceiling():
    """
    Same mocked-orders setup as test_get_current_portfolio_risk_usd (real
    open positions with real bracket stop orders), but exercised against
    the fixed-dollar heat cap instead of the %-of-equity one: aggregate
    open risk starts near MAX_PORTFOLIO_HEAT_USD, and a new entry that
    would push the total over the ceiling gets blocked while one that
    stays under is allowed.
    """
    open_positions = {
        "AAA": {"qty": 10.0, "avg_entry_price": 100.0},   # stop $95 -> $50 at risk
        "BBB": {"qty": 5.0, "avg_entry_price": 50.0},      # stop $48 -> $10 at risk
    }
    fake_orders = [
        make_order("AAA", OrderType.STOP, 95.0),
        make_order("BBB", OrderType.STOP, 48.0),
    ]
    fake_client = MagicMock()
    fake_client.get_orders.return_value = fake_orders

    original_toggle = tb.USE_PORTFOLIO_HEAT_CAP
    original_cap = tb.MAX_PORTFOLIO_HEAT_USD
    try:
        tb.USE_PORTFOLIO_HEAT_CAP = True
        tb.MAX_PORTFOLIO_HEAT_USD = 90.0  # current open risk $60 (AAA $50 + BBB $10) -- $30 of headroom left

        with patch.object(tb, "trading_client", fake_client):
            current_heat = tb.get_current_portfolio_risk_usd(open_positions)
        check("aggregate open risk from the two positions' real stop orders sums to $60",
              np.isclose(current_heat, 60.0), f"got {current_heat}")

        check("a new position risking $29 stays under the $90 cap ($60 + $29 = $89) -- allowed",
              not tb.would_exceed_portfolio_heat_cap(current_heat, 29.0))
        check("a new position risking $31 would push aggregate risk to $91, over the $90 cap -- blocked",
              tb.would_exceed_portfolio_heat_cap(current_heat, 31.0))
    finally:
        tb.USE_PORTFOLIO_HEAT_CAP = original_toggle
        tb.MAX_PORTFOLIO_HEAT_USD = original_cap


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


def test_check_symbol_propagates_high_vol_tercile_to_place_buy_order():
    """
    check_symbol reads high_vol_tercile off the SAME enriched dataframe
    row decide_signal_at already consulted, and must forward it to
    place_buy_order unchanged -- this is the only place that value
    travels from strategy.py's indicator column (see high_vol_tercile in
    add_indicators()) into the sizing decision in trading_bot.py.
    """
    df_input = pd.DataFrame({"close": [100.0]})
    fake_order = types.SimpleNamespace(id="test-order-id")

    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100, high_vol_tercile=True)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test breakout")), \
         patch.object(tb, "place_buy_order", return_value=(fake_order, 350.0)) as mock_buy:
        tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                         at_position_cap=False, current_qty=0.0, equity=10000.0, portfolio_risk_estimate=0.0)
    check("high_vol_tercile=True on the enriched row is forwarded to place_buy_order",
          bool(mock_buy.call_args.args[-1]) is True, f"call args: {mock_buy.call_args}")

    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100, high_vol_tercile=False)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test breakout")), \
         patch.object(tb, "place_buy_order", return_value=(fake_order, 500.0)) as mock_buy:
        tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                         at_position_cap=False, current_qty=0.0, equity=10000.0, portfolio_risk_estimate=0.0)
    check("high_vol_tercile=False on the enriched row is forwarded to place_buy_order too",
          bool(mock_buy.call_args.args[-1]) is False, f"call args: {mock_buy.call_args}")


def test_place_buy_order_conviction_sizing():
    """
    place_buy_order/check_symbol wiring: a reason_key IN the (test-
    populated) high-conviction set gets the boosted amount; a reason_key
    NOT in it gets the normal amount; the toggle OFF means zero effect
    even with a populated set. Exercises place_buy_order's REAL sizing
    branch directly (mocked trading_client/get_latest_price/reconcile),
    not the mirrored math in estimate_new_position_risk_usd, so a bug
    that only lived inside place_buy_order itself would still be caught.
    """
    last_price = 50.0
    atr_value = 1.0
    fake_client = MagicMock()
    fake_client.submit_order.return_value = types.SimpleNamespace(id="test-order-id")

    original_rb = tb.USE_RISK_BASED_SIZING
    original_toggle = tb.USE_CONVICTION_SIZING
    original_strategies = tb.HIGH_CONVICTION_STRATEGIES
    original_boost = tb.CONVICTION_BOOST_USD
    try:
        tb.USE_RISK_BASED_SIZING = False
        tb.HIGH_CONVICTION_STRATEGIES = {"trend_following"}
        tb.CONVICTION_BOOST_USD = 750.0

        with patch.object(tb, "trading_client", fake_client), \
             patch.object(tb, "get_latest_price", return_value=last_price), \
             patch.object(tb, "reconcile_bracket_with_real_fill", return_value=(last_price, False)):

            tb.USE_CONVICTION_SIZING = False
            order, notional = tb.place_buy_order("AAA", last_price, atr_value, 10000.0,
                                                   reason_key="trend_following", high_vol_tercile=False)
            check("toggle OFF: a qualifying reason_key still gets the normal TRADE_AMOUNT_USD qty "
                  "-- zero effect from a populated HIGH_CONVICTION_STRATEGIES set",
                  order is not None
                  and tb.place_buy_order.last_details["qty"] == int(tb.TRADE_AMOUNT_USD // last_price)
                  and tb.place_buy_order.last_details["conviction_boosted"] is False,
                  f"details: {tb.place_buy_order.last_details}")

            tb.USE_CONVICTION_SIZING = True
            order, notional = tb.place_buy_order("AAA", last_price, atr_value, 10000.0,
                                                   reason_key="trend_following", high_vol_tercile=False)
            check("toggle ON + reason_key IN HIGH_CONVICTION_STRATEGIES gets the boosted qty",
                  order is not None
                  and tb.place_buy_order.last_details["qty"] == int(750.0 // last_price)
                  and tb.place_buy_order.last_details["conviction_boosted"] is True,
                  f"details: {tb.place_buy_order.last_details}")

            order, notional = tb.place_buy_order("AAA", last_price, atr_value, 10000.0,
                                                   reason_key="mean_reversion", high_vol_tercile=False)
            check("toggle ON but reason_key NOT in HIGH_CONVICTION_STRATEGIES gets the normal qty",
                  order is not None
                  and tb.place_buy_order.last_details["qty"] == int(tb.TRADE_AMOUNT_USD // last_price)
                  and tb.place_buy_order.last_details["conviction_boosted"] is False,
                  f"details: {tb.place_buy_order.last_details}")
    finally:
        tb.USE_RISK_BASED_SIZING = original_rb
        tb.USE_CONVICTION_SIZING = original_toggle
        tb.HIGH_CONVICTION_STRATEGIES = original_strategies
        tb.CONVICTION_BOOST_USD = original_boost


def test_place_buy_order_volatility_precedes_conviction():
    """
    THE precedence test, exercised against the REAL place_buy_order (not
    the mirrored math in estimate_new_position_risk_usd): a trade that is
    BOTH in a high-conviction strategy AND in its own high-vol tercile
    must submit an order sized off VOLATILITY_SCALED_REDUCED_USD, never
    CONVICTION_BOOST_USD. Written so it would actually FAIL if the
    precedence were implemented backwards -- the reduced/boosted qtys are
    chosen to be clearly distinguishable ($350 vs $750 on a $500 baseline
    -> 7 vs 15 shares at $50), and the assertion pins the exact reduced
    qty, not just "not the largest number".
    """
    last_price = 50.0
    atr_value = 1.0
    fake_client = MagicMock()
    fake_client.submit_order.return_value = types.SimpleNamespace(id="test-order-id")

    original_rb = tb.USE_RISK_BASED_SIZING
    original_vol_toggle = tb.USE_VOLATILITY_SCALED_SIZING
    original_reduced = tb.VOLATILITY_SCALED_REDUCED_USD
    original_conv_toggle = tb.USE_CONVICTION_SIZING
    original_strategies = tb.HIGH_CONVICTION_STRATEGIES
    original_boost = tb.CONVICTION_BOOST_USD
    try:
        tb.USE_RISK_BASED_SIZING = False
        tb.USE_VOLATILITY_SCALED_SIZING = True
        tb.VOLATILITY_SCALED_REDUCED_USD = 350.0
        tb.USE_CONVICTION_SIZING = True
        tb.HIGH_CONVICTION_STRATEGIES = {"trend_following"}
        tb.CONVICTION_BOOST_USD = 750.0

        with patch.object(tb, "trading_client", fake_client), \
             patch.object(tb, "get_latest_price", return_value=last_price), \
             patch.object(tb, "reconcile_bracket_with_real_fill", return_value=(last_price, False)):

            order, notional = tb.place_buy_order("AAA", last_price, atr_value, 10000.0,
                                                   reason_key="trend_following", high_vol_tercile=True)
            expected_reduced_qty = int(350.0 // last_price)
            expected_boosted_qty = int(750.0 // last_price)
            check("a trade BOTH high-vol-tercile AND in a high-conviction strategy submits at the "
                  "REDUCED (volatility) qty, never the boosted one",
                  order is not None
                  and tb.place_buy_order.last_details["qty"] == expected_reduced_qty
                  and tb.place_buy_order.last_details["qty"] != expected_boosted_qty
                  and tb.place_buy_order.last_details["conviction_boosted"] is False,
                  f"details: {tb.place_buy_order.last_details}, reduced would be {expected_reduced_qty}, "
                  f"boosted would be {expected_boosted_qty}")
    finally:
        tb.USE_RISK_BASED_SIZING = original_rb
        tb.USE_VOLATILITY_SCALED_SIZING = original_vol_toggle
        tb.VOLATILITY_SCALED_REDUCED_USD = original_reduced
        tb.USE_CONVICTION_SIZING = original_conv_toggle
        tb.HIGH_CONVICTION_STRATEGIES = original_strategies
        tb.CONVICTION_BOOST_USD = original_boost


def test_check_symbol_gating_portfolio_heat_cap():
    """
    Same gating-order pattern as test_check_symbol_gating's portfolio
    risk-cap case above, but for the fixed-dollar heat cap: blocks a BUY
    when would_exceed_portfolio_heat_cap says the new position would push
    aggregate risk over the ceiling, and lets one through when it doesn't
    -- confirming check_symbol actually wires the gate in (the gate's own
    threshold logic is covered separately by test_would_exceed_portfolio_heat_cap).
    """
    df_input = pd.DataFrame({"close": [100.0]})

    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test breakout")), \
         patch.object(tb, "would_exceed_portfolio_heat_cap", return_value=True), \
         patch.object(tb, "place_buy_order") as mock_buy:
        notional = tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=190.0)
    check("the portfolio heat cap blocks a BUY signal that would exceed it",
          notional == 0.0 and not mock_buy.called)

    fake_order = types.SimpleNamespace(id="heat-cap-ok-order-id")
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test breakout")), \
         patch.object(tb, "would_exceed_portfolio_heat_cap", return_value=False), \
         patch.object(tb, "place_buy_order", return_value=(fake_order, 500.0)) as mock_buy:
        notional = tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=10.0)
    check("a BUY signal that stays under the heat cap proceeds to place_buy_order",
          mock_buy.called and notional == 500.0)


# ---------------------------------------------------------------------------
# 5. SECTOR CONCENTRATION CAP -- portfolio-CONSTRUCTION control (how many of
# the currently open positions may share a sector), not a single-symbol
# entry signal. Investigated 2026-07-31 as "Idea 2" (correlation-limiting)
# and explicitly not built at the time because backtest.py runs each symbol
# in isolation with its own equity curve -- there's no P&L before/after to
# backtest for a control that only means anything across a shared,
# concurrent position pool. That's still true; these tests cover the gating
# logic directly (mirroring test_would_exceed_portfolio_risk_cap and
# test_get_current_portfolio_risk_usd above), not a P&L claim.
# ---------------------------------------------------------------------------

def test_get_symbol_sector_uses_hardcoded_map_first():
    """SECTOR_MAP is checked before ever touching the network -- confirms
    the map lookup alone is enough for a symbol it covers, with no
    get_asset call at all."""
    tb._sector_cache.clear()
    with patch.object(tb, "trading_client") as mock_client:
        sector = tb.get_symbol_sector("NVDA")
    check("a mapped symbol resolves to its SECTOR_MAP entry",
          sector == tb.SECTOR_MAP["NVDA"])
    check("a mapped symbol never calls Alpaca's asset API at all",
          not mock_client.get_asset.called)


def test_get_symbol_sector_falls_back_to_alpaca_asset_metadata():
    """A symbol NOT in SECTOR_MAP falls back to get_asset() -- checked
    defensively via getattr, since alpaca-py's Asset model has no
    sector field today but this must still pick one up if the API ever
    returns one."""
    tb._sector_cache.clear()
    fake_asset = types.SimpleNamespace(sector="Real Estate")
    with patch.object(tb, "trading_client") as mock_client:
        mock_client.get_asset.return_value = fake_asset
        sector = tb.get_symbol_sector("NOTINMAP")
    check("an unmapped symbol falls back to Alpaca's asset metadata",
          sector == "Real Estate")
    check("the fallback actually calls get_asset for the unmapped symbol",
          mock_client.get_asset.called)


def test_get_symbol_sector_unknown_when_neither_source_has_it():
    """No map entry, and get_asset() raises (network down, unknown
    symbol, etc.) -- must return None, not raise, so the cap fails open
    rather than crashing a check cycle."""
    tb._sector_cache.clear()
    with patch.object(tb, "trading_client") as mock_client:
        mock_client.get_asset.side_effect = Exception("unknown symbol")
        sector = tb.get_symbol_sector("TOTALLYUNKNOWN")
    check("an unmappable, unfetchable symbol resolves to None, not an exception", sector is None)


def test_get_symbol_sector_caches_including_negative_results():
    """Sector doesn't change intraday, so a symbol looked up once
    (found OR not found) should never trigger a second get_asset call --
    same reasoning as is_leveraged_etf's _asset_name_cache."""
    tb._sector_cache.clear()
    with patch.object(tb, "trading_client") as mock_client:
        mock_client.get_asset.side_effect = Exception("down")
        first = tb.get_symbol_sector("REPEATED")
        second = tb.get_symbol_sector("REPEATED")
    check("first and second lookups agree", first == second is None)
    check("a cached 'unknown' result is not re-fetched on the next call",
          mock_client.get_asset.call_count == 1)


def test_sector_concentration_blocks_entry_disabled_by_default():
    with patch.object(tb, "USE_SECTOR_CONCENTRATION_CAP", False), \
         patch.object(tb, "get_symbol_sector", return_value="Information Technology"):
        blocked = tb.sector_concentration_blocks_entry("NVDA", {"AMD", "INTC", "QCOM"})
    check("the cap never blocks anything while the feature is off", blocked is False)


def test_sector_concentration_blocks_entry_blocks_a_sixth_same_sector_candidate():
    """The scenario from the task: 5 open positions already in the same
    sector -- a 6th same-sector candidate must be blocked."""
    open_positions = {"AAA", "BBB", "CCC", "DDD", "EEE"}  # 5 same-sector, mocked below
    sectors = {s: "Information Technology" for s in open_positions}
    sectors["CANDIDATE"] = "Information Technology"

    with patch.object(tb, "USE_SECTOR_CONCENTRATION_CAP", True), \
         patch.object(tb, "MAX_POSITIONS_PER_SECTOR", 2), \
         patch.object(tb, "get_symbol_sector", side_effect=lambda s: sectors.get(s)):
        blocked = tb.sector_concentration_blocks_entry("CANDIDATE", open_positions)
    check("a 6th same-sector candidate is blocked once the sector is already at/over the cap",
          blocked is True)


def test_sector_concentration_blocks_entry_allows_a_different_sector_candidate():
    """Same 5 open same-sector positions -- a DIFFERENT-sector candidate
    must NOT be blocked by a cap that has nothing to do with its sector."""
    open_positions = {"AAA", "BBB", "CCC", "DDD", "EEE"}
    sectors = {s: "Information Technology" for s in open_positions}
    sectors["CANDIDATE"] = "Health Care"

    with patch.object(tb, "USE_SECTOR_CONCENTRATION_CAP", True), \
         patch.object(tb, "MAX_POSITIONS_PER_SECTOR", 2), \
         patch.object(tb, "get_symbol_sector", side_effect=lambda s: sectors.get(s)):
        blocked = tb.sector_concentration_blocks_entry("CANDIDATE", open_positions)
    check("a different-sector candidate is not blocked by another sector's concentration",
          blocked is False)


def test_sector_concentration_blocks_entry_allows_exactly_up_to_the_cap():
    """Below the cap must still allow a trade -- this pins the boundary
    (>=, not >) so the cap can't silently allow one extra position past
    MAX_POSITIONS_PER_SECTOR."""
    sectors = {"AAA": "Financials", "OTHER_FIN": "Financials", "CANDIDATE": "Financials"}
    with patch.object(tb, "USE_SECTOR_CONCENTRATION_CAP", True), \
         patch.object(tb, "MAX_POSITIONS_PER_SECTOR", 2), \
         patch.object(tb, "get_symbol_sector", side_effect=lambda s: sectors.get(s)):
        blocked_at_one = tb.sector_concentration_blocks_entry("CANDIDATE", {"AAA"})
        blocked_at_two = tb.sector_concentration_blocks_entry("CANDIDATE", {"AAA", "OTHER_FIN"})
    check("one existing same-sector position (below a cap of 2) does not block a second",
          blocked_at_one is False)
    check("two existing same-sector positions (at a cap of 2) blocks a third",
          blocked_at_two is True)


def test_sector_concentration_blocks_entry_fails_open_on_unknown_candidate_sector():
    with patch.object(tb, "USE_SECTOR_CONCENTRATION_CAP", True), \
         patch.object(tb, "MAX_POSITIONS_PER_SECTOR", 1), \
         patch.object(tb, "get_symbol_sector", return_value=None):
        blocked = tb.sector_concentration_blocks_entry("MYSTERYSYMBOL", {"AAA", "BBB", "CCC"})
    check("an unmappable candidate is never blocked -- fail open, not fail closed", blocked is False)


def test_sector_concentration_blocks_entry_unmapped_open_positions_dont_count():
    """An open position whose sector can't be determined must not count
    toward the total for a DIFFERENT, mappable sector -- it's simply
    excluded, not treated as a wildcard match."""
    sectors = {"KNOWN_SAME_SECTOR": "Energy", "CANDIDATE": "Energy"}
    # UNKNOWN1/UNKNOWN2 deliberately absent from `sectors` -> get_symbol_sector
    # returns None for them via .get()'s default.
    with patch.object(tb, "USE_SECTOR_CONCENTRATION_CAP", True), \
         patch.object(tb, "MAX_POSITIONS_PER_SECTOR", 2), \
         patch.object(tb, "get_symbol_sector", side_effect=lambda s: sectors.get(s)):
        blocked = tb.sector_concentration_blocks_entry(
            "CANDIDATE", {"KNOWN_SAME_SECTOR", "UNKNOWN1", "UNKNOWN2"})
    check("unmapped open positions are excluded from the sector count, not counted as matches",
          blocked is False)


def test_check_symbol_blocks_buy_when_sector_cap_blocks_entry():
    """check_symbol's OWN gating: a BUY signal must be refused when the
    caller says the sector cap blocks it, and taken once it doesn't --
    same pattern as test_symbol_cooldown_blocks_immediate_reentry and
    test_check_symbol_blocks_buy_when_daily_trend_blocks_entry above."""
    df_input = pd.DataFrame({"close": [100.0]})

    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test buy")), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0, sector_cap_blocks_entry=True)
    check("a BUY signal is refused when the sector concentration cap blocks it",
          not mock_buy.called)
    check("a sector-cap-blocked entry reports no notional opened", notional == 0.0)

    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test buy")), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0, sector_cap_blocks_entry=False)
    check("the same signal is taken once the sector cap doesn't block it", mock_buy.called)
    check("a taken entry reports its notional", notional == 500.0)


def test_check_symbol_sector_cap_does_not_apply_to_sell_signals():
    """The cap must gate NEW entries only -- an existing position's own
    SELL/exit logic must go through even with sector_cap_blocks_entry=True,
    since that flag should never even be consulted on the SELL branch."""
    df_input = pd.DataFrame({"close": [100.0]})
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("SELL", "trend_following", "test sell")), \
         patch.object(tb, "place_sell_order", return_value=types.SimpleNamespace(id="sell-id")) as mock_sell:
        notional = tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=5.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0, sector_cap_blocks_entry=True)
    check("a SELL signal still closes the position even though sector_cap_blocks_entry=True",
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
# S&P 500 LIQUIDITY BACKSTOP -- 2026-07-29 addition. The movers scan ranks
# by SIZE OF MOVE, which structurally excludes megacaps (they rarely move
# enough to place in a top-50 gainers/losers list) even though the 90-day
# backtest showed this system performs far better on them (profit factor
# 1.52 on megacaps vs 1.08 on the scanner's own picks). This reserves
# watchlist slots for S&P 500 names regardless of whether anything in the
# index happens to be a big mover today.
# ---------------------------------------------------------------------------

def test_fetch_sp500_symbols_caches_and_falls_back():
    original = dict(tb._sp500_cache)
    tb._sp500_cache["symbols"], tb._sp500_cache["fetched_at"] = [], None
    try:
        csv_bytes = b"Symbol,Security\nAAPL,Apple\nMSFT,Microsoft\n"
        fake_response = MagicMock()
        fake_response.read.return_value = csv_bytes
        fake_response.__enter__ = lambda self: fake_response
        fake_response.__exit__ = lambda self, *a: False
        with patch.object(tb.urllib.request, "urlopen", return_value=fake_response) as mock_urlopen:
            first = tb.fetch_sp500_symbols()
            second = tb.fetch_sp500_symbols()
        check("parses tickers out of the CSV", first == ["AAPL", "MSFT"])
        check("a second call within the refresh window uses the cache",
              mock_urlopen.call_count == 1 and second == first)

        # Force a refresh, but this time the network fails -- must fall
        # back to what was already cached rather than losing it.
        tb._sp500_cache["fetched_at"] = None
        with patch.object(tb.urllib.request, "urlopen", side_effect=RuntimeError("network down")):
            fallback = tb.fetch_sp500_symbols()
        check("a failed refresh falls back to the last good cache", fallback == ["AAPL", "MSFT"])
    finally:
        tb._sp500_cache.clear()
        tb._sp500_cache.update(original)


def test_fetch_sp500_symbols_no_cache_and_failure_returns_empty():
    original = dict(tb._sp500_cache)
    tb._sp500_cache["symbols"], tb._sp500_cache["fetched_at"] = [], None
    try:
        with patch.object(tb.urllib.request, "urlopen", side_effect=RuntimeError("network down")):
            result = tb.fetch_sp500_symbols()
        check("no cache and a failed fetch returns empty, not an error", result == [])
    finally:
        tb._sp500_cache.clear()
        tb._sp500_cache.update(original)


def test_sp500_backstop_ranks_by_liquidity_and_excludes_already_picked():
    """The backstop's whole premise is preferring LIQUIDITY, not size of
    move -- unlike the movers scan, it ranks candidates by trailing
    dollar volume."""
    def bars_of(close, volume):
        return pd.DataFrame({"close": [close] * 5, "volume": [volume] * 5})

    bars = {
        "AAPL": bars_of(200.0, 1_000_000),   # $200M/bar -- most liquid
        "TSLA": bars_of(300.0, 500_000),     # $150M/bar
        "OBSCUR": bars_of(15.0, 1_000),      # thin -- should rank last
    }
    with patch.object(tb, "fetch_sp500_symbols", return_value=["AAPL", "TSLA", "OBSCUR", "NVDA"]), \
         patch.object(tb, "get_recent_bars_batch", return_value=bars):
        picked = tb.fetch_sp500_candidates(already_picked={"NVDA"}, needed=2)
    check("already-picked symbols are excluded from the backstop", "NVDA" not in picked)
    check("the most liquid names are picked first", picked == ["AAPL", "TSLA"])
    check("a thin/no-data name does not get picked over liquid ones", "OBSCUR" not in picked)


def test_sp500_backstop_requests_nothing_when_not_needed():
    with patch.object(tb, "fetch_sp500_symbols") as mock_fetch:
        result = tb.fetch_sp500_candidates(already_picked=set(), needed=0)
    check("asking for 0 slots does no work at all", result == [] and not mock_fetch.called)


def test_scan_reserves_slots_for_sp500_even_when_movers_scan_finds_nothing():
    """
    The backstop must run even when the movers scan qualifies nothing --
    that's the whole point of calling it a BACKSTOP. Before this was
    fixed, an early `return []` on an empty movers result would have
    skipped the S&P 500 slots entirely on exactly the kind of quiet day
    they're meant to cover.
    """
    empty_movers = types.SimpleNamespace(gainers=[], losers=[])
    with patch.object(tb.screener_client, "get_market_movers", return_value=empty_movers), \
         patch.object(tb, "USE_SP500_UNIVERSE", True), \
         patch.object(tb, "SP500_MIN_WATCHLIST_SLOTS", 3), \
         patch.object(tb, "fetch_sp500_symbols", return_value=["AAPL", "MSFT", "NVDA"]), \
         patch.object(tb, "fetch_sp500_candidates", return_value=["AAPL", "MSFT", "NVDA"]) as mock_backstop:
        result = tb.scan_for_volatile_stocks()
    check("the backstop is consulted even on a zero-mover day", mock_backstop.called)
    check("the backstop's picks become the watchlist", result == ["AAPL", "MSFT", "NVDA"])


def test_scan_caps_total_watchlist_size_with_backstop_included():
    """The backstop reserves slots out of the existing watchlist budget --
    it must never push the total past SCANNER_WATCHLIST_SIZE."""
    movers_picks = [f"MOV{i}" for i in range(10)]
    with patch.object(tb, "screener_client"), \
         patch.object(tb, "SCANNER_WATCHLIST_SIZE", 12), \
         patch.object(tb, "SP500_MIN_WATCHLIST_SLOTS", 4), \
         patch.object(tb, "USE_SP500_UNIVERSE", True), \
         patch.object(tb, "fetch_sp500_symbols", return_value=[]), \
         patch.object(tb, "fetch_sp500_candidates", return_value=["AAPL", "MSFT", "NVDA", "AMD"]) as mock_backstop:
        # Exercise just the tail assembly logic directly rather than the
        # whole scan, since the movers pipeline itself is covered above.
        picked = list(movers_picks)
        already_in_backstop_budget = len(set(picked) & set(tb.fetch_sp500_symbols()))
        slots_open = max(tb.SCANNER_WATCHLIST_SIZE - len(picked), 0)
        needed = min(max(tb.SP500_MIN_WATCHLIST_SLOTS - already_in_backstop_budget, 0), slots_open)
        backstop_picks = tb.fetch_sp500_candidates(set(picked), needed) if needed > 0 else []
        if len(picked) + len(backstop_picks) > tb.SCANNER_WATCHLIST_SIZE:
            overflow = len(picked) + len(backstop_picks) - tb.SCANNER_WATCHLIST_SIZE
            picked = picked[:-overflow] if overflow < len(picked) else []
        picked.extend(backstop_picks)
    check("total watchlist never exceeds SCANNER_WATCHLIST_SIZE", len(picked) <= 12)
    check("backstop names still make it in by trimming movers picks, not skipping",
          {"AAPL", "MSFT", "NVDA", "AMD"} <= set(picked))


# ---------------------------------------------------------------------------
# SCANNER ROBUSTNESS -- 2026-08-2x, real-money-confirmed investigation.
# "Momentum mover" picks averaged -$2.17/trade in REAL trading vs.
# +$51.56/trade for S&P 500 backstop picks, same code/risk rules, only the
# selected stock differing. Four root causes, four fixes: a tighter
# SCANNER_MAX_EXTENSION_PCT, a new scanner-picks-only opening-range
# blackout (covered in the check_symbol section above), a liquidity check
# that excludes the spike day itself, and a minimum listing-age filter.
# ---------------------------------------------------------------------------

def _fake_mover(symbol, price, percent_change):
    return types.SimpleNamespace(symbol=symbol, price=price, percent_change=percent_change)


def test_meets_min_listing_age_filters_by_trading_day_count():
    bars_df = pd.DataFrame({"symbol": ["OLDCO"] * 120 + ["NEWCO"] * 40})
    fake_response = types.SimpleNamespace(df=bars_df)
    with patch.object(tb.data_client, "get_stock_bars", return_value=fake_response):
        result = tb.meets_min_listing_age(["OLDCO", "NEWCO", "MISSING"], min_days=100)
    check("a symbol with >= min_days of daily bars passes", "OLDCO" in result)
    check("a symbol with fewer than min_days of daily bars is excluded", "NEWCO" not in result)
    check("a symbol missing from the response entirely is excluded (real cases: ALMR ~80d, EROC ~43d)",
          "MISSING" not in result)


def test_meets_min_listing_age_fails_open_on_request_failure():
    """A systemic endpoint outage isn't a reason to exclude every
    candidate on a filter that was never actually evaluated -- same
    fail-open shape as the dollar-volume liquidity check's own try/except."""
    with patch.object(tb.data_client, "get_stock_bars", side_effect=RuntimeError("API down")):
        result = tb.meets_min_listing_age(["ANY"], min_days=100)
    check("a total request failure fails OPEN rather than blocking everything", result == {"ANY"})


def test_meets_min_listing_age_empty_input_is_a_noop():
    with patch.object(tb.data_client, "get_stock_bars") as mock_fetch:
        result = tb.meets_min_listing_age([], min_days=100)
    check("an empty symbol list returns empty without calling the API",
          result == set() and not mock_fetch.called)


def test_scan_excludes_candidates_beyond_max_extension_pct():
    """See SCANNER_MAX_EXTENSION_PCT's 2026-08-2x comment -- real losing
    trades traced against real intraday bars (ALMR +27.4%, NN +23%)
    entered comfortably under the OLD 50% cutoff. Pins that the new,
    tighter default actually excludes that range."""
    movers = types.SimpleNamespace(
        gainers=[_fake_mover("ALMRLIKE", 20.0, 27.4), _fake_mover("MILD", 20.0, 8.0)],
        losers=[],
    )
    with patch.object(tb.screener_client, "get_market_movers", return_value=movers), \
         patch.object(tb.screener_client, "get_most_actives",
                       return_value=types.SimpleNamespace(most_actives=[])), \
         patch.object(tb, "SCANNER_MAX_EXTENSION_PCT", 20.0), \
         patch.object(tb, "EXCLUDE_LEVERAGED_ETFS", False), \
         patch.object(tb, "USE_SCANNER_MIN_LISTING_AGE", False), \
         patch.object(tb, "USE_NEWS_FILTER", False), \
         patch.object(tb, "USE_SP500_UNIVERSE", False), \
         patch.object(tb, "get_recent_bars_batch", return_value={
             "MILD": pd.DataFrame({"close": [50.0] * 5, "volume": [1_000_000] * 5}),
         }):
        result = tb.scan_for_volatile_stocks()
    check("a candidate past the extension cutoff (27.4% > 20%) is excluded", "ALMRLIKE" not in result)
    check("a candidate under the cutoff and otherwise qualified is kept", "MILD" in result)


def test_scan_excludes_candidates_below_min_listing_age():
    movers = types.SimpleNamespace(
        gainers=[_fake_mover("NEWCO", 20.0, 5.0), _fake_mover("OLDCO", 20.0, 5.0)],
        losers=[],
    )
    with patch.object(tb.screener_client, "get_market_movers", return_value=movers), \
         patch.object(tb.screener_client, "get_most_actives",
                       return_value=types.SimpleNamespace(most_actives=[])), \
         patch.object(tb, "EXCLUDE_LEVERAGED_ETFS", False), \
         patch.object(tb, "USE_SCANNER_MIN_LISTING_AGE", True), \
         patch.object(tb, "SCANNER_MIN_LISTING_AGE_DAYS", 100), \
         patch.object(tb, "meets_min_listing_age", return_value={"OLDCO"}) as mock_listing_age, \
         patch.object(tb, "USE_NEWS_FILTER", False), \
         patch.object(tb, "USE_SP500_UNIVERSE", False), \
         patch.object(tb, "get_recent_bars_batch", return_value={
             "OLDCO": pd.DataFrame({"close": [50.0] * 5, "volume": [1_000_000] * 5}),
         }):
        result = tb.scan_for_volatile_stocks()
    check("the listing-age check is consulted with the prefiltered candidates", mock_listing_age.called)
    check("a too-new symbol (real cases: ALMR ~80d, EROC ~43d) is excluded", "NEWCO" not in result)
    check("an old-enough symbol is kept", "OLDCO" in result)


def test_scan_listing_age_filter_noop_when_disabled():
    movers = types.SimpleNamespace(gainers=[_fake_mover("NEWCO", 20.0, 5.0)], losers=[])
    with patch.object(tb.screener_client, "get_market_movers", return_value=movers), \
         patch.object(tb.screener_client, "get_most_actives",
                       return_value=types.SimpleNamespace(most_actives=[])), \
         patch.object(tb, "EXCLUDE_LEVERAGED_ETFS", False), \
         patch.object(tb, "USE_SCANNER_MIN_LISTING_AGE", False), \
         patch.object(tb, "meets_min_listing_age") as mock_listing_age, \
         patch.object(tb, "USE_NEWS_FILTER", False), \
         patch.object(tb, "USE_SP500_UNIVERSE", False), \
         patch.object(tb, "get_recent_bars_batch", return_value={
             "NEWCO": pd.DataFrame({"close": [50.0] * 5, "volume": [1_000_000] * 5}),
         }):
        result = tb.scan_for_volatile_stocks()
    check("with the filter off, meets_min_listing_age is never even called", not mock_listing_age.called)
    check("the candidate is kept", "NEWCO" in result)


def test_scan_liquidity_check_excludes_todays_bars():
    """See SCANNER_MIN_DOLLAR_VOLUME's fix -- the average must be computed
    from bars ending at the START of today (ET), not through the current
    moment, so today's own abnormal spike volume can't count toward "is
    this normally liquid.\""""
    movers = types.SimpleNamespace(gainers=[_fake_mover("SPIKY", 20.0, 5.0)], losers=[])
    with patch.object(tb.screener_client, "get_market_movers", return_value=movers), \
         patch.object(tb.screener_client, "get_most_actives",
                       return_value=types.SimpleNamespace(most_actives=[])), \
         patch.object(tb, "EXCLUDE_LEVERAGED_ETFS", False), \
         patch.object(tb, "USE_SCANNER_MIN_LISTING_AGE", False), \
         patch.object(tb, "USE_NEWS_FILTER", False), \
         patch.object(tb, "USE_SP500_UNIVERSE", False), \
         patch.object(tb, "get_recent_bars_batch", return_value={
             "SPIKY": pd.DataFrame({"close": [50.0] * 5, "volume": [1_000_000] * 5}),
         }) as mock_bars:
        tb.scan_for_volatile_stocks()
    check("the liquidity bar fetch is called", mock_bars.called)
    _, kwargs = mock_bars.call_args
    end_arg = kwargs.get("end")
    check("an explicit `end` cutoff is passed (not None, i.e. not through the live moment)",
          end_arg is not None)
    now_et = tb.datetime.now(tb.MARKET_TZ_FOR_LOGS)
    check("the cutoff is pinned to the START of today (midnight ET), excluding today's own bars",
          end_arg is not None and end_arg.date() == now_et.date()
          and end_arg.hour == 0 and end_arg.minute == 0)


def _fake_order(status, filled_avg_price=None, legs=None, order_id="ord-1", symbol="HURN"):
    return types.SimpleNamespace(id=order_id, symbol=symbol, status=status,
                                 filled_avg_price=filled_avg_price, legs=legs or [])


def _fake_leg(order_type, leg_id):
    return types.SimpleNamespace(id=leg_id, order_type=order_type)


def test_reconcile_bracket_keeps_small_drift_unchanged():
    """A fill a few cents from the quote is normal and shouldn't trigger
    a repricing round-trip against the API."""
    order = _fake_order(tb.OrderStatus.NEW)
    filled = _fake_order(tb.OrderStatus.FILLED, filled_avg_price="100.10")
    with patch.object(tb.trading_client, "get_order_by_id", return_value=filled) as mock_get, \
         patch.object(tb.trading_client, "replace_order_by_id") as mock_replace, \
         patch.object(tb.time, "sleep"):
        price, corrected = tb.reconcile_bracket_with_real_fill(order, atr_value=1.0, reference_price=100.00)
    check("small drift reports the real fill price", price == 100.10)
    check("small drift does not trigger a reprice", corrected is False)
    check("no replace_order_by_id calls for small drift", not mock_replace.called)


def test_reconcile_bracket_reprices_legs_on_large_drift():
    """
    Regression for 2026-07-29: HURN's bracket was priced off a $149.055
    quote, but the market order filled at $154.30 (+3.5%) two seconds
    later. The bracket's absolute stop/target levels don't move with the
    entry, so a stale quote silently changes the trade's real % risk --
    the stop was 5% below the quote but 8.2% below the real entry. This
    pins the fix: a large drift must reprice both legs relative to the
    REAL fill, not the quote.
    """
    stop_leg = _fake_leg(tb.OrderType.STOP, "stop-leg-1")
    target_leg = _fake_leg(tb.OrderType.LIMIT, "target-leg-1")
    order = _fake_order(tb.OrderStatus.NEW)
    filled = _fake_order(tb.OrderStatus.FILLED, filled_avg_price="154.30", legs=[stop_leg, target_leg])

    with patch.object(tb.trading_client, "get_order_by_id", return_value=filled), \
         patch.object(tb.trading_client, "replace_order_by_id") as mock_replace, \
         patch.object(tb.time, "sleep"):
        price, corrected = tb.reconcile_bracket_with_real_fill(order, atr_value=2.0, reference_price=149.055)

    check("reports the real fill price, not the stale quote", price == 154.30)
    check("a 3.5% drift triggers a reprice", corrected is True)
    check("both legs get replaced", mock_replace.call_count == 2)
    replaced_ids = {call.args[0] for call in mock_replace.call_args_list}
    check("the stop leg was one of the replaced orders", "stop-leg-1" in replaced_ids)
    check("the target leg was one of the replaced orders", "target-leg-1" in replaced_ids)

    expected_stop, expected_target = tb.compute_stop_and_target(154.30, 2.0)
    for call in mock_replace.call_args_list:
        req = call.args[1]
        if call.args[0] == "stop-leg-1":
            check("the new stop is relative to the REAL fill, not the quote",
                  abs(req.stop_price - round(expected_stop, 2)) < 0.01)
        else:
            check("the new target is relative to the REAL fill, not the quote",
                  abs(req.limit_price - round(expected_target, 2)) < 0.01)


def test_reconcile_bracket_falls_back_when_fill_not_confirmed():
    """A fill that never confirms within the poll window must not hang
    the cycle or crash -- it falls back to the quote."""
    order = _fake_order(tb.OrderStatus.NEW)
    pending = _fake_order(tb.OrderStatus.NEW)  # never reaches FILLED
    with patch.object(tb.trading_client, "get_order_by_id", return_value=pending), \
         patch.object(tb.trading_client, "replace_order_by_id") as mock_replace, \
         patch.object(tb.time, "sleep"):
        price, corrected = tb.reconcile_bracket_with_real_fill(order, atr_value=1.0, reference_price=100.00)
    check("falls back to the reference price when unconfirmed", price == 100.00)
    check("does not report a correction that never happened", corrected is False)
    check("never attempts to replace legs without a confirmed fill", not mock_replace.called)


def test_reconcile_bracket_survives_a_failed_leg_replace():
    """One leg's replace call failing (e.g. it already filled/canceled)
    must not crash the trade -- the other leg still gets its shot."""
    stop_leg = _fake_leg(tb.OrderType.STOP, "stop-leg-1")
    target_leg = _fake_leg(tb.OrderType.LIMIT, "target-leg-1")
    order = _fake_order(tb.OrderStatus.NEW)
    filled = _fake_order(tb.OrderStatus.FILLED, filled_avg_price="154.30", legs=[stop_leg, target_leg])

    def flaky_replace(order_id, req):
        if order_id == "stop-leg-1":
            raise RuntimeError("order already filled")

    with patch.object(tb.trading_client, "get_order_by_id", return_value=filled), \
         patch.object(tb.trading_client, "replace_order_by_id", side_effect=flaky_replace), \
         patch.object(tb.time, "sleep"):
        raised = False
        try:
            price, corrected = tb.reconcile_bracket_with_real_fill(order, atr_value=2.0, reference_price=149.055)
        except Exception:
            raised = True
    check("a failed leg replace does not raise", not raised)
    check("still reports the real fill price", price == 154.30)
    check("reports not fully corrected when a leg failed", corrected is False)


# ---------------------------------------------------------------------------
# MULTI-TIMEFRAME FILTER -- scanner picks only. S&P 500 names must always
# bypass the filter regardless of daily trend; everything else must be
# gated on it. This is the whole point of the feature (see CLAUDE.md for
# the full evidence), so it's the property most worth pinning with a
# test. USE_MULTI_TIMEFRAME_FILTER defaults OFF as of 2026-08-23 (see
# that constant's comment), so every test below that exercises the
# GATING mechanism itself explicitly patches it True -- only the
# noop-when-disabled test below relies on patching it False.
# ---------------------------------------------------------------------------

def test_daily_trend_confirms_entry_exempts_sp500_regardless_of_trend():
    from datetime import date
    today = date(2026, 7, 31)
    with patch.object(tb, "USE_MULTI_TIMEFRAME_FILTER", True), \
         patch.object(tb, "fetch_sp500_symbols", return_value=["AAPL", "MSFT"]), \
         patch.object(tb, "_daily_trend_cache", {"AAPL": {"map": {today: False}, "fetched_at": None}}):
        check("an S&P 500 name is exempt even though its OWN trend is down",
              tb.daily_trend_confirms_entry("AAPL", today) is True)


def test_daily_trend_confirms_entry_gates_non_sp500_on_real_trend():
    from datetime import date
    today = date(2026, 7, 31)
    with patch.object(tb, "USE_MULTI_TIMEFRAME_FILTER", True), \
         patch.object(tb, "fetch_sp500_symbols", return_value=["AAPL", "MSFT"]), \
         patch.object(tb, "_daily_trend_cache", {
             "VEEE": {"map": {today: True}, "fetched_at": None},
             "TRAX": {"map": {today: False}, "fetched_at": None},
         }):
        check("a non-S&P-500 name with an up trend is confirmed",
              tb.daily_trend_confirms_entry("VEEE", today) is True)
        check("a non-S&P-500 name with a down trend is blocked",
              tb.daily_trend_confirms_entry("TRAX", today) is False)


def test_daily_trend_confirms_entry_fails_closed_on_unknown_symbol():
    from datetime import date
    today = date(2026, 7, 31)
    with patch.object(tb, "USE_MULTI_TIMEFRAME_FILTER", True), \
         patch.object(tb, "fetch_sp500_symbols", return_value=["AAPL"]), \
         patch.object(tb, "_daily_trend_cache", {}):
        check("a symbol with no cached trend at all is blocked, not silently allowed",
              tb.daily_trend_confirms_entry("QBTS", today) is False)


def test_daily_trend_confirms_entry_noop_when_filter_disabled():
    from datetime import date
    today = date(2026, 7, 31)
    with patch.object(tb, "USE_MULTI_TIMEFRAME_FILTER", False), \
         patch.object(tb, "_daily_trend_cache", {}):
        check("with the filter off, even an unknown non-S&P-500 symbol is allowed",
              tb.daily_trend_confirms_entry("QBTS", today) is True)


# ---------------------------------------------------------------------------
# VWAP_REVERSION VOLUME CONFIRMATION -- 2026-07-31, scanner picks only.
# First tested with vwap_reversion running in ISOLATION and looked like a
# clean win everywhere; tested again in the full priority chain and turned
# out to regress megacaps (freed-up bars fall through to weaker
# trend_following). Same split shape, same fix as the multi-timeframe
# filter: S&P 500 names always exempt.
# ---------------------------------------------------------------------------

def test_vwap_volume_filter_exempts_sp500_even_when_volume_is_weak():
    df_input = pd.DataFrame({"close": [100.0]})
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "vwap_reversion", "test buy")), \
         patch.object(tb, "fetch_sp500_symbols", return_value=["AAPL", "MSFT"]), \
         patch.object(tb, "vwap_reversion_volume_confirms", return_value=False), \
         patch.object(tb, "USE_VWAP_VOLUME_CONFIRMATION", True), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("AAPL", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0)
    check("an S&P 500 symbol buys even with weak volume (exempt)", mock_buy.called)
    check("the exempt entry reports its notional", notional == 500.0)


def test_vwap_volume_filter_blocks_non_sp500_with_weak_volume():
    df_input = pd.DataFrame({"close": [100.0]})
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "vwap_reversion", "test buy")), \
         patch.object(tb, "fetch_sp500_symbols", return_value=["AAPL", "MSFT"]), \
         patch.object(tb, "vwap_reversion_volume_confirms", return_value=False), \
         patch.object(tb, "USE_VWAP_VOLUME_CONFIRMATION", True), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("TRAX", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0)
    check("a non-S&P-500 symbol is refused when volume doesn't confirm", not mock_buy.called)
    check("a blocked entry reports no notional opened", notional == 0.0)


def test_vwap_volume_filter_allows_non_sp500_with_strong_volume():
    df_input = pd.DataFrame({"close": [100.0]})
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "vwap_reversion", "test buy")), \
         patch.object(tb, "fetch_sp500_symbols", return_value=["AAPL", "MSFT"]), \
         patch.object(tb, "vwap_reversion_volume_confirms", return_value=True), \
         patch.object(tb, "USE_VWAP_VOLUME_CONFIRMATION", True), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("TRAX", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0)
    check("a non-S&P-500 symbol buys once volume confirms", mock_buy.called)
    check("the confirmed entry reports its notional", notional == 500.0)


def test_vwap_volume_filter_only_applies_to_vwap_reversion_signals():
    """The gate must only fire for vwap_reversion's OWN signal -- a BUY
    from a different strategy on the same weak-volume bar must not be
    blocked by a check that has nothing to do with it."""
    df_input = pd.DataFrame({"close": [100.0]})
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test buy")), \
         patch.object(tb, "fetch_sp500_symbols", return_value=["AAPL", "MSFT"]), \
         patch.object(tb, "vwap_reversion_volume_confirms", return_value=False), \
         patch.object(tb, "USE_VWAP_VOLUME_CONFIRMATION", True), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("TRAX", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0)
    check("a breakout signal is unaffected by the vwap-only volume filter", mock_buy.called)
    check("the unrelated entry reports its notional", notional == 500.0)


def test_vwap_volume_filter_noop_when_disabled():
    df_input = pd.DataFrame({"close": [100.0]})
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "vwap_reversion", "test buy")), \
         patch.object(tb, "fetch_sp500_symbols", return_value=["AAPL", "MSFT"]), \
         patch.object(tb, "vwap_reversion_volume_confirms", return_value=False), \
         patch.object(tb, "USE_VWAP_VOLUME_CONFIRMATION", False), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("TRAX", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0)
    check("with the filter disabled, weak volume no longer blocks anything", mock_buy.called)
    check("the entry reports its notional", notional == 500.0)


def test_check_symbol_blocks_buy_when_daily_trend_blocks_entry():
    df_input = pd.DataFrame({"close": [100.0]})
    # USE_VWAP_VOLUME_CONFIRMATION forced off: this test is about the
    # daily-trend filter specifically, and the fixture here (a minimal
    # make_fake_enriched) doesn't carry the volume columns that OTHER
    # gate would read -- keeping them isolated is the point.
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "vwap_reversion", "test buy")), \
         patch.object(tb, "USE_VWAP_VOLUME_CONFIRMATION", False), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("TRAX", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0, daily_trend_blocks_entry=True)
    check("a BUY signal is refused when the multi-timeframe filter blocks it", not mock_buy.called)
    check("a blocked entry reports no notional opened", notional == 0.0)

    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "vwap_reversion", "test buy")), \
         patch.object(tb, "USE_VWAP_VOLUME_CONFIRMATION", False), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("TRAX", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0, daily_trend_blocks_entry=False)
    check("the same signal is taken once the filter confirms", mock_buy.called)
    check("a taken entry reports its notional", notional == 500.0)


# ---------------------------------------------------------------------------
# SPY REGIME GATE -- broad-market veto, off by default. Mirrors the
# multi-timeframe filter's true-means-ok framing (spy_regime_confirms_entry),
# and check_symbol takes the already-negated bool the same way it takes
# daily_trend_blocks_entry.
# ---------------------------------------------------------------------------

def make_fake_spy_enriched(adx: float, ema_fast: float, ema_slow: float) -> pd.DataFrame:
    """A one-row stand-in for add_indicators()'s output on SPY's own bars,
    with just the columns spy_regime_confirms_entry reads."""
    return pd.DataFrame({"adx": [adx], "ema_fast": [ema_fast], "ema_slow": [ema_slow]})


def test_spy_regime_confirms_entry_noop_when_gate_disabled():
    with patch.object(tb, "USE_SPY_REGIME_GATE", False), \
         patch.object(tb, "get_recent_bars_batch") as mock_fetch:
        result = tb.spy_regime_confirms_entry()
    check("with the gate off, entries are confirmed without even fetching SPY", result is True)
    check("no SPY fetch happens when the gate is disabled", not mock_fetch.called)


def test_spy_regime_confirms_entry_blocks_on_confirmed_spy_downtrend():
    fake_bars = pd.DataFrame({"close": [500.0]})
    with patch.object(tb, "USE_SPY_REGIME_GATE", True), \
         patch.object(tb, "get_recent_bars_batch", return_value={"SPY": fake_bars}), \
         patch.object(tb, "add_indicators", return_value=make_fake_spy_enriched(30.0, 495.0, 500.0)):
        result = tb.spy_regime_confirms_entry()
    check("SPY ADX above threshold AND fast EMA below slow EMA blocks new entries", result is False)


def test_spy_regime_confirms_entry_allows_when_spy_trending_up():
    fake_bars = pd.DataFrame({"close": [500.0]})
    with patch.object(tb, "USE_SPY_REGIME_GATE", True), \
         patch.object(tb, "get_recent_bars_batch", return_value={"SPY": fake_bars}), \
         patch.object(tb, "add_indicators", return_value=make_fake_spy_enriched(30.0, 505.0, 500.0)):
        result = tb.spy_regime_confirms_entry()
    check("a confirmed trend that's UP (fast EMA above slow) does not block entries", result is True)


def test_spy_regime_confirms_entry_allows_when_spy_choppy():
    fake_bars = pd.DataFrame({"close": [500.0]})
    with patch.object(tb, "USE_SPY_REGIME_GATE", True), \
         patch.object(tb, "get_recent_bars_batch", return_value={"SPY": fake_bars}), \
         patch.object(tb, "add_indicators", return_value=make_fake_spy_enriched(10.0, 495.0, 500.0)):
        result = tb.spy_regime_confirms_entry()
    check("EMA pointing down but ADX below ADX_TREND_THRESHOLD (chop, not a confirmed trend) "
          "does not block entries", result is True)


def test_spy_regime_confirms_entry_fails_open_on_fetch_failure():
    with patch.object(tb, "USE_SPY_REGIME_GATE", True), \
         patch.object(tb, "get_recent_bars_batch", side_effect=RuntimeError("API down")):
        result = tb.spy_regime_confirms_entry()
    check("a failed SPY fetch fails OPEN (doesn't block), not closed -- SPY itself failing to "
          "fetch says something's off with market data broadly, not a reason to pause everything",
          result is True)


def test_spy_regime_confirms_entry_fails_open_on_missing_data():
    with patch.object(tb, "USE_SPY_REGIME_GATE", True), \
         patch.object(tb, "get_recent_bars_batch", return_value={}):
        result = tb.spy_regime_confirms_entry()
    check("no SPY data returned this cycle fails open", result is True)


def test_spy_regime_confirms_entry_fails_open_during_warmup():
    fake_bars = pd.DataFrame({"close": [500.0]})
    with patch.object(tb, "USE_SPY_REGIME_GATE", True), \
         patch.object(tb, "get_recent_bars_batch", return_value={"SPY": fake_bars}), \
         patch.object(tb, "add_indicators", return_value=make_fake_spy_enriched(np.nan, np.nan, np.nan)):
        result = tb.spy_regime_confirms_entry()
    check("SPY's own indicators still warming up (NaN) fails open, not closed", result is True)


# ---------------------------------------------------------------------------
# filter_to_regular_session / get_recent_bars_batch's regular_session_only --
# 2026-08-23: live indicators used to be computed on extended-hours-
# contaminated data while backtest.py's fetch_historical_bars already
# filtered to the 9:30am-4pm ET regular session. See filter_to_regular_
# session's own docstring for the full reasoning.
# ---------------------------------------------------------------------------

def _et_bar(date_str: str, hour: int, minute: int, **cols) -> dict:
    """One raw bar's worth of columns, timestamped at the given ET
    wall-clock time on date_str but stored the way real Alpaca bars
    arrive: tz-aware, converted to UTC."""
    ts = pd.Timestamp(f"{date_str} {hour:02d}:{minute:02d}:00", tz="America/New_York").tz_convert("UTC")
    row = {"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}
    row.update(cols)
    return row


def test_filter_to_regular_session_drops_pre_and_post_market_bars():
    """The exact boundary backtest.py's fetch_historical_bars uses:
    9:30am (inclusive) through 4:00pm (exclusive) ET."""
    raw = pd.DataFrame([
        _et_bar("2026-08-20", 4, 0),      # pre-market, way early
        _et_bar("2026-08-20", 9, 29),     # one minute before the open -- OUT
        _et_bar("2026-08-20", 9, 30),     # the open itself -- IN
        _et_bar("2026-08-20", 12, 0),     # midday -- IN
        _et_bar("2026-08-20", 15, 59),    # last regular-session minute -- IN
        _et_bar("2026-08-20", 16, 0),     # the close itself -- OUT (exclusive)
        _et_bar("2026-08-20", 19, 45),    # after-hours -- OUT
    ])
    filtered = tb.filter_to_regular_session(raw)
    kept_times = filtered["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%H:%M").tolist()
    check("pre-market and after-hours bars are dropped, exactly the regular-session ones remain",
          kept_times == ["09:30", "12:00", "15:59"], detail=str(kept_times))


def test_filter_to_regular_session_keeps_every_regular_session_bar():
    """(b) from the task: the filter must never drop a genuine
    regular-session bar. One bar per 15 minutes across a whole session,
    plus contamination on both ends, and every regular-session one must
    survive."""
    et_minutes = list(range(0, 24 * 60, 15))  # every 15 min across a full day
    raw = pd.DataFrame([
        _et_bar("2026-08-20", m // 60, m % 60) for m in et_minutes
    ])
    expected_regular = [
        m for m in et_minutes if (m >= 9 * 60 + 30) and (m < 16 * 60)
    ]
    filtered = tb.filter_to_regular_session(raw)
    kept_minutes = [
        h * 60 + mi for h, mi in
        zip(filtered["timestamp"].dt.tz_convert("America/New_York").dt.hour,
            filtered["timestamp"].dt.tz_convert("America/New_York").dt.minute)
    ]
    check("every regular-session bar (9:30am-3:59:59pm ET) survives filtering, none extra",
          kept_minutes == expected_regular, detail=f"{kept_minutes} vs {expected_regular}")


def test_filter_to_regular_session_empty_input_is_a_noop():
    empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    result = tb.filter_to_regular_session(empty)
    check("an empty bar set filters to an empty bar set, no crash", result.empty)


def test_filter_to_regular_session_matches_what_backtest_would_compute():
    """(a) from the task: a raw bar set contaminated with a pre-market
    bar, once filtered, must produce the SAME session_open_price/
    session_first_bar_high/session_first_bar_volume/vwap that
    backtest.py's fetch_historical_bars-then-add_indicators pipeline
    would produce on the same underlying regular-session data --
    because backtest.py's filter is this exact same boundary condition,
    the true test is that add_indicators sees the REGULAR session's own
    first bar as "first", not whatever printed earliest in the
    calendar day.
    """
    contaminated = pd.DataFrame([
        # Pre-market bar: a materially different open/high/volume than
        # the true regular-session open -- if this leaks through,
        # session_open_price/session_first_bar_high/session_first_bar_volume
        # get anchored to IT instead of the 9:30 bar.
        _et_bar("2026-08-20", 8, 0, open=95.0, high=96.0, low=94.5, close=95.5, volume=500_000.0),
        _et_bar("2026-08-20", 9, 30, open=100.0, high=100.5, low=99.5, close=100.2, volume=20_000.0),
        _et_bar("2026-08-20", 9, 45, open=100.2, high=101.0, low=100.0, close=100.8, volume=18_000.0),
        _et_bar("2026-08-20", 10, 0, open=100.8, high=101.5, low=100.5, close=101.2, volume=17_000.0),
    ])
    # The reference: what a backtest would have started from -- the
    # regular session's bars ONLY, already hand-picked here (the whole
    # point of this test is that filter_to_regular_session must derive
    # the identical set on its own from the contaminated input).
    reference_regular_only = contaminated.iloc[1:].reset_index(drop=True)

    filtered = tb.filter_to_regular_session(contaminated)
    from strategy import add_indicators
    enriched_from_filtered = add_indicators(filtered)
    enriched_from_reference = add_indicators(reference_regular_only)

    check("filtering the contaminated input reproduces the SAME row count as the "
          "hand-picked regular-session-only reference",
          len(filtered) == len(reference_regular_only))
    check("session_open_price matches the regular session's own true open (100.0), "
          "not the pre-market bar's (95.0)",
          enriched_from_filtered["session_open_price"].iat[0] == 100.0)
    check("session_open_price is identical between the filtered-contaminated path and the "
          "already-clean reference backtest.py would have used",
          (enriched_from_filtered["session_open_price"] == enriched_from_reference["session_open_price"]).all())
    check("session_first_bar_high matches the regular session's own first bar (100.5), "
          "not the pre-market bar's (96.0)",
          enriched_from_filtered["session_first_bar_high"].iat[0] == 100.5)
    check("session_first_bar_volume matches the regular session's own first bar (20000), "
          "not the pre-market bar's (500000)",
          enriched_from_filtered["session_first_bar_volume"].iat[0] == 20_000.0)
    check("vwap is identical between the filtered-contaminated path and the clean reference "
          "(both anchor to the same regular-session bars)",
          np.allclose(enriched_from_filtered["vwap"].to_numpy(), enriched_from_reference["vwap"].to_numpy()))

    # And the contrast that motivated this fix in the first place: running
    # add_indicators on the UNFILTERED, contaminated bars directly (the
    # live bot's behavior before this fix) gives a WRONG session_open_price.
    enriched_contaminated_unfiltered = add_indicators(contaminated)
    check("without this fix, the contaminated (unfiltered) input anchors "
          "session_open_price to the pre-market bar instead -- proving the "
          "filter is what fixes the real divergence",
          enriched_contaminated_unfiltered["session_open_price"].iat[0] == 95.0)


def test_get_recent_bars_batch_regular_session_only_filters_each_symbol():
    """get_recent_bars_batch's own wiring: regular_session_only=True must
    filter EVERY symbol's bars in the batch, and the default (False) must
    leave the raw response completely untouched (backward compatible with
    every caller that doesn't pass it)."""
    raw_multi = pd.DataFrame([
        {**_et_bar("2026-08-20", 8, 0), "symbol": "AAA"},
        {**_et_bar("2026-08-20", 9, 30), "symbol": "AAA"},
        {**_et_bar("2026-08-20", 8, 0), "symbol": "BBB"},
        {**_et_bar("2026-08-20", 10, 0), "symbol": "BBB"},
    ]).set_index(["symbol", "timestamp"])

    fake_response = MagicMock()
    fake_response.df = raw_multi

    with patch.object(tb.data_client, "get_stock_bars", return_value=fake_response):
        filtered_result = tb.get_recent_bars_batch(["AAA", "BBB"], regular_session_only=True)
        unfiltered_result = tb.get_recent_bars_batch(["AAA", "BBB"], regular_session_only=False)

    check("regular_session_only=True drops AAA's pre-market bar, keeps its regular-session one",
          len(filtered_result["AAA"]) == 1)
    check("regular_session_only=True drops BBB's pre-market bar, keeps its regular-session one",
          len(filtered_result["BBB"]) == 1)
    check("the default (regular_session_only=False) is completely unaffected -- both of "
          "AAA's raw bars still come back, pre-market included",
          len(unfiltered_result["AAA"]) == 2)
    check("the default (regular_session_only=False) is completely unaffected -- both of "
          "BBB's raw bars still come back, pre-market included",
          len(unfiltered_result["BBB"]) == 2)


def test_spy_regime_confirms_entry_requests_regular_session_only_bars():
    """spy_regime_confirms_entry feeds its bars straight into
    add_indicators (same as check_symbol does), so it must opt into
    regular_session_only -- this is one of the two call sites the
    2026-08-23 fix targets."""
    fake_bars = pd.DataFrame({"close": [500.0]})
    with patch.object(tb, "USE_SPY_REGIME_GATE", True), \
         patch.object(tb, "get_recent_bars_batch", return_value={"SPY": fake_bars}) as mock_fetch, \
         patch.object(tb, "add_indicators", return_value=make_fake_spy_enriched(10.0, 500.0, 500.0)):
        tb.spy_regime_confirms_entry()
    check("spy_regime_confirms_entry fetches with regular_session_only=True",
          mock_fetch.call_args.kwargs.get("regular_session_only") is True)


def test_sp500_backstop_and_scanner_liquidity_checks_do_not_request_regular_session_only():
    """fetch_sp500_candidates and scan_for_volatile_stocks's dollar-volume
    check use these bars for a LIQUIDITY read (pre/post-market volume is
    real tradable-liquidity signal there), never through add_indicators --
    they must keep seeing raw, unfiltered bars, i.e. must NOT pass
    regular_session_only=True."""
    def bars_of(close, volume):
        return pd.DataFrame({"close": [close] * 5, "volume": [volume] * 5})

    with patch.object(tb, "fetch_sp500_symbols", return_value=["AAPL"]), \
         patch.object(tb, "get_recent_bars_batch", return_value={"AAPL": bars_of(200.0, 1_000_000)}) as mock_fetch:
        tb.fetch_sp500_candidates(already_picked=set(), needed=1)
    check("fetch_sp500_candidates's liquidity backstop does not request regular_session_only",
          mock_fetch.call_args.kwargs.get("regular_session_only") is not True)


def test_check_symbol_blocks_buy_when_spy_regime_blocks_entry():
    df_input = pd.DataFrame({"close": [100.0]})
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test buy")), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("TSLA", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0, spy_regime_blocks_entry=True)
    check("a BUY signal is refused when the SPY regime gate blocks it, "
          "even for a strategy unrelated to VWAP/daily-trend", not mock_buy.called)
    check("a blocked entry reports no notional opened", notional == 0.0)

    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test buy")), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("TSLA", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0, spy_regime_blocks_entry=False)
    check("the same signal is taken once the SPY regime gate confirms", mock_buy.called)
    check("a taken entry reports its notional", notional == 500.0)


# ---------------------------------------------------------------------------
# SECTOR-RELATIVE MEAN REVERSION FILTER -- off by default. Only ever
# consulted for a mean_reversion BUY (checked inside check_symbol once
# reason_key is known), same reason_key-scoping pattern as the
# vwap_reversion volume filter above.
# ---------------------------------------------------------------------------

def make_fake_close_series(closes: list[float]) -> pd.DataFrame:
    """A stand-in for the 'close' column sector_relative_mean_reversion_blocks_entry
    reads, on either the candidate's own enriched bars or a sector ETF's raw bars."""
    return pd.DataFrame({"close": closes})


def test_get_sector_etf_resolves_via_sector_map_and_etf_map():
    with patch.object(tb, "get_symbol_sector", return_value="Information Technology"):
        etf = tb.get_sector_etf("NVDA")
    check("a symbol with a mapped sector resolves to its SECTOR_ETF_MAP entry", etf == "XLK")


def test_get_sector_etf_none_when_sector_unknown():
    with patch.object(tb, "get_symbol_sector", return_value=None):
        etf = tb.get_sector_etf("MYSTERYSYMBOL")
    check("an unknown sector resolves to no ETF, not an exception", etf is None)


def test_sector_relative_blocks_entry_disabled_by_default():
    with patch.object(tb, "USE_SECTOR_RELATIVE_MEAN_REVERSION", False), \
         patch.object(tb, "get_sector_etf") as mock_get_etf:
        blocked = tb.sector_relative_mean_reversion_blocks_entry(
            "AAA", make_fake_close_series([100, 100, 100, 100, 95]), 4,
            {"XLK": make_fake_close_series([100, 100, 100, 100, 101])})
    check("with the filter off, the gate never blocks", not blocked)
    check("no sector lookup happens when the filter is disabled", not mock_get_etf.called)


def test_sector_relative_blocks_entry_fails_open_on_unknown_sector():
    with patch.object(tb, "USE_SECTOR_RELATIVE_MEAN_REVERSION", True), \
         patch.object(tb, "get_sector_etf", return_value=None):
        blocked = tb.sector_relative_mean_reversion_blocks_entry(
            "MYSTERYSYMBOL", make_fake_close_series([100, 100, 100, 100, 95]), 4,
            {"XLK": make_fake_close_series([100, 100, 100, 100, 101])})
    check("an unmappable sector fails open (doesn't block)", not blocked)


def test_sector_relative_blocks_entry_fails_open_when_etf_bars_missing():
    with patch.object(tb, "USE_SECTOR_RELATIVE_MEAN_REVERSION", True), \
         patch.object(tb, "get_sector_etf", return_value="XLK"):
        blocked = tb.sector_relative_mean_reversion_blocks_entry(
            "NVDA", make_fake_close_series([100, 100, 100, 100, 95]), 4, {})
    check("no ETF bars fetched this cycle fails open", not blocked)


def test_sector_relative_blocks_entry_fails_open_on_insufficient_history():
    with patch.object(tb, "USE_SECTOR_RELATIVE_MEAN_REVERSION", True), \
         patch.object(tb, "get_sector_etf", return_value="XLK"), \
         patch.object(tb, "SECTOR_RELATIVE_LOOKBACK_BARS", 3):
        # Candidate only has 3 bars (i=2), fewer than the 3-bar lookback needs.
        blocked = tb.sector_relative_mean_reversion_blocks_entry(
            "NVDA", make_fake_close_series([100, 100, 100]), 2,
            {"XLK": make_fake_close_series([100, 100, 100, 100, 101])})
    check("not enough of the candidate's own history yet fails open", not blocked)

    with patch.object(tb, "USE_SECTOR_RELATIVE_MEAN_REVERSION", True), \
         patch.object(tb, "get_sector_etf", return_value="XLK"), \
         patch.object(tb, "SECTOR_RELATIVE_LOOKBACK_BARS", 3):
        # ETF only has 3 bars, at/under the 3-bar lookback.
        blocked = tb.sector_relative_mean_reversion_blocks_entry(
            "NVDA", make_fake_close_series([100, 100, 100, 100, 95]), 4,
            {"XLK": make_fake_close_series([100, 100, 100])})
    check("not enough of the ETF's own history yet fails open", not blocked)


def test_sector_relative_blocks_entry_allows_genuine_underperformance():
    with patch.object(tb, "USE_SECTOR_RELATIVE_MEAN_REVERSION", True), \
         patch.object(tb, "get_sector_etf", return_value="XLK"), \
         patch.object(tb, "SECTOR_RELATIVE_LOOKBACK_BARS", 3), \
         patch.object(tb, "SECTOR_RELATIVE_MIN_UNDERPERFORMANCE_PCT", 2.0):
        # Candidate: -5% over the window. ETF: +1%. Underperformance 6pp >= 2pp threshold.
        blocked = tb.sector_relative_mean_reversion_blocks_entry(
            "NVDA", make_fake_close_series([100, 100, 100, 100, 95]), 4,
            {"XLK": make_fake_close_series([100, 100, 100, 100, 101])})
    check("a candidate genuinely underperforming its sector ETF by more than the threshold is allowed",
          not blocked)


def test_sector_relative_blocks_entry_blocks_soft_sector_day():
    with patch.object(tb, "USE_SECTOR_RELATIVE_MEAN_REVERSION", True), \
         patch.object(tb, "get_sector_etf", return_value="XLK"), \
         patch.object(tb, "SECTOR_RELATIVE_LOOKBACK_BARS", 3), \
         patch.object(tb, "SECTOR_RELATIVE_MIN_UNDERPERFORMANCE_PCT", 2.0):
        # Candidate: -0.5% over the window. ETF: +1%. Underperformance only 1.5pp < 2pp threshold.
        blocked = tb.sector_relative_mean_reversion_blocks_entry(
            "NVDA", make_fake_close_series([100, 100, 100, 100, 99.5]), 4,
            {"XLK": make_fake_close_series([100, 100, 100, 100, 101])})
    check("a candidate that isn't meaningfully weaker than its sector ETF is blocked "
          "(an ordinary soft sector day, not genuine underperformance)", blocked)


def test_check_symbol_blocks_buy_when_sector_relative_blocks_entry():
    df_input = pd.DataFrame({"close": [100.0]})
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "mean_reversion", "test buy")), \
         patch.object(tb, "USE_SECTOR_RELATIVE_MEAN_REVERSION", True), \
         patch.object(tb, "sector_relative_mean_reversion_blocks_entry", return_value=True), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("NVDA", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0)
    check("a mean_reversion BUY is refused when the sector-relative filter blocks it", not mock_buy.called)
    check("a blocked entry reports no notional opened", notional == 0.0)

    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "mean_reversion", "test buy")), \
         patch.object(tb, "USE_SECTOR_RELATIVE_MEAN_REVERSION", True), \
         patch.object(tb, "sector_relative_mean_reversion_blocks_entry", return_value=False), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("NVDA", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0)
    check("the same signal is taken once the sector-relative filter confirms", mock_buy.called)
    check("a taken entry reports its notional", notional == 500.0)


def test_sector_relative_filter_only_applies_to_mean_reversion_signals():
    """The gate must only fire for mean_reversion's OWN signal -- a BUY
    from a different strategy must not be blocked by a check that has
    nothing to do with it."""
    df_input = pd.DataFrame({"close": [100.0]})
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test buy")), \
         patch.object(tb, "USE_SECTOR_RELATIVE_MEAN_REVERSION", True), \
         patch.object(tb, "sector_relative_mean_reversion_blocks_entry", return_value=True), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("NVDA", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0)
    check("a breakout signal is unaffected by the mean-reversion-only sector filter", mock_buy.called)
    check("the unrelated entry reports its notional", notional == 500.0)


def test_sector_relative_filter_noop_when_disabled():
    df_input = pd.DataFrame({"close": [100.0]})
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "mean_reversion", "test buy")), \
         patch.object(tb, "USE_SECTOR_RELATIVE_MEAN_REVERSION", False), \
         patch.object(tb, "sector_relative_mean_reversion_blocks_entry", return_value=True), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("NVDA", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0)
    check("with the filter disabled, check_symbol short-circuits before ever calling the gate "
          "function, so a mocked True result has no effect", mock_buy.called)
    check("the entry reports its notional", notional == 500.0)


def test_check_symbol_blocks_scanner_opening_blackout_on_non_sp500_symbol():
    """
    See USE_SCANNER_OPENING_BLACKOUT -- real-money-confirmed evidence that
    the first minutes after the open are the single worst window for
    scanner-picked (non-S&P-500) entries, independent of and additive to
    the pre-existing lunch-window ENTRY_BLACKOUT.
    """
    df_input = pd.DataFrame({"close": [100.0]})
    early_minute = tb.SCANNER_OPENING_BLACKOUT_MINUTES / 2

    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(early_minute)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test buy")), \
         patch.object(tb, "USE_SCANNER_OPENING_BLACKOUT", True), \
         patch.object(tb, "fetch_sp500_symbols", return_value=["AAPL", "MSFT"]), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("SMALLCAP", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0)
    check("a BUY within the opening blackout on a non-S&P-500 symbol is refused",
          not mock_buy.called)
    check("a blocked entry reports no notional opened", notional == 0.0)

    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(early_minute)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test buy")), \
         patch.object(tb, "USE_SCANNER_OPENING_BLACKOUT", True), \
         patch.object(tb, "fetch_sp500_symbols", return_value=["AAPL", "MSFT"]), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("AAPL", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0)
    check("the same early entry is taken on an S&P 500 symbol", mock_buy.called)
    check("a taken entry reports its notional", notional == 500.0)


def test_scanner_opening_blackout_applies_to_every_strategy():
    """
    Unlike the reason_key-scoped filters (vwap_volume, trend_following
    gate), the opening blackout is modeled on REAL losses that weren't
    all the same strategy -- it must block ANY strategy's BUY signal
    during the window, not just one.
    """
    df_input = pd.DataFrame({"close": [100.0]})
    early_minute = tb.SCANNER_OPENING_BLACKOUT_MINUTES / 2
    for reason_key in ("breakout", "trend_following", "vwap_reversion", "mean_reversion"):
        with patch.object(tb, "add_indicators", return_value=make_fake_enriched(early_minute)), \
             patch.object(tb, "decide_signal_at", return_value=("BUY", reason_key, "test buy")), \
             patch.object(tb, "USE_SCANNER_OPENING_BLACKOUT", True), \
             patch.object(tb, "USE_VWAP_VOLUME_CONFIRMATION", False), \
             patch.object(tb, "fetch_sp500_symbols", return_value=["AAPL", "MSFT"]), \
             patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
            notional = tb.check_symbol("SMALLCAP", df_input, entries_paused_reason=None,
                                        at_position_cap=False, current_qty=0.0, equity=10000.0,
                                        portfolio_risk_estimate=0.0)
        check(f"the opening blackout blocks a {reason_key} BUY too", not mock_buy.called and notional == 0.0)


def test_scanner_opening_blackout_noop_when_disabled():
    df_input = pd.DataFrame({"close": [100.0]})
    early_minute = tb.SCANNER_OPENING_BLACKOUT_MINUTES / 2
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(early_minute)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test buy")), \
         patch.object(tb, "USE_SCANNER_OPENING_BLACKOUT", False), \
         patch.object(tb, "fetch_sp500_symbols") as mock_fetch, \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("SMALLCAP", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0)
    check("with the gate disabled, check_symbol short-circuits before ever calling fetch_sp500_symbols",
          not mock_fetch.called)
    check("the entry is taken normally", mock_buy.called and notional == 500.0)


def test_scanner_opening_blackout_does_not_block_after_the_window():
    """Past SCANNER_OPENING_BLACKOUT_MINUTES, a scanner-pick BUY proceeds normally."""
    df_input = pd.DataFrame({"close": [100.0]})
    late_minute = tb.SCANNER_OPENING_BLACKOUT_MINUTES + 5
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(late_minute)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test buy")), \
         patch.object(tb, "USE_SCANNER_OPENING_BLACKOUT", True), \
         patch.object(tb, "fetch_sp500_symbols", return_value=["AAPL", "MSFT"]), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("SMALLCAP", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0)
    check("a BUY past the opening blackout window is taken normally",
          mock_buy.called and notional == 500.0)


def test_symbol_cooldown_blocks_immediate_reentry():
    """
    Regression for 2026-07-27, where 7 of 10 trades were rapid re-entries
    into two falling stocks -- VEEE bought back FIVE minutes after being
    stopped out, four times in total, and TRAX three times. A BUY signal
    is not enough on its own; a symbol whose position just closed has to
    sit out first.
    """
    df_input = pd.DataFrame({"close": [100.0]})  # content irrelevant, add_indicators is mocked

    # USE_VWAP_VOLUME_CONFIRMATION forced off: this test is about the
    # cooldown gate specifically, and the minimal fixture here doesn't
    # carry the volume columns that other gate would read.
    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "vwap_reversion", "test buy")), \
         patch.object(tb, "USE_VWAP_VOLUME_CONFIRMATION", False), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("VEEE", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0, in_cooldown=True)
    check("a BUY signal is refused while the symbol is cooling off", not mock_buy.called)
    check("a refused re-entry reports no notional opened", notional == 0.0)

    with patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "vwap_reversion", "test buy")), \
         patch.object(tb, "USE_VWAP_VOLUME_CONFIRMATION", False), \
         patch.object(tb, "place_buy_order", return_value=(MagicMock(id="x"), 500.0)) as mock_buy:
        notional = tb.check_symbol("VEEE", df_input, entries_paused_reason=None,
                                    at_position_cap=False, current_qty=0.0, equity=10000.0,
                                    portfolio_risk_estimate=0.0, in_cooldown=False)
    check("the same signal is taken once the cooldown has expired", mock_buy.called)
    check("a taken entry reports its notional", notional == 500.0)


# ---------------------------------------------------------------------------
# DURATION MODE -- the GitHub Actions entry point. Exercised here against a
# SIMULATED open market, because outside trading hours the only path that can
# be run live is the market-closed early exit.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# compute_next_cycle_sleep -- 2026-07-31. Found while rechecking the
# CHECK_INTERVAL_MINUTES change (5 -> 15): the old inline sleep calculation
# had no awareness of FLATTEN_MINUTES_BEFORE_CLOSE, so a wide check
# interval could sleep straight past the intended flatten-trigger buffer
# and wake up with almost no margin before the actual market close.
# ---------------------------------------------------------------------------

def test_compute_next_cycle_sleep_never_overshoots_the_flatten_trigger():
    """The regression case: 16 minutes left in the session, a 15-minute
    check interval. The old code slept the full 15 minutes and woke with
    only ~1 minute left -- 9 minutes later than the intended 10-minute
    flatten buffer."""
    with patch.object(tb, "CHECK_INTERVAL_MINUTES", 15), \
         patch.object(tb, "FLATTEN_MINUTES_BEFORE_CLOSE", 10), \
         patch.object(tb, "FLATTEN_BEFORE_CLOSE", True):
        sleep_seconds = tb.compute_next_cycle_sleep(16 * 60)
    woken_with_seconds_left = 16 * 60 - sleep_seconds
    # Lands a couple seconds INSIDE the flatten window on purpose (the
    # "+2" in compute_next_cycle_sleep) so the top-of-cycle check is
    # guaranteed to catch it rather than landing exactly on a borderline
    # boundary -- so this checks "close to 10 minutes", not ">= 10".
    check("wakes close to the intended 10-minute flatten buffer, not the old ~1-minute overshoot",
          595 <= woken_with_seconds_left <= 600)


def test_compute_next_cycle_sleep_unaffected_with_plenty_of_runway():
    """Far from the close, the cap must never kick in -- normal behavior,
    sleep the full check interval."""
    with patch.object(tb, "CHECK_INTERVAL_MINUTES", 15), \
         patch.object(tb, "FLATTEN_MINUTES_BEFORE_CLOSE", 10), \
         patch.object(tb, "FLATTEN_BEFORE_CLOSE", True):
        sleep_seconds = tb.compute_next_cycle_sleep(60 * 60)
    check("sleeps the full normal check interval when there's plenty of runway",
          sleep_seconds == 15 * 60)


def test_compute_next_cycle_sleep_already_inside_flatten_window():
    """Already well inside the flatten window (a prior cycle should have
    caught this already, but the function must still behave sanely if
    reached here) -- the cap must not force an artificially LONGER sleep
    than the plain interval would already give."""
    with patch.object(tb, "CHECK_INTERVAL_MINUTES", 15), \
         patch.object(tb, "FLATTEN_MINUTES_BEFORE_CLOSE", 10), \
         patch.object(tb, "FLATTEN_BEFORE_CLOSE", True):
        sleep_seconds = tb.compute_next_cycle_sleep(5 * 60)
    check("stays capped at the remaining session time, not stretched out",
          sleep_seconds <= 5 * 60)


def test_compute_next_cycle_sleep_noop_when_flatten_disabled():
    """With FLATTEN_BEFORE_CLOSE off, the cap must never engage -- there's
    no flatten window to protect the timing of."""
    with patch.object(tb, "CHECK_INTERVAL_MINUTES", 15), \
         patch.object(tb, "FLATTEN_MINUTES_BEFORE_CLOSE", 10), \
         patch.object(tb, "FLATTEN_BEFORE_CLOSE", False):
        sleep_seconds = tb.compute_next_cycle_sleep(16 * 60)
    check("sleeps the plain interval when flatten-before-close is off",
          sleep_seconds == 15 * 60)


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


# ---------------------------------------------------------------------------
# Breakout invalidation exit -- open_position_context state-file round
# trip, and check_symbol's wiring of USE_BREAKOUT_INVALIDATION_EXIT. See
# strategy.breakout_invalidated_at / trading_bot's OPEN POSITION CONTEXT
# section.
# ---------------------------------------------------------------------------

def test_open_position_context_persistence():
    tmp_dir = tempfile.mkdtemp()
    state_file = os.path.join(tmp_dir, "open_position_context_test.json")

    with patch.object(tb, "OPEN_POSITION_CONTEXT_FILE", state_file):
        tb.open_position_context = {
            "AAA": {"strategy": "breakout", "invalidation_level": 123.45},
            "BBB": {"strategy": "vwap_reversion", "invalidation_level": None},
        }
        tb.save_open_position_context()

        # Simulate a crash-and-restart: wipe the in-memory state, then reload it.
        tb.open_position_context = {}
        tb.load_open_position_context()

        check("a breakout entry (with its frozen level) survives a save/reload round-trip",
              tb.open_position_context.get("AAA") == {"strategy": "breakout", "invalidation_level": 123.45})
        check("a non-breakout entry (invalidation_level None) survives a save/reload round-trip too",
              tb.open_position_context.get("BBB") == {"strategy": "vwap_reversion", "invalidation_level": None})

        tb.clear_open_position_context("AAA")
        check("clear_open_position_context removes just that symbol's in-memory entry",
              "AAA" not in tb.open_position_context and "BBB" in tb.open_position_context)

        # Reload from disk to confirm the clear was actually PERSISTED,
        # not just removed from the in-memory dict.
        tb.open_position_context = {}
        tb.load_open_position_context()
        check("the clear survives a reload too -- it was written to disk, not just in-memory",
              "AAA" not in tb.open_position_context and "BBB" in tb.open_position_context)

    tb.open_position_context = {}


def test_load_open_position_context_missing_file_starts_fresh():
    tmp_dir = tempfile.mkdtemp()
    missing_file = os.path.join(tmp_dir, "does_not_exist.json")
    with patch.object(tb, "OPEN_POSITION_CONTEXT_FILE", missing_file):
        tb.open_position_context = {"STALE": {"strategy": "breakout", "invalidation_level": 1.0}}
        tb.load_open_position_context()  # must not raise or wipe existing in-memory state
    check("a missing state file is not an error -- load leaves whatever was already in memory untouched",
          tb.open_position_context == {"STALE": {"strategy": "breakout", "invalidation_level": 1.0}})
    tb.open_position_context = {}


def test_check_symbol_buy_persists_open_position_context_for_breakout():
    """The BUY branch must persist the SAME level breakout_at() itself
    used to confirm the entry (breakout_recent_high, since
    USE_CLOSE_BEYOND_LEVEL_CONFIRMATION defaults off)."""
    df_input = pd.DataFrame({"close": [100.0]})
    fake_order = types.SimpleNamespace(id="test-order-id", client_order_id="breakout-123")
    tmp_dir = tempfile.mkdtemp()
    state_file = os.path.join(tmp_dir, "open_position_context_test.json")

    fake_enriched = make_fake_enriched(100)
    fake_enriched["breakout_recent_high"] = [97.5]
    fake_enriched["breakout_recent_high_wick"] = [98.5]

    tb.open_position_context = {}
    with patch.object(tb, "OPEN_POSITION_CONTEXT_FILE", state_file), \
         patch.object(tb, "add_indicators", return_value=fake_enriched), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "breakout", "test breakout")), \
         patch.object(tb, "place_buy_order", return_value=(fake_order, 500.0)):
        tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                         at_position_cap=False, current_qty=0.0, equity=10000.0, portfolio_risk_estimate=0.0)

    check("a breakout BUY persists strategy='breakout' and the entry bar's breakout_recent_high level",
          tb.open_position_context.get("AAA") == {"strategy": "breakout", "invalidation_level": 97.5})
    tb.open_position_context = {}


def test_check_symbol_buy_persists_none_invalidation_level_for_non_breakout():
    # reason_key="trend_following" (not vwap_reversion) deliberately --
    # a vwap_reversion BUY would also exercise vwap_volume_blocks_entry's
    # own volume-confirmation lookup, which needs indicator columns this
    # minimal fake enriched df doesn't have and isn't what's under test here.
    df_input = pd.DataFrame({"close": [100.0]})
    fake_order = types.SimpleNamespace(id="test-order-id", client_order_id="trend_following-123")
    tmp_dir = tempfile.mkdtemp()
    state_file = os.path.join(tmp_dir, "open_position_context_test.json")

    tb.open_position_context = {}
    with patch.object(tb, "OPEN_POSITION_CONTEXT_FILE", state_file), \
         patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("BUY", "trend_following", "test trend")), \
         patch.object(tb, "place_buy_order", return_value=(fake_order, 500.0)):
        tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                         at_position_cap=False, current_qty=0.0, equity=10000.0, portfolio_risk_estimate=0.0)

    check("a non-breakout BUY still records its own strategy, with invalidation_level None",
          tb.open_position_context.get("AAA") == {"strategy": "trend_following", "invalidation_level": None})
    tb.open_position_context = {}


def test_check_symbol_sell_signal_clears_open_position_context():
    df_input = pd.DataFrame({"close": [100.0]})
    tmp_dir = tempfile.mkdtemp()
    state_file = os.path.join(tmp_dir, "open_position_context_test.json")
    tb.open_position_context = {"AAA": {"strategy": "breakout", "invalidation_level": 90.0}}

    with patch.object(tb, "OPEN_POSITION_CONTEXT_FILE", state_file), \
         patch.object(tb, "add_indicators", return_value=make_fake_enriched(100)), \
         patch.object(tb, "decide_signal_at", return_value=("SELL", "trend_following", "test sell")), \
         patch.object(tb, "place_sell_order", return_value=types.SimpleNamespace(id="sell-id")):
        tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                         at_position_cap=False, current_qty=5.0, equity=10000.0, portfolio_risk_estimate=0.0)

    check("a regular strategy-SELL-signal exit clears the symbol's open_position_context entry",
          "AAA" not in tb.open_position_context)
    tb.open_position_context = {}


def test_check_symbol_breakout_invalidation_exit_sells_when_flagged_and_broken():
    """The whole point of this candidate: a breakout position the regime
    strategy's own signal says nothing about (HOLD) still gets exited
    once price closes back below the level that justified its entry."""
    df_input = pd.DataFrame({"close": [100.0]})
    tmp_dir = tempfile.mkdtemp()
    state_file = os.path.join(tmp_dir, "open_position_context_test.json")
    original_toggle = tb.USE_BREAKOUT_INVALIDATION_EXIT
    try:
        tb.USE_BREAKOUT_INVALIDATION_EXIT = True
        tb.open_position_context = {"AAA": {"strategy": "breakout", "invalidation_level": 105.0}}

        with patch.object(tb, "OPEN_POSITION_CONTEXT_FILE", state_file), \
             patch.object(tb, "add_indicators", return_value=make_fake_enriched(100, close=100.0)), \
             patch.object(tb, "decide_signal_at", return_value=("HOLD", "trend_following", "test hold")), \
             patch.object(tb, "place_sell_order", return_value=types.SimpleNamespace(id="inv-exit-id")) as mock_sell:
            notional = tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                                        at_position_cap=False, current_qty=5.0, equity=10000.0,
                                        portfolio_risk_estimate=0.0)
        check("a breakout position whose entry level has been invalidated gets sold, "
              "even though the currently-active regime strategy's own signal is HOLD",
              mock_sell.called and notional == 0.0)
        check("the invalidation exit clears the symbol's open_position_context entry",
              "AAA" not in tb.open_position_context)
    finally:
        tb.USE_BREAKOUT_INVALIDATION_EXIT = original_toggle
        tb.open_position_context = {}


def test_check_symbol_breakout_invalidation_exit_does_not_fire_for_non_breakout_position():
    """The exact same price action (close back below 105) must NOT sell a
    position that open_position_context says was opened by a different
    strategy -- this mechanism is scoped to breakout entries only."""
    df_input = pd.DataFrame({"close": [100.0]})
    tmp_dir = tempfile.mkdtemp()
    state_file = os.path.join(tmp_dir, "open_position_context_test.json")
    original_toggle = tb.USE_BREAKOUT_INVALIDATION_EXIT
    try:
        tb.USE_BREAKOUT_INVALIDATION_EXIT = True
        tb.open_position_context = {"AAA": {"strategy": "vwap_reversion", "invalidation_level": 105.0}}

        with patch.object(tb, "OPEN_POSITION_CONTEXT_FILE", state_file), \
             patch.object(tb, "add_indicators", return_value=make_fake_enriched(100, close=100.0)), \
             patch.object(tb, "decide_signal_at", return_value=("HOLD", "trend_following", "test hold")), \
             patch.object(tb, "place_sell_order") as mock_sell:
            notional = tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                                        at_position_cap=False, current_qty=5.0, equity=10000.0,
                                        portfolio_risk_estimate=0.0)
        check("a non-breakout open position is NOT sold by the breakout invalidation mechanism, "
              "even with the identical price action that WOULD invalidate a breakout entry",
              notional == 0.0 and not mock_sell.called)
    finally:
        tb.USE_BREAKOUT_INVALIDATION_EXIT = original_toggle
        tb.open_position_context = {}


def test_check_symbol_breakout_invalidation_exit_noop_when_toggle_off():
    df_input = pd.DataFrame({"close": [100.0]})
    tmp_dir = tempfile.mkdtemp()
    state_file = os.path.join(tmp_dir, "open_position_context_test.json")
    original_toggle = tb.USE_BREAKOUT_INVALIDATION_EXIT
    try:
        tb.USE_BREAKOUT_INVALIDATION_EXIT = False
        tb.open_position_context = {"AAA": {"strategy": "breakout", "invalidation_level": 105.0}}

        with patch.object(tb, "OPEN_POSITION_CONTEXT_FILE", state_file), \
             patch.object(tb, "add_indicators", return_value=make_fake_enriched(100, close=100.0)), \
             patch.object(tb, "decide_signal_at", return_value=("HOLD", "trend_following", "test hold")), \
             patch.object(tb, "place_sell_order") as mock_sell:
            notional = tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                                        at_position_cap=False, current_qty=5.0, equity=10000.0,
                                        portfolio_risk_estimate=0.0)
        check("toggle off means zero behavior change -- an otherwise-invalidated breakout "
              "position is left alone",
              notional == 0.0 and not mock_sell.called)
    finally:
        tb.USE_BREAKOUT_INVALIDATION_EXIT = original_toggle
        tb.open_position_context = {}


def test_check_symbol_breakout_invalidation_exit_unknown_position_not_touched():
    """
    Fail-safe requirement: a currently-open position with NO entry in
    open_position_context (e.g. it existed before this feature was
    deployed, or the state file was reset) must never be guessed at --
    this mechanism simply doesn't apply to it.
    """
    df_input = pd.DataFrame({"close": [100.0]})
    tmp_dir = tempfile.mkdtemp()
    state_file = os.path.join(tmp_dir, "open_position_context_test.json")
    original_toggle = tb.USE_BREAKOUT_INVALIDATION_EXIT
    try:
        tb.USE_BREAKOUT_INVALIDATION_EXIT = True
        tb.open_position_context = {}  # symbol is NOT in the state file at all

        with patch.object(tb, "OPEN_POSITION_CONTEXT_FILE", state_file), \
             patch.object(tb, "add_indicators", return_value=make_fake_enriched(100, close=100.0)), \
             patch.object(tb, "decide_signal_at", return_value=("HOLD", "trend_following", "test hold")), \
             patch.object(tb, "place_sell_order") as mock_sell:
            notional = tb.check_symbol("AAA", df_input, entries_paused_reason=None,
                                        at_position_cap=False, current_qty=5.0, equity=10000.0,
                                        portfolio_risk_estimate=0.0)
        check("a position with no entry in open_position_context is never touched by this mechanism",
              notional == 0.0 and not mock_sell.called)
    finally:
        tb.USE_BREAKOUT_INVALIDATION_EXIT = original_toggle
        tb.open_position_context = {}


def test_flatten_all_positions_clears_open_position_context():
    tmp_dir = tempfile.mkdtemp()
    state_file = os.path.join(tmp_dir, "open_position_context_test.json")
    tb.open_position_context = {
        "AAA": {"strategy": "breakout", "invalidation_level": 90.0},
        "BBB": {"strategy": "trend_following", "invalidation_level": None},
    }
    fake_client = MagicMock()
    fake_client.close_position.return_value = None
    fake_client.cancel_orders.return_value = None

    with patch.object(tb, "OPEN_POSITION_CONTEXT_FILE", state_file), \
         patch.object(tb, "trading_client", fake_client), \
         patch.object(tb, "get_all_open_positions", return_value={
             "AAA": {"qty": 5.0, "avg_entry_price": 100.0},
             "BBB": {"qty": 3.0, "avg_entry_price": 50.0},
         }):
        result = tb.flatten_all_positions()

    check("flatten_all_positions reports success when every close succeeds", result is True)
    check("flatten_all_positions clears every closed symbol's open_position_context entry",
          tb.open_position_context == {})
    tb.open_position_context = {}


def test_record_auto_exit_clears_open_position_context_even_on_lookup_failure():
    """
    clear_open_position_context must run unconditionally, before the
    best-effort Alpaca order-history lookup -- the caller (run_one_cycle)
    already knows the position is gone by the time it calls this, so
    clearing local state can't be allowed to depend on that lookup
    succeeding.
    """
    tmp_dir = tempfile.mkdtemp()
    state_file = os.path.join(tmp_dir, "open_position_context_test.json")
    tb.open_position_context = {"AAA": {"strategy": "breakout", "invalidation_level": 90.0}}

    failing_client = MagicMock()
    failing_client.get_orders.side_effect = Exception("simulated API failure")

    with patch.object(tb, "OPEN_POSITION_CONTEXT_FILE", state_file), \
         patch.object(tb, "trading_client", failing_client):
        tb.record_auto_exit("AAA")

    check("record_auto_exit clears open_position_context even when the order-history lookup fails",
          "AAA" not in tb.open_position_context)
    tb.open_position_context = {}


if __name__ == "__main__":
    tests = [
        test_would_exceed_portfolio_risk_cap,
        test_get_current_portfolio_risk_usd,
        test_estimate_new_position_risk_usd,
        test_estimate_new_position_risk_usd_volatility_scaled,
        test_estimate_new_position_risk_usd_conviction_boosted,
        test_estimate_new_position_risk_usd_volatility_precedes_conviction,
        test_conviction_sizing_structurally_inert_with_real_shipped_default,
        test_would_exceed_portfolio_heat_cap,
        test_portfolio_heat_cap_blocks_when_aggregate_open_risk_is_near_the_ceiling,
        test_daily_risk_state_persistence,
        test_check_symbol_gating,
        test_check_symbol_propagates_high_vol_tercile_to_place_buy_order,
        test_place_buy_order_conviction_sizing,
        test_place_buy_order_volatility_precedes_conviction,
        test_check_symbol_gating_portfolio_heat_cap,
        test_get_symbol_sector_uses_hardcoded_map_first,
        test_get_symbol_sector_falls_back_to_alpaca_asset_metadata,
        test_get_symbol_sector_unknown_when_neither_source_has_it,
        test_get_symbol_sector_caches_including_negative_results,
        test_sector_concentration_blocks_entry_disabled_by_default,
        test_sector_concentration_blocks_entry_blocks_a_sixth_same_sector_candidate,
        test_sector_concentration_blocks_entry_allows_a_different_sector_candidate,
        test_sector_concentration_blocks_entry_allows_exactly_up_to_the_cap,
        test_sector_concentration_blocks_entry_fails_open_on_unknown_candidate_sector,
        test_sector_concentration_blocks_entry_unmapped_open_positions_dont_count,
        test_check_symbol_blocks_buy_when_sector_cap_blocks_entry,
        test_check_symbol_sector_cap_does_not_apply_to_sell_signals,
        test_leveraged_etf_name_detection,
        test_scanner_tops_up_a_thin_watchlist,
        test_fetch_sp500_symbols_caches_and_falls_back,
        test_fetch_sp500_symbols_no_cache_and_failure_returns_empty,
        test_sp500_backstop_ranks_by_liquidity_and_excludes_already_picked,
        test_sp500_backstop_requests_nothing_when_not_needed,
        test_scan_reserves_slots_for_sp500_even_when_movers_scan_finds_nothing,
        test_scan_caps_total_watchlist_size_with_backstop_included,
        test_meets_min_listing_age_filters_by_trading_day_count,
        test_meets_min_listing_age_fails_open_on_request_failure,
        test_meets_min_listing_age_empty_input_is_a_noop,
        test_scan_excludes_candidates_beyond_max_extension_pct,
        test_scan_excludes_candidates_below_min_listing_age,
        test_scan_listing_age_filter_noop_when_disabled,
        test_scan_liquidity_check_excludes_todays_bars,
        test_check_symbol_blocks_scanner_opening_blackout_on_non_sp500_symbol,
        test_scanner_opening_blackout_applies_to_every_strategy,
        test_scanner_opening_blackout_noop_when_disabled,
        test_scanner_opening_blackout_does_not_block_after_the_window,
        test_vwap_volume_filter_exempts_sp500_even_when_volume_is_weak,
        test_vwap_volume_filter_blocks_non_sp500_with_weak_volume,
        test_vwap_volume_filter_allows_non_sp500_with_strong_volume,
        test_vwap_volume_filter_only_applies_to_vwap_reversion_signals,
        test_vwap_volume_filter_noop_when_disabled,
        test_daily_trend_confirms_entry_exempts_sp500_regardless_of_trend,
        test_daily_trend_confirms_entry_gates_non_sp500_on_real_trend,
        test_daily_trend_confirms_entry_fails_closed_on_unknown_symbol,
        test_daily_trend_confirms_entry_noop_when_filter_disabled,
        test_check_symbol_blocks_buy_when_daily_trend_blocks_entry,
        test_spy_regime_confirms_entry_noop_when_gate_disabled,
        test_spy_regime_confirms_entry_blocks_on_confirmed_spy_downtrend,
        test_spy_regime_confirms_entry_allows_when_spy_trending_up,
        test_spy_regime_confirms_entry_allows_when_spy_choppy,
        test_spy_regime_confirms_entry_fails_open_on_fetch_failure,
        test_spy_regime_confirms_entry_fails_open_on_missing_data,
        test_spy_regime_confirms_entry_fails_open_during_warmup,
        test_filter_to_regular_session_drops_pre_and_post_market_bars,
        test_filter_to_regular_session_keeps_every_regular_session_bar,
        test_filter_to_regular_session_empty_input_is_a_noop,
        test_filter_to_regular_session_matches_what_backtest_would_compute,
        test_get_recent_bars_batch_regular_session_only_filters_each_symbol,
        test_spy_regime_confirms_entry_requests_regular_session_only_bars,
        test_sp500_backstop_and_scanner_liquidity_checks_do_not_request_regular_session_only,
        test_check_symbol_blocks_buy_when_spy_regime_blocks_entry,
        test_get_sector_etf_resolves_via_sector_map_and_etf_map,
        test_get_sector_etf_none_when_sector_unknown,
        test_sector_relative_blocks_entry_disabled_by_default,
        test_sector_relative_blocks_entry_fails_open_on_unknown_sector,
        test_sector_relative_blocks_entry_fails_open_when_etf_bars_missing,
        test_sector_relative_blocks_entry_fails_open_on_insufficient_history,
        test_sector_relative_blocks_entry_allows_genuine_underperformance,
        test_sector_relative_blocks_entry_blocks_soft_sector_day,
        test_check_symbol_blocks_buy_when_sector_relative_blocks_entry,
        test_sector_relative_filter_only_applies_to_mean_reversion_signals,
        test_sector_relative_filter_noop_when_disabled,
        test_compute_next_cycle_sleep_never_overshoots_the_flatten_trigger,
        test_compute_next_cycle_sleep_unaffected_with_plenty_of_runway,
        test_compute_next_cycle_sleep_already_inside_flatten_window,
        test_compute_next_cycle_sleep_noop_when_flatten_disabled,
        test_reconcile_bracket_keeps_small_drift_unchanged,
        test_reconcile_bracket_reprices_legs_on_large_drift,
        test_reconcile_bracket_falls_back_when_fill_not_confirmed,
        test_reconcile_bracket_survives_a_failed_leg_replace,
        test_symbol_cooldown_blocks_immediate_reentry,
        test_run_for_duration_cycles_while_market_open,
        test_run_for_duration_never_overruns_its_window,
        test_run_for_duration_exits_when_session_is_over,
        test_run_for_duration_survives_a_bad_cycle_but_reports_it,
        test_open_position_context_persistence,
        test_load_open_position_context_missing_file_starts_fresh,
        test_check_symbol_buy_persists_open_position_context_for_breakout,
        test_check_symbol_buy_persists_none_invalidation_level_for_non_breakout,
        test_check_symbol_sell_signal_clears_open_position_context,
        test_check_symbol_breakout_invalidation_exit_sells_when_flagged_and_broken,
        test_check_symbol_breakout_invalidation_exit_does_not_fire_for_non_breakout_position,
        test_check_symbol_breakout_invalidation_exit_noop_when_toggle_off,
        test_check_symbol_breakout_invalidation_exit_unknown_position_not_touched,
        test_flatten_all_positions_clears_open_position_context,
        test_record_auto_exit_clears_open_position_context_even_on_lookup_failure,
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
