"""
Synthetic regression tests for backtest.py's entry-fill slippage.
--------------------------------------------------------------------
Same style as test_strategy.py: no network, no pytest, hand-built
pandas DataFrames with known expected outcomes, run directly with
`py test_backtest.py`.

backtest.py deliberately does NOT import trading_bot.py (it mirrors that
file's logic independently so it can run standalone) -- this file
mirrors that same independence and only imports backtest.py.

Importing backtest.py reads your real .env for Alpaca keys (same as
running it directly) but makes no network calls at import time -- every
Alpaca client call it makes lives inside functions only invoked from
`if __name__ == "__main__":`, none of which this file calls.
"""

import pandas as pd
from unittest.mock import patch

import backtest as bt

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def ts_range(day: str, start_time: str, n: int, bar_minutes: int = 5) -> list:
    start = pd.Timestamp(f"{day} {start_time}", tz="America/New_York")
    return [(start + pd.Timedelta(minutes=bar_minutes * i)).tz_convert("UTC") for i in range(n)]


def flat_bars(n: int = 20, close: float = 100.0) -> pd.DataFrame:
    """
    A quiet, unmoving stock, entirely within one session, well clear of
    the entry blackout window (1-2pm ET) and the pre-close entry cutoff.
    Flat (no stop/take-profit ever triggers) so the ONE position opened
    stays open until the end-of-day flatten closes it -- exactly one
    completed trade to inspect, and its entry_price is never touched by
    anything except the entry-slippage logic under test.
    """
    timestamps = ts_range("2026-08-17", "09:35", n)  # Monday, well after the open
    closes = [close] * n
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [50_000] * n,
    })


def run_forced_buy(is_scanner_pick: bool, close: float = 100.0) -> dict:
    """
    Forces decide_signal_at to return a BUY (reason_key trend_following,
    so none of the reason_key-specific filters/gates get in the way) on
    every bar, and runs simulate() with the given is_scanner_pick.
    Position sizing/state (apply_vwap_volume_filter, the various gates)
    are all left at their defaults (off), same as calling simulate()
    with only the fields this test cares about. Returns the single
    resulting trade dict.
    """
    bars = flat_bars(close=close)
    with patch.object(bt, "decide_signal_at", return_value=("BUY", "trend_following", "test buy")):
        trades = bt.simulate("TEST", bars, starting_equity=100_000, is_scanner_pick=is_scanner_pick)
    check(f"forcing a BUY signal produces exactly one trade (is_scanner_pick={is_scanner_pick})",
          len(trades) == 1, f"got {len(trades)}")
    return trades[0]


# ---------------------------------------------------------------------------
# 1. The core claim: a scanner-pick entry fills WORSE than the bar's
#    close, by exactly ENTRY_SLIPPAGE_PCT_SCANNER.
# ---------------------------------------------------------------------------

def test_scanner_pick_entry_slips_by_the_calibrated_scanner_constant():
    close = 100.0
    trade = run_forced_buy(is_scanner_pick=True, close=close)
    expected_entry = close * (1 + bt.ENTRY_SLIPPAGE_PCT_SCANNER / 100)
    check("a scanner-pick entry fills WORSE than the bar's close (adverse direction)",
          trade["entry_price"] > close, f"entry {trade['entry_price']} vs close {close}")
    check("the fill matches close * (1 + ENTRY_SLIPPAGE_PCT_SCANNER/100) exactly",
          abs(trade["entry_price"] - expected_entry) < 1e-9,
          f"got {trade['entry_price']}, expected {expected_entry}")


# ---------------------------------------------------------------------------
# 2. S&P 500 names use the OTHER constant (measured near zero -- not
#    presumed identical to the scanner-pick figure).
# ---------------------------------------------------------------------------

def test_sp500_pick_entry_slips_by_the_calibrated_sp500_constant():
    close = 100.0
    trade = run_forced_buy(is_scanner_pick=False, close=close)
    expected_entry = close * (1 + bt.ENTRY_SLIPPAGE_PCT_SP500 / 100)
    check("the fill matches close * (1 + ENTRY_SLIPPAGE_PCT_SP500/100) exactly",
          abs(trade["entry_price"] - expected_entry) < 1e-9,
          f"got {trade['entry_price']}, expected {expected_entry}")


# ---------------------------------------------------------------------------
# 3. The two constants are independently calibrated, not the same value
#    in disguise -- pin that a scanner pick and an S&P 500 pick on the
#    IDENTICAL bar data get DIFFERENT fills whenever the constants differ.
# ---------------------------------------------------------------------------

def test_scanner_and_sp500_entries_differ_when_the_constants_differ():
    close = 100.0
    scanner_trade = run_forced_buy(is_scanner_pick=True, close=close)
    sp500_trade = run_forced_buy(is_scanner_pick=False, close=close)
    if bt.ENTRY_SLIPPAGE_PCT_SCANNER != bt.ENTRY_SLIPPAGE_PCT_SP500:
        check("a scanner pick and an S&P 500 pick on the same bar get different fills",
              scanner_trade["entry_price"] != sp500_trade["entry_price"])
    else:
        check("constants happen to be equal this run -- nothing to differentiate (not a real failure)",
              True)


# ---------------------------------------------------------------------------
# 4. Not calling out is_scanner_pick at all must reproduce the OLD,
#    zero-slippage-entry behavior exactly -- no caller should silently
#    pick up a haircut it never asked for.
# ---------------------------------------------------------------------------

def test_omitting_is_scanner_pick_defaults_to_the_sp500_constant():
    close = 100.0
    bars = flat_bars(close=close)
    with patch.object(bt, "decide_signal_at", return_value=("BUY", "trend_following", "test buy")):
        trades = bt.simulate("TEST", bars, starting_equity=100_000)  # is_scanner_pick omitted
    expected_entry = close * (1 + bt.ENTRY_SLIPPAGE_PCT_SP500 / 100)
    check("omitting is_scanner_pick uses ENTRY_SLIPPAGE_PCT_SP500, not the scanner figure",
          abs(trades[0]["entry_price"] - expected_entry) < 1e-9,
          f"got {trades[0]['entry_price']}, expected {expected_entry}")


# ---------------------------------------------------------------------------
# 5. THE load-bearing consistency check the task actually asked for:
#    compute_stop_and_target and stop_is_wider_than_noise must both be
#    called with the SLIPPAGE-ADJUSTED entry price, not the original bar
#    close -- otherwise the position's stop/target would be priced off a
#    number the position was never actually opened at, and the whole
#    risk-per-share calculation (position sizing, "is this stop wider
#    than noise") would be silently wrong. Spies on both real strategy.py
#    functions (still calling straight through to them, so their actual
#    math is unaffected) and inspects what they were called with.
# ---------------------------------------------------------------------------

def test_stop_target_and_noise_guard_use_the_slippage_adjusted_entry_price():
    close = 100.0
    bars = flat_bars(close=close)
    expected_entry = close * (1 + bt.ENTRY_SLIPPAGE_PCT_SCANNER / 100)

    stop_target_calls = []
    real_compute_stop_and_target = bt.compute_stop_and_target

    def spy_compute_stop_and_target(entry_price, atr_value=None):
        stop_target_calls.append(entry_price)
        return real_compute_stop_and_target(entry_price, atr_value)

    noise_guard_calls = []
    real_stop_is_wider_than_noise = bt.stop_is_wider_than_noise

    def spy_stop_is_wider_than_noise(entry_price, atr_value, stop_price):
        noise_guard_calls.append(entry_price)
        return real_stop_is_wider_than_noise(entry_price, atr_value, stop_price)

    with patch.object(bt, "decide_signal_at", return_value=("BUY", "trend_following", "test buy")), \
         patch.object(bt, "compute_stop_and_target", side_effect=spy_compute_stop_and_target), \
         patch.object(bt, "stop_is_wider_than_noise", side_effect=spy_stop_is_wider_than_noise):
        trades = bt.simulate("TEST", bars, starting_equity=100_000, is_scanner_pick=True)

    check("exactly one trade opened", len(trades) == 1, f"got {len(trades)}")
    check("compute_stop_and_target was called exactly once, for the one entry",
          len(stop_target_calls) == 1, f"got {len(stop_target_calls)} calls")
    check("compute_stop_and_target received the SLIPPAGE-ADJUSTED entry price, not the raw bar close",
          stop_target_calls and abs(stop_target_calls[0] - expected_entry) < 1e-9
          and abs(stop_target_calls[0] - close) > 1e-9,
          f"got {stop_target_calls}, expected [{expected_entry}] (raw close was {close})")
    check("stop_is_wider_than_noise received the SAME slippage-adjusted entry price",
          noise_guard_calls and abs(noise_guard_calls[0] - expected_entry) < 1e-9,
          f"got {noise_guard_calls}, expected [{expected_entry}]")
    check("the trade's own recorded entry_price matches what both risk functions were fed "
          "(no double-counting, no drift between what's recorded and what's risk-managed)",
          trades and abs(trades[0]["entry_price"] - expected_entry) < 1e-9)


if __name__ == "__main__":
    tests = [
        test_scanner_pick_entry_slips_by_the_calibrated_scanner_constant,
        test_sp500_pick_entry_slips_by_the_calibrated_sp500_constant,
        test_scanner_and_sp500_entries_differ_when_the_constants_differ,
        test_omitting_is_scanner_pick_defaults_to_the_sp500_constant,
        test_stop_target_and_noise_guard_use_the_slippage_adjusted_entry_price,
    ]
    for t in tests:
        print(f"\n{t.__name__}")
        t()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        raise SystemExit(1)
    print("All tests passed.")
