"""
Tests for daily_summary.py's pure logic -- order pairing and strategy
extraction. Does NOT test send_email (needs a real SMTP connection) or
fetch_todays_filled_orders (needs a real Alpaca connection) -- those are
thin wrappers around external calls with nothing to unit test.

Sets dummy EMAIL_ADDRESS/EMAIL_APP_PASSWORD before import so the
module's own startup guard doesn't reject a missing-but-not-needed-here
email config -- this file never calls send_email for real.
"""

import os
import types
import tempfile
from datetime import datetime, timezone

os.environ.setdefault("EMAIL_ADDRESS", "test@example.com")
os.environ.setdefault("EMAIL_APP_PASSWORD", "dummy-app-password")

import daily_summary as ds

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def make_order(symbol, side, qty, price, client_order_id=None, minute=0):
    return types.SimpleNamespace(
        symbol=symbol,
        side=types.SimpleNamespace(value=side),
        filled_qty=str(qty),
        filled_avg_price=str(price),
        filled_at=datetime(2026, 7, 24, 14, minute, tzinfo=timezone.utc),
        client_order_id=client_order_id,
    )


def test_extract_strategy():
    check("extracts the reason_key prefix from a tagged client_order_id",
          ds.extract_strategy("vwap_reversion-1753272000") == "vwap_reversion")
    check("handles a multi-word-ish reason_key with underscores correctly",
          ds.extract_strategy("gap_continuation-1753272000") == "gap_continuation")
    check("returns 'unknown' for None", ds.extract_strategy(None) == "unknown")
    check("returns 'unknown' for a malformed id with no separator", ds.extract_strategy("noSeparatorHere") == "unknown")


def test_pair_round_trip_trades_basic():
    orders = [
        make_order("TSLA", "buy", 1, 400.00, "breakout-1000", minute=0),
        make_order("TSLA", "sell", 1, 410.00, None, minute=30),
    ]
    completed, still_open = ds.pair_round_trip_trades(orders)
    check("pairs a simple buy+sell into one completed trade", len(completed) == 1 and len(still_open) == 0)
    if completed:
        t = completed[0]
        check("invested = qty * entry_price", t["invested"] == 400.00)
        check("pnl = (exit - entry) * qty", abs(t["pnl"] - 10.00) < 1e-9)
        check("strategy correctly read back from client_order_id", t["strategy"] == "breakout")


def test_pair_round_trip_trades_multiple_symbols():
    orders = [
        make_order("TSLA", "buy", 1, 400.00, "breakout-1000", minute=0),
        make_order("NVDA", "buy", 2, 200.00, "orb-1001", minute=1),
        make_order("TSLA", "sell", 1, 390.00, None, minute=30),
        make_order("NVDA", "sell", 2, 210.00, None, minute=31),
    ]
    completed, still_open = ds.pair_round_trip_trades(orders)
    check("pairs interleaved orders per-symbol correctly, not just sequentially",
          len(completed) == 2, f"got {len(completed)}")
    by_symbol = {t["symbol"]: t for t in completed}
    check("TSLA trade paired with its own exit, not NVDA's",
          "TSLA" in by_symbol and abs(by_symbol["TSLA"]["pnl"] - (-10.00)) < 1e-9)
    check("NVDA trade paired with its own exit, not TSLA's",
          "NVDA" in by_symbol and abs(by_symbol["NVDA"]["pnl"] - 20.00) < 1e-9)


def test_pair_round_trip_trades_still_open():
    orders = [
        make_order("TSLA", "buy", 1, 400.00, "breakout-1000", minute=0),
        # no matching sell today
    ]
    completed, still_open = ds.pair_round_trip_trades(orders)
    check("an unmatched buy is reported as still-open, not silently dropped",
          len(completed) == 0 and len(still_open) == 1)


def test_pair_round_trip_trades_sell_with_no_prior_buy_is_skipped():
    orders = [
        make_order("TSLA", "sell", 1, 410.00, None, minute=0),
    ]
    completed, still_open = ds.pair_round_trip_trades(orders)
    check("a sell with no prior buy today is skipped, not misattributed",
          len(completed) == 0 and len(still_open) == 0)


def test_build_email_body_totals():
    orders = [
        make_order("TSLA", "buy", 1, 400.00, "breakout-1000", minute=0),
        make_order("TSLA", "sell", 1, 410.00, None, minute=30),
        make_order("NVDA", "buy", 2, 200.00, "orb-1001", minute=1),
        make_order("NVDA", "sell", 2, 190.00, None, minute=31),
    ]
    completed, still_open = ds.pair_round_trip_trades(orders)
    subject, body = ds.build_email_body(completed, still_open, "2026-07-24")
    check("subject reflects the correct total P&L sign and trade count",
          "2026-07-24" in subject and "2 trades" in subject)
    check("body mentions both symbols traded", "TSLA" in body and "NVDA" in body)
    check("body includes both strategy names", "breakout" in body and "orb" in body)

    subject_empty, body_empty = ds.build_email_body([], [], "2026-07-24")
    check("an empty day still produces a valid 'no trades' email, not an error",
          "No trades today" in body_empty)


def test_open_position_counts_as_a_trade():
    """
    Regression for 2026-07-24: one TSLA share was bought and never sold,
    and the summary went out saying "0 trades" -- reading as "the bot did
    nothing" when it had actually put $311 to work. A position opened
    today is a trade whether or not it closed today.
    """
    orders = [make_order("TSLA", "buy", 1, 311.67, "vwap_reversion-1000", minute=0)]
    completed, still_open = ds.pair_round_trip_trades(orders)
    subject, body = ds.build_email_body(
        completed, still_open, "2026-07-24",
        open_positions={"TSLA": {"current_price": 313.30, "qty": 1.0}},
    )
    check("an open position counts as 1 trade, not 0", "1 trade" in subject)
    check("the subject does NOT claim zero trades", "0 trade" not in subject)
    check("the subject flags that it is still open", "still open" in subject.lower())
    check("the body names the symbol", "TSLA" in body)
    check("the body attributes the strategy", "vwap_reversion" in body)
    check("the body reports the amount invested", "311.67" in body)
    check("unrealized P&L is marked to market", "+1.63" in body)
    check("the body warns the position was left open", "WARNING" in body)


def test_open_position_without_live_price_still_reported():
    """If the positions lookup fails, the trade must still be reported --
    just without a mark-to-market number."""
    orders = [make_order("TSLA", "buy", 1, 311.67, "orb-1000", minute=0)]
    completed, still_open = ds.pair_round_trip_trades(orders)
    subject, body = ds.build_email_body(completed, still_open, "2026-07-24", open_positions={})
    check("still counted as a trade with no live price", "1 trade" in subject)
    check("invested amount still reported", "311.67" in body)
    check("P&L shown as n/a rather than a fabricated 0", "n/a" in body)


def _with_watchlist_state(payload):
    """Points daily_summary at a temp state file for one check."""
    import json as _json
    d = tempfile.mkdtemp()
    path = os.path.join(d, "watchlist_state.json")
    if payload is not None:
        with open(path, "w") as f:
            _json.dump(payload, f)
    original = ds.WATCHLIST_STATE_FILE
    ds.WATCHLIST_STATE_FILE = path
    return original


def test_scanner_health_flags_a_stale_scan():
    """
    Regression for the failure that hid for two days: the scanner returned
    nothing on every run, fell back to SYMBOLS, and left last_scan_time
    frozen at the last success. Every workflow run was green throughout.
    This is the literal state file from that outage.
    """
    original = _with_watchlist_state(
        {"active_watchlist": ["SMCI"], "last_scan_time": "2026-07-22T10:39:46.826489-04:00"})
    try:
        health = ds.build_scanner_health_lines("2026-07-24")
        subject, body = ds.build_email_body([], [], "2026-07-24", {}, health)
        check("a stale scan is reported as a WARNING",
              any("WARNING" in line for line in health))
        check("the warning names the date of the last good scan",
              any("2026-07-22" in line for line in health))
        check("the subject is flagged so it's visible without opening the email",
              subject.startswith("[CHECK ME]"))
        check("the warning reaches the email body", "WARNING" in body)
    finally:
        ds.WATCHLIST_STATE_FILE = original


def test_scanner_health_reports_ok_when_scan_is_current():
    original = _with_watchlist_state(
        {"active_watchlist": ["SAFT", "RNG"], "last_scan_time": "2026-07-24T10:05:00-04:00"})
    try:
        health = ds.build_scanner_health_lines("2026-07-24")
        subject, _ = ds.build_email_body([], [], "2026-07-24", {}, health)
        check("a same-day scan reports OK", any("OK" in line for line in health))
        check("no WARNING on a healthy day", not any("WARNING" in line for line in health))
        check("a healthy day is not flagged in the subject",
              not subject.startswith("[CHECK ME]"))
        check("the health line lists what was being watched",
              any("SAFT" in line for line in health))
    finally:
        ds.WATCHLIST_STATE_FILE = original


def test_scanner_health_handles_never_scanned_and_missing_file():
    original = _with_watchlist_state({"active_watchlist": [], "last_scan_time": None})
    try:
        health = ds.build_scanner_health_lines("2026-07-24")
        check("a never-successful scan is a WARNING",
              any("WARNING" in line for line in health))
    finally:
        ds.WATCHLIST_STATE_FILE = original

    original = _with_watchlist_state(None)  # file deliberately absent
    try:
        health = ds.build_scanner_health_lines("2026-07-24")
        check("a missing state file is reported, not crashed on", len(health) == 1)
    finally:
        ds.WATCHLIST_STATE_FILE = original


if __name__ == "__main__":
    tests = [
        test_extract_strategy,
        test_pair_round_trip_trades_basic,
        test_pair_round_trip_trades_multiple_symbols,
        test_pair_round_trip_trades_still_open,
        test_pair_round_trip_trades_sell_with_no_prior_buy_is_skipped,
        test_build_email_body_totals,
        test_open_position_counts_as_a_trade,
        test_open_position_without_live_price_still_reported,
        test_scanner_health_flags_a_stale_scan,
        test_scanner_health_reports_ok_when_scan_is_current,
        test_scanner_health_handles_never_scanned_and_missing_file,
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
