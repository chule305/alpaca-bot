"""
Tests for trade_recorder.py.

The load-bearing property here isn't formatting -- it's that recording a
trade can NEVER break trading. A bot that refuses to place an order
because it couldn't append a CSV row would be a far worse failure than a
missing row, so the "never raises" tests matter more than the rest.
"""

import os
import csv
import tempfile

import numpy as np
import pandas as pd

import trade_recorder as tr

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def _fresh_file():
    path = os.path.join(tempfile.mkdtemp(), "trades.csv")
    tr.TRADE_HISTORY_FILE = path
    return path


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_writes_header_once_then_appends():
    path = _fresh_file()
    tr.record_trade("BUY", "TSLA", trading_day_et="2026-07-27", strategy="orb", qty=3, price=100.0)
    tr.record_trade("SELL", "TSLA", trading_day_et="2026-07-27", strategy="orb", qty=3, price=110.0)

    with open(path, encoding="utf-8") as f:
        header_count = sum(1 for line in f if line.startswith("timestamp_utc"))
    rows = _rows(path)
    check("header is written exactly once, not per row", header_count == 1)
    check("both trades are recorded", len(rows) == 2)
    check("first row is the buy", rows[0]["action"] == "BUY" and rows[0]["symbol"] == "TSLA")
    check("second row is the sell", rows[1]["action"] == "SELL")
    check("strategy attribution is preserved", rows[0]["strategy"] == "orb")


def test_column_layout_never_shifts():
    """A CSV whose columns move mid-file is painful to analyze later, so
    every field is written even when unknown."""
    path = _fresh_file()
    tr.record_trade("BUY", "AAA", strategy="breakout", qty=1, price=10.0,
                    context={"rsi": 55.0, "adx": 30.0})
    tr.record_trade("FLATTEN", "BBB")  # almost nothing known
    rows = _rows(path)
    check("every row has the full column set",
          all(set(r.keys()) == set(tr.FIELDS) for r in rows))
    check("unknown fields are empty, not missing", rows[1]["rsi"] == "")
    check("known context is written", rows[0]["rsi"] == "55")


def test_nan_and_none_become_empty_cells():
    path = _fresh_file()
    tr.record_trade("BUY", "AAA", qty=None, price=float("nan"),
                    context={"rsi": np.nan, "adx": 25.5})
    row = _rows(path)[0]
    check("None becomes an empty cell", row["qty"] == "")
    check("float NaN becomes an empty cell", row["price"] == "")
    check("numpy NaN becomes an empty cell", row["rsi"] == "")
    check("a real value alongside NaN still lands", row["adx"] == "25.5")


def test_recording_never_raises():
    """The whole point: a broken recorder must not stop a trade."""
    tr.TRADE_HISTORY_FILE = os.path.join(tempfile.mkdtemp(), "no_such_dir", "trades.csv")
    raised = False
    try:
        tr.record_trade("BUY", "TSLA", qty=1, price=100.0)
    except Exception:
        raised = True
    check("an unwritable path is swallowed, not raised", not raised)

    path = _fresh_file()
    raised = False
    try:
        # An object that explodes on comparison and formatting alike.
        class Hostile:
            def __eq__(self, other): raise RuntimeError("boom")
            def __ne__(self, other): raise RuntimeError("boom")
            def __str__(self): raise RuntimeError("boom")
        tr.record_trade("BUY", "TSLA", qty=Hostile())
    except Exception:
        raised = True
    check("a hostile value is swallowed, not raised", not raised)


def test_bad_details_cannot_masquerade_as_a_failed_order():
    """
    check_symbol calls this right after a SUCCESSFUL order, from inside a
    try block whose handler logs "Order failed". So a recorder blowing up
    on malformed details would report a failure for an order that had
    already gone through -- the worst kind of wrong log line. Dicts are
    passed in rather than unpacked at the call site to keep that
    impossible; this pins the behavior.
    """
    path = _fresh_file()
    raised = False
    try:
        tr.record_trade("BUY", "TSLA", strategy="orb",
                        details="not a mapping at all", context=None)
    except Exception:
        raised = True
    check("non-mapping details are ignored, not raised", not raised)
    rows = _rows(path)
    check("the trade is still recorded despite bad details", len(rows) == 1)
    check("the fields that WERE valid still land", rows[0]["strategy"] == "orb")


def test_extract_context_reads_the_decision_bar():
    df = pd.DataFrame({
        "close": [10.0, 11.0, 12.0],
        "rsi": [40.0, 50.0, 60.0],
        "adx": [20.0, 25.0, 30.0],
        "vwap": [10.5, 11.5, 12.5],
    })
    context = tr.extract_context(df, 2)
    check("reads values at the requested bar, not the first",
          context["rsi"] == "60" and context["close"] == "12")
    check("picks up every present indicator column",
          {"close", "rsi", "adx", "vwap"} <= set(context))
    check("ignores indicator columns that aren't present",
          "atr" not in context)

    check("a missing dataframe yields no context rather than an error",
          tr.extract_context(None, 0) == {})
    check("an out-of-range index yields no context rather than an error",
          isinstance(tr.extract_context(df, 999), dict))


if __name__ == "__main__":
    tests = [
        test_writes_header_once_then_appends,
        test_column_layout_never_shifts,
        test_nan_and_none_become_empty_cells,
        test_recording_never_raises,
        test_bad_details_cannot_masquerade_as_a_failed_order,
        test_extract_context_reads_the_decision_bar,
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
