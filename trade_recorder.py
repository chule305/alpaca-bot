"""
Trade Recorder
----------------
Writes a structured, append-only row for every trade the bot actually
places, into `trades.csv`.

WHY THIS EXISTS SEPARATELY FROM ALPACA'S ORDER HISTORY:
Alpaca can tell you *what* was traded and at what price. It cannot tell
you *why* -- what RSI, ADX, VWAP distance, ATR or relative volume looked
like at the instant the decision was made. That context only exists for
the microsecond the bot is looking at it, and it's exactly what's needed
to answer the questions that actually improve the strategy: "does
vwap_reversion only work when ADX is low?", "are our losers all entries
made in the first 15 minutes?", "is the ATR-based stop too tight on
high-volatility names?".

`client_order_id` already smuggles the strategy name into Alpaca (see
place_buy_order), but there's nowhere to put twenty indicator values, so
they get recorded here instead.

DESIGN NOTES:
- Append-only CSV, header written once. Survives GitHub Actions' fresh
  checkout per run because the workflow commits it back, same as the
  state JSON files.
- Every field is written even when unknown (as an empty cell) so the
  column layout never shifts between rows -- a CSV whose columns move
  around mid-file is painful to analyze later.
- Recording must NEVER break trading. Every entry point is wrapped so a
  disk error or a bad value results in a logged warning and nothing
  more; a bot that refuses to trade because it couldn't write a log
  row would be a far worse failure than a missing row.
"""

import os
import csv
import logging
from datetime import datetime, timezone

log = logging.getLogger()

TRADE_HISTORY_FILE = os.getenv("TRADE_HISTORY_FILE", "trades.csv").strip()

# Order matters -- this is the CSV column order, and it's append-only.
# Add new fields at the END so existing rows stay readable.
FIELDS = [
    # What happened
    "timestamp_utc",
    "trading_day_et",
    "symbol",
    "action",           # BUY / SELL / FLATTEN
    "strategy",         # reason_key that triggered it
    "reason",           # the human-readable explanation logged at the time
    # The money
    "qty",
    "price",
    "notional",
    "stop_loss",
    "take_profit",
    "sizing_style",
    "equity",
    # Portfolio state at the moment of the decision
    "open_positions",
    "minutes_since_open",
    # Identifiers, for tying back to Alpaca's own records
    "order_id",
    "client_order_id",
    # Decision context -- the part that exists nowhere else
    "close",
    "rsi",
    "adx",
    "atr",
    "ema_fast",
    "ema_slow",
    "vwap",
    "volume",
    "rvol_avg_volume",
    "breakout_recent_high",
    "orb_high",
    "prior_session_close",
]

# Columns copied straight off the indicator dataframe row, when one is
# available. Kept as a list so adding an indicator is a one-line change.
CONTEXT_COLUMNS = [
    "close", "rsi", "adx", "atr", "ema_fast", "ema_slow", "vwap", "volume",
    "rvol_avg_volume", "breakout_recent_high", "orb_high", "prior_session_close",
    "minutes_since_open",
]


def extract_strategy(client_order_id: str | None) -> str:
    """client_order_id is "{reason_key}-{unix_timestamp}" (see place_buy_order) -- pull the reason_key back out.

    Lives here (not in daily_summary.py or trading_bot.py) so both can import it
    without a circular/side-effect dependency -- daily_summary.py raises
    SystemExit at import time if email env vars aren't set, so trading_bot.py
    must never import it directly.
    """
    if not client_order_id or "-" not in client_order_id:
        return "unknown"
    return client_order_id.rsplit("-", 1)[0]


def _clean(value):
    """CSV-safe scalar: NaN/None become empty, floats get sane precision."""
    if value is None:
        return ""
    try:
        # Catches numpy NaN, pandas NA, and float('nan') alike.
        if value != value:
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def extract_context(enriched, index) -> dict:
    """
    Pulls the indicator values at the decision bar out of the enriched
    dataframe. Returns {} if anything is off rather than raising --
    missing context is acceptable, a failed trade is not.
    """
    context = {}
    if enriched is None or index is None:
        return context
    for column in CONTEXT_COLUMNS:
        try:
            if column in enriched.columns:
                context[column] = _clean(enriched[column].iat[index])
        except Exception:
            continue
    return context


def record_trade(action: str, symbol: str, trading_day_et: str = "", **fields) -> None:
    """
    Appends one row describing a trade. Never raises.

    Pass `context=` (indicator values) and/or `details=` (order sizing and
    bracket levels) as dicts; any other keyword matching a name in FIELDS
    is written directly.

    Dict arguments are passed as dicts rather than unpacked by the caller
    on purpose. `record_trade(**something)` would evaluate the unpacking
    at the CALL SITE, outside this function's try/except -- and inside
    check_symbol that lands in the generic order-error handler, which
    would log "Order failed" for an order that had already succeeded.
    Merging them in here keeps every failure mode inside the safety net.
    """
    try:
        row = {name: "" for name in FIELDS}
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row["trading_day_et"] = trading_day_et
        row["symbol"] = symbol
        row["action"] = action

        for source in (fields.pop("context", None), fields.pop("details", None), fields):
            if not source:
                continue
            try:
                items = source.items()
            except Exception:
                continue  # not a mapping -- skip it rather than blow up
            for key, value in items:
                if key in row:
                    row[key] = _clean(value)

        write_header = not os.path.exists(TRADE_HISTORY_FILE) or os.path.getsize(TRADE_HISTORY_FILE) == 0
        with open(TRADE_HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        # Deliberately swallowed -- see the module docstring.
        log.warning(f"Could not record trade to {TRADE_HISTORY_FILE}: {e}")
