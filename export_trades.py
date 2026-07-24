"""
Trade History Exporter
------------------------
A standalone, one-off script (separate from trading_bot.py) that pulls
your full order history and account equity curve from Alpaca and saves
them as CSV files you can share for analysis.

Run it anytime with:
    py export_trades.py

IMPORTANT NOTE ON COMPLETENESS: orders that were rejected immediately at
submission time (like the "not fractionable" or "trading halt" errors
you've seen) never become real Alpaca order records -- they're rejected
before Alpaca ever creates one. So this export won't show those specific
failed attempts. trading_log.txt remains the record for those -- keep
sharing relevant excerpts from it alongside this export for the full
picture.
"""

import os
import csv
from datetime import datetime
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest
from alpaca.trading.enums import QueryOrderStatus

load_dotenv()

# .strip() defends against a trailing newline/whitespace in a pasted
# credential -- see trading_bot.py for the full explanation.
API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
SECRET_KEY = (os.getenv("ALPACA_SECRET_KEY") or "").strip()

if not API_KEY or not SECRET_KEY or "your_paper" in API_KEY:
    raise SystemExit("ERROR: Fill in your Alpaca PAPER API keys in .env first.")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# ---------------------------------------------------------------------------
# Export 1: full order history (every order Alpaca actually accepted)
# ---------------------------------------------------------------------------

print("Fetching order history...")
orders = trading_client.get_orders(
    filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500)
)
orders = sorted(orders, key=lambda o: o.submitted_at)

orders_file = "orders_export.csv"
with open(orders_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "symbol", "side", "order_type", "status", "qty", "filled_qty",
        "notional", "filled_avg_price", "submitted_at", "filled_at"
    ])
    for o in orders:
        writer.writerow([
            o.symbol,
            o.side.value if o.side else "",
            o.order_type.value if o.order_type else "",
            o.status.value if o.status else "",
            o.qty,
            o.filled_qty,
            o.notional,
            o.filled_avg_price,
            o.submitted_at,
            o.filled_at,
        ])

print(f"Saved {len(orders)} orders to {orders_file}")

# ---------------------------------------------------------------------------
# Export 2: account equity curve over time (best-effort -- secondary)
# ---------------------------------------------------------------------------

try:
    print("Fetching equity history...")
    history = trading_client.get_portfolio_history(
        GetPortfolioHistoryRequest(period="1M", timeframe="15Min")
    )
    equity_file = "equity_history_export.csv"
    with open(equity_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "equity", "profit_loss", "profit_loss_pct"])
        for i, ts in enumerate(history.timestamp or []):
            writer.writerow([
                datetime.fromtimestamp(ts),
                history.equity[i] if history.equity else "",
                history.profit_loss[i] if history.profit_loss else "",
                history.profit_loss_pct[i] if history.profit_loss_pct else "",
            ])
    print(f"Saved equity history to {equity_file}")
except Exception as e:
    print(f"(Skipped equity history export -- not critical: {e})")

print("\nDone. Upload orders_export.csv (and equity_history_export.csv, if it")
print("was created) back to Claude, along with any trading_log.txt excerpts")
print("you want analyzed alongside them.")
