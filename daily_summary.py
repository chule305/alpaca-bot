"""
Daily Trade Summary Email
---------------------------
Standalone script, run once a day after market close (see
.github/workflows/daily_summary.yml): pulls today's completed trades
directly from Alpaca's real order history and emails a plain-text
summary (symbols traded, amount invested, P&L). Nothing local persists
between GitHub Actions runs except what's committed to git, so this
intentionally reads from Alpaca itself -- the one source of truth that's
always there -- rather than trying to reconstruct the day from logs.

Each buy order's client_order_id is tagged with the strategy that
triggered it (see place_buy_order in trading_bot.py) -- read back here
to attribute each trade to a strategy, since Alpaca itself has no
concept of "why" an order was placed.

KNOWN LIMITATION: pairs each BUY with the NEXT SELL for that symbol,
which is exact (not a heuristic) as long as the bot never holds more
than one position per symbol at once -- true today (see check_symbol's
current_qty == 0 gate) but would need revisiting if that ever changes.
A SELL with no prior BUY today (e.g. an overnight hold closing the next
morning, which shouldn't happen with FLATTEN_BEFORE_CLOSE=true) is
skipped rather than guessed at.
"""

import os
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.common.enums import Sort

load_dotenv()

# .strip() on every credential defends against a trailing newline/
# whitespace sneaking into a pasted secret (e.g. a GitHub Actions secret
# box) -- that produces a cryptic "Invalid header value"/SMTP auth error
# with no obvious link back to "check for a stray newline", so it's
# worth doing unconditionally. Confirmed this exact failure mode live:
# a trailing "\n" on ALPACA_API_KEY reproduces requests' InvalidHeader
# error exactly.
API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
SECRET_KEY = (os.getenv("ALPACA_SECRET_KEY") or "").strip()
EMAIL_ADDRESS = (os.getenv("EMAIL_ADDRESS") or "").strip()
EMAIL_APP_PASSWORD = (os.getenv("EMAIL_APP_PASSWORD") or "").strip()
EMAIL_TO = (os.getenv("EMAIL_TO") or EMAIL_ADDRESS).strip()
# Provider-agnostic -- defaults to Gmail's server, but EMAIL_SMTP_SERVER
# lets this send from any SMTP provider (e.g. smtp.mail.me.com for
# iCloud) without touching code. All major providers use port 587 with
# STARTTLS and require an app-specific password, not your normal one.
EMAIL_SMTP_SERVER = (os.getenv("EMAIL_SMTP_SERVER") or "smtp.gmail.com").strip()
EMAIL_SMTP_PORT = int((os.getenv("EMAIL_SMTP_PORT") or "587").strip())

if not API_KEY or not SECRET_KEY or "your_paper" in API_KEY:
    raise SystemExit("ERROR: Fill in your Alpaca PAPER API keys in .env first.")
if not EMAIL_ADDRESS or not EMAIL_APP_PASSWORD:
    raise SystemExit("ERROR: EMAIL_ADDRESS and EMAIL_APP_PASSWORD must be set "
                      "(GitHub secrets in CI, or .env locally) to send the summary email. "
                      "EMAIL_APP_PASSWORD is an APP-SPECIFIC password from your email "
                      "provider, NOT your normal account password.")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
MARKET_TZ = ZoneInfo("America/New_York")


def fetch_todays_filled_orders() -> list:
    """All FILLED orders (CLOSED also includes cancelled/rejected, so filter further) from today, ET calendar day, oldest first."""
    now_et = datetime.now(MARKET_TZ)
    start_of_day_et = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
    orders = trading_client.get_orders(
        filter=GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=start_of_day_et.astimezone(timezone.utc),
            until=now_et.astimezone(timezone.utc),
            limit=500,
            direction=Sort.ASC,
        )
    )
    return [o for o in orders if o.filled_at is not None and o.filled_qty and float(o.filled_qty) > 0]


def fetch_open_positions() -> dict[str, dict]:
    """
    Live data for anything still held, so a position opened today but not
    yet closed can be reported with a real current value instead of being
    treated as a non-event. Returns {} on failure -- the summary still
    goes out, just without live marks.
    """
    try:
        return {
            p.symbol: {"current_price": float(p.current_price), "qty": float(p.qty)}
            for p in trading_client.get_all_positions()
        }
    except Exception as e:
        print(f"(Could not fetch open positions -- reporting entry prices only: {e})")
        return {}


def extract_strategy(client_order_id: str | None) -> str:
    """client_order_id is "{reason_key}-{unix_timestamp}" (see place_buy_order) -- pull the reason_key back out."""
    if not client_order_id or "-" not in client_order_id:
        return "unknown"
    return client_order_id.rsplit("-", 1)[0]


def pair_round_trip_trades(orders: list) -> tuple[list[dict], list[dict]]:
    """
    Pairs each BUY fill with the NEXT SELL fill for the same symbol into
    a completed round-trip trade -- exact, not a heuristic, as long as
    the bot never holds more than one position per symbol at a time
    (see module docstring). Returns (completed_trades, still_open) --
    still_open is any BUY with no matching SELL yet today (shouldn't
    normally happen given the end-of-day flatten, but reported rather
    than silently dropped if it does).
    """
    by_symbol: dict[str, list] = {}
    for o in orders:
        by_symbol.setdefault(o.symbol, []).append(o)

    completed = []
    still_open = []
    for symbol, symbol_orders in by_symbol.items():
        pending_buy = None
        for o in symbol_orders:
            side = o.side.value if hasattr(o.side, "value") else str(o.side)
            qty = float(o.filled_qty)
            price = float(o.filled_avg_price)
            if side == "buy":
                if pending_buy is not None:
                    # A second buy before a matching sell for the same
                    # symbol -- shouldn't happen given the bot's own
                    # current_qty==0 gate, but don't silently lose track
                    # of the first one if it somehow does.
                    still_open.append(pending_buy)
                pending_buy = {
                    "symbol": symbol,
                    "strategy": extract_strategy(o.client_order_id),
                    "entry_time": o.filled_at,
                    "entry_price": price,
                    "qty": qty,
                }
            elif side == "sell" and pending_buy is not None:
                invested = pending_buy["qty"] * pending_buy["entry_price"]
                pnl = (price - pending_buy["entry_price"]) * pending_buy["qty"]
                pnl_pct = (price / pending_buy["entry_price"] - 1) * 100
                completed.append({
                    **pending_buy,
                    "exit_time": o.filled_at,
                    "exit_price": price,
                    "invested": invested,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                })
                pending_buy = None
        if pending_buy is not None:
            still_open.append(pending_buy)

    completed.sort(key=lambda t: t["entry_time"])
    return completed, still_open


def build_email_body(completed: list[dict], still_open: list[dict], today_str: str,
                      open_positions: dict[str, dict] | None = None) -> tuple[str, str]:
    """
    A position opened today counts as a trade whether or not it closed
    today. The first version of this counted only completed BUY->SELL
    round trips, so 2026-07-24 -- one TSLA share bought and still held --
    went out with the subject "0 trades", which read as "the bot did
    nothing" when it had in fact put money to work.
    """
    open_positions = open_positions or {}

    # Mark still-open positions to market so they can be reported with a
    # real number rather than just an entry price.
    open_rows = []
    for t in still_open:
        live = open_positions.get(t["symbol"])
        invested = t["qty"] * t["entry_price"]
        row = {**t, "invested": invested, "pnl": None, "pnl_pct": None}
        if live:
            # Computed from OUR entry and qty rather than read off the
            # position's own unrealized_pl, which covers the whole
            # position and could include shares from another day.
            row["pnl"] = (live["current_price"] - t["entry_price"]) * t["qty"]
            row["pnl_pct"] = (live["current_price"] / t["entry_price"] - 1) * 100
            row["current_price"] = live["current_price"]
        open_rows.append(row)

    realized_pnl = sum(t["pnl"] for t in completed)
    unrealized_pnl = sum(r["pnl"] for r in open_rows if r["pnl"] is not None)
    total_pnl = realized_pnl + unrealized_pnl
    total_invested = sum(t["invested"] for t in completed) + sum(r["invested"] for r in open_rows)
    trade_count = len(completed) + len(open_rows)

    wins = [t for t in completed if t["pnl"] > 0]
    win_rate = (len(wins) / len(completed) * 100) if completed else 0.0

    sign = "+" if total_pnl >= 0 else ""
    noun = "trade" if trade_count == 1 else "trades"
    still_held = f", {len(open_rows)} still open" if open_rows else ""
    subject = (f"[PAPER] Trading bot -- {today_str}: {sign}${total_pnl:,.2f} "
                f"({trade_count} {noun}{still_held})")

    lines = [
        f"Daily trading summary for {today_str} (PAPER account -- no real money)",
        "=" * 68,
        "",
    ]

    if not trade_count:
        lines.append("No trades today.")
    else:
        lines.append(f"{'Symbol':<8}{'Strategy':<18}{'Invested':>12}{'P&L':>12}{'P&L %':>9}  Status")
        lines.append("-" * 68)
        for t in completed:
            lines.append(
                f"{t['symbol']:<8}{t['strategy']:<18}${t['invested']:>10,.2f}"
                f"  {t['pnl']:>+9,.2f}  {t['pnl_pct']:>+6.2f}%  closed"
            )
        for r in open_rows:
            if r["pnl"] is None:
                lines.append(
                    f"{r['symbol']:<8}{r['strategy']:<18}${r['invested']:>10,.2f}"
                    f"  {'n/a':>9}  {'n/a':>7}  STILL OPEN"
                )
            else:
                lines.append(
                    f"{r['symbol']:<8}{r['strategy']:<18}${r['invested']:>10,.2f}"
                    f"  {r['pnl']:>+9,.2f}  {r['pnl_pct']:>+6.2f}%  STILL OPEN"
                )
        lines.append("-" * 68)
        lines.append(f"{'TOTAL':<26}${total_invested:>10,.2f}  {total_pnl:>+9,.2f}")
        lines.append("")
        lines.append(f"Trades: {trade_count} | Total invested: ${total_invested:,.2f} | "
                      f"Total P&L: ${total_pnl:+,.2f}")
        if completed:
            lines.append(f"Closed: {len(completed)} | Win rate: {win_rate:.0f}% | "
                          f"Realized P&L: ${realized_pnl:+,.2f}")
        if open_rows:
            lines.append(f"Open:   {len(open_rows)} | Unrealized P&L: ${unrealized_pnl:+,.2f} "
                          f"(marked at the price when this email was sent, not a final result)")
            lines.append("")
            lines.append(f"WARNING: {len(open_rows)} position(s) were left open overnight. "
                          f"FLATTEN_BEFORE_CLOSE is meant to close everything ~10 min before the "
                          f"bell, so this means the bot was not running during that window -- "
                          f"worth checking the Actions tab.")

    lines.append("")
    lines.append("-- Sent automatically by the trading bot's daily_summary.py")
    return subject, "\n".join(lines)


def send_email(subject: str, body: str) -> None:
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_TO

    with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
        server.send_message(msg)


if __name__ == "__main__":
    today_str = datetime.now(MARKET_TZ).strftime("%Y-%m-%d")
    orders = fetch_todays_filled_orders()
    completed, still_open = pair_round_trip_trades(orders)
    open_positions = fetch_open_positions() if still_open else {}
    subject, body = build_email_body(completed, still_open, today_str, open_positions)
    print(body)
    send_email(subject, body)
    print(f"\nEmail sent to {EMAIL_TO}.")
