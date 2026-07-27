"""
Backtester
-----------
Replays the EXACT same strategy logic from strategy.py against months of
historical price data, in minutes instead of weeks. This is how we turn
"let's wait and see if this works" into "let's actually check."

Usage:
    py backtest.py                     Test the symbols in your .env, 60 days back
    py backtest.py TSLA NVDA           Test specific symbols, 60 days back
    py backtest.py TSLA --days 90      Test TSLA over the last 90 days

Every trade is simulated WITH the live bot's real stop-loss/take-profit
protection (using STOP_LOSS_PCT/TAKE_PROFIT_PCT, or ATR-based levels if
USE_ATR_STOPS=true) -- this used to also run a second "without risk
management" version side by side, but that comparison isn't something
this project needs to keep re-deriving on every run, so it's gone.
strategy.compute_stop_and_target is the single source of truth here,
same helper the live bot uses.

Position sizing mirrors the live bot too: risk-based by default
(strategy.compute_position_size), sized off a per-symbol simulated
equity curve that starts at TRADE_AMOUNT_USD and compounds as trades
close, rather than a flat dollar amount every time.

SPEED: indicators are computed ONCE per symbol up front (see
strategy.add_indicators), not recomputed on every growing window like
the original version of this file did -- that was accidentally O(n^2)
per symbol, since every indicator here only ever looks backward, so
recomputing them bar-by-bar on a growing slice was pure waste. Expect
roughly an order of magnitude faster than the old ~20s/symbol/60-days.

IMPORTANT LIMITATIONS (read this before trusting the numbers):
- Entries/exits assume you get filled at the exact bar's close price
  (or the exact stop/take-profit level for those exits). Real trading
  has slippage -- your real fill will often be slightly worse,
  especially on fast-moving or halted stocks like the ones this bot
  favors. Treat these results as an upper bound, not a guarantee.
- When a single bar's price range touches BOTH the stop-loss and the
  take-profit level, this assumes the stop-loss hit first (the more
  conservative, pessimistic assumption), since we don't have tick-by-
  tick data to know which actually happened first.
- It's possible to fool yourself by tuning settings until they fit
  what already happened. A strategy that worked great on the last 60
  days isn't guaranteed to work on the next 60.
- This tests each symbol in isolation -- it does NOT simulate
  portfolio-level controls that only make sense across your whole
  account at once, namely MAX_CONCURRENT_POSITIONS, MAX_PORTFOLIO_RISK_PCT,
  and the daily loss circuit breaker (all three live in trading_bot.py
  only). A single-symbol backtest has no notion of "how many other
  positions are open right now" or "today's total account P&L."
"""

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from strategy import (
    add_indicators, decide_signal_at, compute_stop_and_target, compute_position_size,
    stop_is_wider_than_noise,
    BAR_MINUTES, TRADE_AMOUNT_USD,
    FLATTEN_BEFORE_CLOSE, STOP_LOSS_PCT, TAKE_PROFIT_PCT, USE_ATR_STOPS,
    USE_RISK_BASED_SIZING, RISK_PER_TRADE_PCT,
    STOP_NEW_ENTRIES_MINUTES_BEFORE_CLOSE, ENTRY_BLACKOUT_START_MINUTES, ENTRY_BLACKOUT_END_MINUTES,
)

load_dotenv()
# .strip() defends against a trailing newline/whitespace in a pasted
# credential -- see trading_bot.py for the full explanation.
API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
SECRET_KEY = (os.getenv("ALPACA_SECRET_KEY") or "").strip()

if not API_KEY or not SECRET_KEY or "your_paper" in API_KEY:
    raise SystemExit("ERROR: Fill in your Alpaca PAPER API keys in .env first.")

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Used only if the real account can't be reached (see get_starting_equity)
# -- Alpaca paper accounts default to $100,000, so that's the fallback too.
DEFAULT_BACKTEST_EQUITY = float(os.getenv("DEFAULT_BACKTEST_EQUITY", 100_000))
# How far past the stop price a triggered stop actually fills, in ATRs.
# Measured from live fills on 2026-07-27 (three VEEE stop-outs, all
# 0.51-0.54 x ATR). Set to 0 to restore the old perfect-fill assumption,
# but be aware that assumption systematically flatters volatile stocks.
STOP_SLIPPAGE_ATR_FRACTION = float(os.getenv("STOP_SLIPPAGE_ATR_FRACTION", 0.5))


def get_starting_equity() -> float:
    """
    Uses your REAL Alpaca paper account equity as the starting capital
    for each symbol's simulated equity curve (each symbol is still
    tested in isolation, independently -- same simplification as
    before, just with a realistic capital base). $500 (the old flat-
    per-trade amount) was fine when position size was just "$500 every
    trade," but is far too small to be a stand-in for account equity
    once sizing is risk-based: 1% of $500 is $5, which doesn't buy even
    1 share of most stocks at a normal stop distance, so trades started
    rounding down to 0 shares and getting silently skipped.
    """
    try:
        account = trading_client.get_account()
        return float(account.equity)
    except Exception as e:
        print(f"Could not fetch real account equity ({e}), using ${DEFAULT_BACKTEST_EQUITY:,.0f} instead.")
        return DEFAULT_BACKTEST_EQUITY

# Widened from the original TSLA/NVDA/COIN after the first backtest showed
# those three are behaviorally correlated (high-beta growth/risk-sentiment
# names that tend to move together). AMD and PLTR added real diversity and
# performed well in a live-validated run (+24.9%, +4.4%). MSTR was tried
# too but dropped after that same run: worst symbol (-18.6%) with a 42%
# max drawdown, by far the riskiest thing in the set. See CLAUDE.md.
DEFAULT_SYMBOLS = "TSLA,NVDA,COIN,AMD,PLTR"


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")


def fetch_historical_bars(symbol: str, days_back: int) -> pd.DataFrame:
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
        start=datetime.now(timezone.utc) - timedelta(days=days_back),
    )
    bars = data_client.get_stock_bars(request).df
    if bars.empty:
        return bars
    bars = bars.reset_index()
    if "symbol" in bars.columns:
        bars = bars[bars["symbol"] == symbol].reset_index(drop=True)

    # Alpaca includes pre-market/after-hours bars by default, with no
    # request-level flag to exclude them. The live bot only ever checks
    # symbols while the market is genuinely open (9:30am-4pm ET), so a
    # backtest that includes extended-hours bars would be testing trades
    # that could never actually happen live. Filter them out here so the
    # backtest reflects reality.
    et_time = bars["timestamp"].dt.tz_convert(MARKET_TZ)
    is_regular_session = (
        ((et_time.dt.hour > 9) | ((et_time.dt.hour == 9) & (et_time.dt.minute >= 30)))
        & (et_time.dt.hour < 16)
    )
    bars = bars[is_regular_session].reset_index(drop=True)
    return bars


def minutes_until_close(timestamp) -> float:
    """Minutes remaining until 4:00pm ET on this bar's trading day."""
    et_time = timestamp.tz_convert(MARKET_TZ) if timestamp.tzinfo else timestamp.tz_localize(timezone.utc).tz_convert(MARKET_TZ)
    close = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
    return (close - et_time).total_seconds() / 60


# ---------------------------------------------------------------------------
# SIMULATION
# ---------------------------------------------------------------------------

def simulate(symbol: str, bars: pd.DataFrame, starting_equity: float) -> list[dict]:
    """
    Walks through the bars one at a time, calling the SAME decision logic
    the live bot uses (decide_signal_at), using only data up to and
    including the current bar (no lookahead). Indicators are computed
    ONCE up front via add_indicators -- every indicator here only looks
    backward, so this gives identical values to recomputing on each
    growing slice, just without doing the redundant work. Every position
    carries the live bot's real stop-loss/take-profit protection.

    Position size mirrors the live bot: risk-based by default, sized off
    a simulated equity curve that starts at starting_equity and compounds
    as trades close (falls back to flat, non-compounding TRADE_AMOUNT_USD
    sizing if USE_RISK_BASED_SIZING is off, same as live). starting_equity
    should be a realistic account size -- $500 (the old flat-per-trade
    amount) massively undersizes every position at 1% risk/trade on a
    normal-priced stock, to the point that most trades round down to 0
    shares and get skipped. See get_starting_equity() below.

    Returns a list of completed trades.
    """
    trades = []
    position = None  # None, or a dict describing the open simulated position
    enriched = add_indicators(bars)
    n = len(enriched)
    equity = starting_equity

    for i in range(n):
        if i < 1:
            continue
        current_bar = enriched.iloc[i]

        is_last_bar_of_day = (
            i == n - 1
            or enriched.iloc[i]["timestamp"].date() != enriched.iloc[i + 1]["timestamp"].date()
        )

        # --- Manage an existing position: check for an exit ---
        if position is not None:
            exit_price = None
            exit_reason = None

            # Conservative assumption: if both levels were touched in the
            # same bar, assume the stop-loss hit first.
            if current_bar["low"] <= position["stop_price"]:
                # A stop is a MARKET order once triggered -- it fills at
                # whatever is available next, not at the stop price. The
                # old code assumed a perfect fill, which quietly flattered
                # exactly the volatile stocks where stops slip worst, and
                # made high-ATR names look far more tradeable than they are.
                #
                # 0.5 x ATR is measured, not guessed: all three VEEE
                # stop-outs on 2026-07-27 slipped 0.51-0.54 x ATR
                # (stop 16.56 -> filled 15.90, 16.90 -> 16.31,
                # 15.87 -> 15.33). Capped at the bar's actual low, since
                # you cannot fill worse than the worst price that traded.
                atr_now = current_bar["atr"]
                slip = 0.0
                if not pd.isna(atr_now) and atr_now > 0:
                    slip = STOP_SLIPPAGE_ATR_FRACTION * atr_now
                exit_price = max(position["stop_price"] - slip, current_bar["low"])
                exit_reason = "stop-loss"
            elif current_bar["high"] >= position["take_profit_price"]:
                exit_price = position["take_profit_price"]
                exit_reason = "take-profit"

            if exit_price is None:
                signal, _reason_key, _reason = decide_signal_at(enriched, i)
                if signal == "SELL":
                    exit_price = current_bar["close"]
                    exit_reason = "strategy sell signal"
                elif FLATTEN_BEFORE_CLOSE and is_last_bar_of_day:
                    exit_price = current_bar["close"]
                    exit_reason = "end-of-day flatten"

            if exit_price is not None:
                pnl = (exit_price - position["entry_price"]) * position["qty"]
                pnl_pct = (exit_price / position["entry_price"] - 1) * 100
                equity += pnl
                trades.append({
                    "symbol": symbol,
                    "entry_time": position["entry_time"],
                    "entry_price": position["entry_price"],
                    "entry_reason": position["entry_reason"],
                    "exit_time": current_bar["timestamp"],
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "qty": position["qty"],
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                })
                position = None
                continue  # don't also open a fresh position on the same bar we just exited

        # --- Look for a new entry ---
        if position is None:
            if minutes_until_close(current_bar["timestamp"]) <= STOP_NEW_ENTRIES_MINUTES_BEFORE_CLOSE:
                continue  # too close to the bell to open something new today
            if ENTRY_BLACKOUT_START_MINUTES <= current_bar["minutes_since_open"] < ENTRY_BLACKOUT_END_MINUTES:
                continue  # historically weak entry window (see strategy.py)

            signal, reason_key, reason = decide_signal_at(enriched, i)
            if signal == "BUY":
                entry_price = current_bar["close"]
                stop_price, take_profit_price = compute_stop_and_target(entry_price, current_bar["atr"])
                # Same volatility guard the live bot applies -- a stop
                # inside the stock's own bar-to-bar noise isn't a stop.
                # Mirrored here so backtests can't flatter a setup the
                # live bot would refuse to take.
                if not stop_is_wider_than_noise(entry_price, current_bar["atr"], stop_price):
                    continue
                if USE_RISK_BASED_SIZING:
                    qty = compute_position_size(equity, entry_price, stop_price)
                else:
                    qty = int(TRADE_AMOUNT_USD // entry_price)
                if qty < 1:
                    continue  # not enough simulated capital/risk budget for even 1 share
                position = {
                    "entry_time": current_bar["timestamp"],
                    "entry_price": entry_price,
                    "entry_reason": reason_key,
                    "qty": qty,
                    "stop_price": stop_price,
                    "take_profit_price": take_profit_price,
                }

    return trades


def compute_stats(trades: list[dict], starting_capital: float) -> dict | None:
    if not trades:
        return None

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)

    equity = starting_capital
    peak = starting_capital
    max_dd = 0.0
    for t in trades:
        equity += t["pnl"]
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))

    return {
        "starting_capital": starting_capital,
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": len(wins) / len(trades) * 100,
        "total_pnl": total_pnl,
        "total_return_pct": total_pnl / starting_capital * 100,
        "avg_win_pct": float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0.0,
        "avg_loss_pct": float(np.mean([t["pnl_pct"] for t in losses])) if losses else 0.0,
        "max_drawdown_pct": max_dd,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
    }


def print_stats(stats: dict | None) -> None:
    if stats is None:
        print("    No trades were triggered in this period.")
        return
    print(f"    Trades: {stats['total_trades']} ({stats['wins']} wins / {stats['losses']} losses, "
          f"{stats['win_rate_pct']:.0f}% win rate)")
    print(f"    Total return: {stats['total_return_pct']:+.1f}%  (${stats['total_pnl']:+.2f} "
          f"on ${stats['starting_capital']:,.0f})")
    print(f"    Avg win: {stats['avg_win_pct']:+.1f}%  |  Avg loss: {stats['avg_loss_pct']:+.1f}%")
    print(f"    Max drawdown: {stats['max_drawdown_pct']:.1f}%  |  Profit factor: {stats['profit_factor']:.2f}")


def print_entry_reason_breakdown(trades: list[dict]) -> None:
    """Per-strategy trade counts and win rate, so individual strategies can be judged instead of one black box."""
    if not trades:
        return
    df = pd.DataFrame(trades)
    print("\n  By strategy:")
    for reason, group in df.groupby("entry_reason"):
        wins = (group["pnl"] > 0).sum()
        total = len(group)
        print(f"    {reason}: {total} trades, {wins}/{total} wins ({wins / total * 100:.0f}%), "
              f"total P&L ${group['pnl'].sum():+.2f}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the trading strategy against historical data.")
    parser.add_argument("symbols", nargs="*", help="Symbols to test (default: SYMBOLS from .env)")
    parser.add_argument("--days", type=int, default=60, help="How many days back to test (default: 60)")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols] if args.symbols else \
        [s.strip().upper() for s in os.getenv("SYMBOLS", DEFAULT_SYMBOLS).split(",") if s.strip()]

    starting_equity = get_starting_equity() if USE_RISK_BASED_SIZING else TRADE_AMOUNT_USD
    sizing_desc = (f"risk-based, {RISK_PER_TRADE_PCT:.1f}% of equity/trade" if USE_RISK_BASED_SIZING
                   else f"flat ${TRADE_AMOUNT_USD:.0f}/trade")
    print(f"Backtesting {', '.join(symbols)} over the last {args.days} days "
          f"({BAR_MINUTES}-minute bars, {sizing_desc}, starting at ${starting_equity:,.0f} "
          f"simulated equity per symbol)...")
    if USE_ATR_STOPS:
        print("Risk management: ATR-based stops (USE_ATR_STOPS=true)\n")
    else:
        print(f"Risk management: stop-loss -{STOP_LOSS_PCT:.0f}% / take-profit +{TAKE_PROFIT_PCT:.0f}%\n")
    print("=" * 70)

    all_trades = []

    for symbol in symbols:
        print(f"\n{symbol}")
        try:
            bars = fetch_historical_bars(symbol, args.days)
        except Exception as e:
            print(f"  Could not fetch historical data: {e}")
            continue

        if bars.empty:
            print("  No historical data returned -- skipping.")
            continue

        print(f"  ({len(bars)} bars loaded)")

        trades = simulate(symbol, bars, starting_equity)
        all_trades.extend(trades)

        print_stats(compute_stats(trades, starting_equity))
        print_entry_reason_breakdown(trades)

    if len(symbols) > 1:
        print("\n" + "=" * 70)
        print("COMBINED (all symbols)")
        combined_capital = starting_equity * len(symbols)
        print_stats(compute_stats(all_trades, combined_capital))
        print_entry_reason_breakdown(all_trades)

    # Save detailed trade-by-trade log
    if all_trades:
        pd.DataFrame(all_trades).to_csv("backtest_trades.csv", index=False)
        print("\nSaved trade-by-trade detail to backtest_trades.csv")

    print("\nUpload the CSV back to Claude if you want a deeper look at individual trades.")
