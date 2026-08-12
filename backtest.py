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
import io
import urllib.request
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
    stop_is_wider_than_noise, compute_daily_trend_map, vwap_reversion_volume_confirms,
    breakout_invalidated_at,
    BAR_MINUTES, TRADE_AMOUNT_USD,
    FLATTEN_BEFORE_CLOSE, STOP_LOSS_PCT, TAKE_PROFIT_PCT, USE_ATR_STOPS,
    USE_RISK_BASED_SIZING, RISK_PER_TRADE_PCT,
    STOP_NEW_ENTRIES_MINUTES_BEFORE_CLOSE, ENTRY_BLACKOUT_START_MINUTES, ENTRY_BLACKOUT_END_MINUTES,
    ADX_TREND_THRESHOLD, RSI_PERIOD,
    USE_VOLATILITY_SCALED_SIZING, VOLATILITY_SCALED_REDUCED_USD,
    USE_CONVICTION_SIZING, HIGH_CONVICTION_STRATEGIES, CONVICTION_BOOST_USD,
    USE_BREAKOUT_INVALIDATION_EXIT, USE_CLOSE_BEYOND_LEVEL_CONFIRMATION,
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

# Mirrors trading_bot.py's USE_MULTI_TIMEFRAME_FILTER -- SCANNER PICKS
# ONLY (non-S&P-500 symbols). See that file's comment for the full
# reasoning and the 90-day comparison this is based on. A one-shot fetch
# here (not the TTL-cached version trading_bot.py uses), since a
# backtest run fetches once and exits rather than running for hours.
USE_MULTI_TIMEFRAME_FILTER = os.getenv("USE_MULTI_TIMEFRAME_FILTER", "true").strip().lower() in ("1", "true", "yes")
# Mirrors trading_bot.py's USE_VWAP_VOLUME_CONFIRMATION -- SCANNER PICKS
# ONLY, same S&P 500 exemption and same reasoning (see
# VWAP_REVERSION_MIN_VOLUME_MULT's comment in strategy.py).
USE_VWAP_VOLUME_CONFIRMATION = os.getenv("USE_VWAP_VOLUME_CONFIRMATION", "true").strip().lower() in ("1", "true", "yes")
SP500_LIST_URL = os.getenv(
    "SP500_LIST_URL",
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
)

# Broad-market regime gate: veto new long entries in EVERY symbol when
# SPY itself is in a confirmed downtrend on this same BAR_MINUTES
# timeframe. Deliberately reuses the exact ADX/trend machinery
# strategy.py already computes for every symbol (ADX_TREND_THRESHOLD,
# the ema_fast/ema_slow pair trend_following_at reads) rather than
# inventing a new indicator -- "trending down" here means precisely what
# it means for any other symbol's own trend_following regime: ADX >=
# ADX_TREND_THRESHOLD (a confirmed trend, not chop) AND ema_fast <
# ema_slow (that trend pointing down). Applied externally, not inside
# strategy.py's decision functions, for the same reason the S&P-500-
# membership gates above (USE_MULTI_TIMEFRAME_FILTER,
# USE_VWAP_VOLUME_CONFIRMATION) live here: "is a DIFFERENT symbol's tape
# down" is operational/cross-symbol context, and strategy.py stays a
# pure, single-symbol, network-free decision file by design.
#
# Default FLIPPED TO OFF after a 90-day backtest (2026-08-06) came back
# net negative on BOTH universes, on every combined metric at once --
# not a tradeoff, a clean loss:
#
#              trades   win rate   total return   profit factor   max DD
#   megacap:   99->69    60%->58%   +7.6%->+4.9%    1.79->1.74     1.7%->1.8%
#   scanner:   96->70    49%->46%   +1.3%->+0.9%    1.15->1.12     1.8%->2.6%
#
# The theory (don't fight the broad market's own tape) is reasonable,
# but ADX>=25-and-falling-EMAs on SPY specifically didn't turn out to be
# a clean enough read of "bad time to open a long" -- it vetoed real
# winners along with the losers it was meant to catch, cut trade count
# by roughly a third in both universes, and left profit factor and max
# drawdown no better (worse, on the scanner side) than just taking every
# signal. Kept in code and toggleable, same as USE_RVOL_SPIKE/
# USE_ROSS_HOOK above -- one 90-day window on two symbol sets isn't the
# final word, and a different ADX threshold or a broader index (QQQ?)
# might tell a different story. Re-test before re-enabling.
USE_SPY_REGIME_GATE = os.getenv("USE_SPY_REGIME_GATE", "false").strip().lower() in ("1", "true", "yes")
SPY_REGIME_GATE_SYMBOL = os.getenv("SPY_REGIME_GATE_SYMBOL", "SPY").strip().upper()

# Duplicated from trading_bot.py's SECTOR_MAP -- backtest.py deliberately
# doesn't import trading_bot.py (same reasoning as fetch_sp500_symbols_once
# above being its own one-shot version of trading_bot.py's TTL-cached
# fetch_sp500_symbols, rather than a cross-import: this file is meant to
# run standalone for a single CLI invocation, not pull in the live bot's
# whole module-level setup). Keep this in sync by hand if SECTOR_MAP
# there ever changes -- see that file's own comment for the full sourcing
# rationale on why it's not exhaustive.
SECTOR_MAP: dict[str, str] = {
    "NVDA": "Information Technology", "AMD": "Information Technology",
    "INTC": "Information Technology", "QCOM": "Information Technology",
    "AVGO": "Information Technology", "TXN": "Information Technology",
    "MU": "Information Technology", "AMAT": "Information Technology",
    "SMCI": "Information Technology", "ARM": "Information Technology",
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "CRM": "Information Technology", "ORCL": "Information Technology",
    "ADBE": "Information Technology", "CSCO": "Information Technology",
    "IBM": "Information Technology", "PLTR": "Information Technology",
    "NOW": "Information Technology", "PANW": "Information Technology",
    "SOXL": "Information Technology", "SOXS": "Information Technology",
    "GOOGL": "Communication Services", "GOOG": "Communication Services",
    "META": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services", "TMUS": "Communication Services",
    "TSLA": "Consumer Discretionary", "AMZN": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "BKNG": "Consumer Discretionary",
    "COIN": "Financials", "JPM": "Financials", "V": "Financials",
    "MA": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "AXP": "Financials", "SCHW": "Financials",
    "UNH": "Health Care", "JNJ": "Health Care", "LLY": "Health Care",
    "PFE": "Health Care", "MRK": "Health Care", "ABBV": "Health Care",
    "ABT": "Health Care", "TMO": "Health Care", "DHR": "Health Care",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "MMM": "Industrials", "UPS": "Industrials", "HON": "Industrials",
    "RTX": "Industrials", "LMT": "Industrials",
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples", "PM": "Consumer Staples",
}

# One SPDR sector ETF per GICS sector name in SECTOR_MAP -- see
# trading_bot.py's own SECTOR_ETF_MAP for the full comment.
SECTOR_ETF_MAP: dict[str, str] = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Consumer Discretionary": "XLY",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

_sector_cache: dict[str, str | None] = {}


def get_symbol_sector(symbol: str) -> str | None:
    """
    One-shot duplicate of trading_bot.py's get_symbol_sector (SECTOR_MAP
    first, then a live get_asset() fallback) -- see the comment on
    SECTOR_MAP above for why this file keeps its own copy instead of
    importing trading_bot.py.
    """
    if symbol in _sector_cache:
        return _sector_cache[symbol]
    sector = SECTOR_MAP.get(symbol)
    if sector is None:
        try:
            asset = trading_client.get_asset(symbol)
            sector = getattr(asset, "sector", None) or getattr(asset, "industry", None) or None
        except Exception:
            sector = None
    _sector_cache[symbol] = sector
    return sector


def get_sector_etf(symbol: str) -> str | None:
    """The SPDR sector ETF for `symbol`'s own sector, or None if either
    lookup comes up empty -- see trading_bot.py's get_sector_etf."""
    sector = get_symbol_sector(symbol)
    if sector is None:
        return None
    return SECTOR_ETF_MAP.get(sector)


# Sector-relative mean-reversion filter -- mirrors trading_bot.py's
# USE_SECTOR_RELATIVE_MEAN_REVERSION exactly (same toggle name, same
# defaults, same research motivation). See that file's comment for the
# full reasoning, including why this is NOT the same idea as
# USE_SPY_REGIME_GATE above despite both comparing against an external
# reference series, and the 2026-08-11 backtest result this shipped with.
USE_SECTOR_RELATIVE_MEAN_REVERSION = os.getenv("USE_SECTOR_RELATIVE_MEAN_REVERSION", "false").strip().lower() in ("1", "true", "yes")
SECTOR_RELATIVE_LOOKBACK_BARS = int(os.getenv("SECTOR_RELATIVE_LOOKBACK_BARS", RSI_PERIOD))
SECTOR_RELATIVE_MIN_UNDERPERFORMANCE_PCT = float(os.getenv("SECTOR_RELATIVE_MIN_UNDERPERFORMANCE_PCT", 2.0))


def fetch_sp500_symbols_once() -> set:
    """One-shot version of trading_bot.py's fetch_sp500_symbols -- no TTL
    cache needed for a single CLI run. Returns an empty set on failure,
    which means the multi-timeframe filter degrades to applying to EVERY
    symbol (the validated-safe direction -- see compute_daily_trend_map)
    rather than silently skipping it for a run that happens to include a
    real S&P 500 name."""
    try:
        with urllib.request.urlopen(SP500_LIST_URL, timeout=10) as response:
            text = response.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(text))
        return {s.strip().upper() for s in df["Symbol"] if isinstance(s, str) and s.strip()}
    except Exception as e:
        print(f"Could not fetch the S&P 500 list ({e}) -- multi-timeframe filter will "
              f"apply to every symbol this run rather than skipping S&P 500 names.")
        return set()


def fetch_daily_trend_map(symbol: str, days_back: int) -> dict:
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=datetime.now(timezone.utc) - timedelta(days=days_back + 40),  # extra EMA warmup
    )
    daily = data_client.get_stock_bars(request).df
    if daily.empty:
        return {}
    daily = daily.reset_index()
    if "symbol" in daily.columns:
        daily = daily[daily["symbol"] == symbol].reset_index(drop=True)
    return compute_daily_trend_map(daily)


def fetch_spy_regime_bars(days_back: int) -> pd.DataFrame | None:
    """
    SPY's OWN bars over the same date range and BAR_MINUTES timeframe
    being tested, enriched via add_indicators() ONCE per run -- every
    symbol's simulate() call reuses this same dataframe rather than each
    one re-fetching/re-computing SPY's regime for itself. Returns None on
    a fetch failure or empty response; callers treat that as "gate
    disabled this run" (fail OPEN), not "block everything" -- SPY itself
    failing to fetch means something is wrong with market data broadly,
    which isn't a reason to silently veto every other symbol's entries on
    a filter that was never actually evaluated.
    """
    try:
        bars = fetch_historical_bars(SPY_REGIME_GATE_SYMBOL, days_back)
    except Exception as e:
        print(f"SPY regime gate: could not fetch {SPY_REGIME_GATE_SYMBOL} bars ({e}) -- gate disabled this run.")
        return None
    if bars.empty:
        print(f"SPY regime gate: no {SPY_REGIME_GATE_SYMBOL} data returned -- gate disabled this run.")
        return None
    return add_indicators(bars)


def compute_spy_bearish_at_bars(spy_enriched: pd.DataFrame, bar_timestamps: pd.Series) -> np.ndarray:
    """
    For each timestamp in bar_timestamps (one symbol's own bars), looks
    up whether SPY's OWN regime was reading a confirmed downtrend as of
    the most recent SPY bar at or before that timestamp -- an as-of
    backward join, not an equality join, because a thinly-traded scanner
    pick can be missing bars SPY has (a halt, a data gap), so timestamps
    between the two series don't always line up exactly. "Backward" is
    what keeps this lookahead-free: a symbol's bar can only ever see
    SPY's already-closed bars, never a future one.
    """
    spy_regime = spy_enriched[["timestamp", "adx", "ema_fast", "ema_slow"]].copy()
    spy_regime["spy_bearish_trend"] = (
        (spy_regime["adx"] >= ADX_TREND_THRESHOLD) & (spy_regime["ema_fast"] < spy_regime["ema_slow"])
    )
    aligned = pd.merge_asof(
        pd.DataFrame({"timestamp": bar_timestamps}),
        spy_regime[["timestamp", "spy_bearish_trend"]],
        on="timestamp", direction="backward",
    )
    # No prior SPY bar yet (this symbol's history starts before SPY's, or
    # SPY was still warming up its own ADX/EMA) -- unknown reads as "not
    # bearish" so the gate degrades to a no-op rather than a lookahead-
    # free but wrong "always block" for the warmup period.
    return aligned["spy_bearish_trend"].fillna(False).to_numpy()


def fetch_sector_etf_bars(symbols: list[str], days_back: int) -> dict[str, pd.DataFrame]:
    """
    Fetches each distinct sector ETF this run's `symbols` actually needs
    (via SECTOR_MAP/SECTOR_ETF_MAP), ONE fetch per distinct ETF rather
    than one per symbol -- several megacap symbols commonly share a
    sector (e.g. NVDA/AMD both map to XLK), so this avoids re-fetching
    the same ETF's bars redundantly. Only called at all when
    USE_SECTOR_RELATIVE_MEAN_REVERSION is on.
    """
    needed_etfs = sorted({
        SECTOR_ETF_MAP[sector]
        for sector in (get_symbol_sector(s) for s in symbols)
        if sector in SECTOR_ETF_MAP
    })
    etf_bars = {}
    for etf_symbol in needed_etfs:
        try:
            bars = fetch_historical_bars(etf_symbol, days_back)
        except Exception as e:
            print(f"Sector-relative mean reversion: could not fetch {etf_symbol} bars ({e}) -- "
                  f"filter disabled this run for symbols in that sector.")
            continue
        if not bars.empty:
            etf_bars[etf_symbol] = bars
    return etf_bars


def compute_sector_relative_return_at_bars(etf_bars: pd.DataFrame, bar_timestamps: pd.Series) -> np.ndarray:
    """
    For each timestamp in bar_timestamps (one candidate symbol's own
    bars), looks up its sector ETF's own SECTOR_RELATIVE_LOOKBACK_BARS-bar
    trailing return as of the most recent ETF bar at or before that
    timestamp -- same as-of backward join, same lookahead-safety
    reasoning, as compute_spy_bearish_at_bars above (a candidate can be
    missing bars its ETF has, and must only ever see the ETF's already-
    closed bars). Returns NaN wherever the ETF's own return isn't defined
    yet (its own warmup, or no ETF bar at/before this timestamp) --
    simulate() treats NaN as "can't judge, don't block" (fails open),
    same as every other missing-data case in this filter.
    """
    etf = etf_bars[["timestamp", "close"]].copy()
    etf["etf_return_pct"] = etf["close"].pct_change(periods=SECTOR_RELATIVE_LOOKBACK_BARS) * 100
    aligned = pd.merge_asof(
        pd.DataFrame({"timestamp": bar_timestamps}),
        etf[["timestamp", "etf_return_pct"]],
        on="timestamp", direction="backward",
    )
    return aligned["etf_return_pct"].to_numpy()


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

def simulate(symbol: str, bars: pd.DataFrame, starting_equity: float,
             daily_trend_map: dict | None = None, apply_vwap_volume_filter: bool = False,
             spy_regime_bars: pd.DataFrame | None = None,
             sector_etf_bars: pd.DataFrame | None = None) -> list[dict]:
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

    spy_regime_bars: SPY's own add_indicators()-enriched bars (see
    fetch_spy_regime_bars), precomputed ONCE per run and passed in here
    for every symbol -- only consulted when USE_SPY_REGIME_GATE is on.
    None means either the gate is off or SPY's bars couldn't be fetched;
    either way this symbol's entries are ungated by it.

    sector_etf_bars: THIS symbol's own sector ETF's raw bars (see
    fetch_sector_etf_bars / get_sector_etf), only consulted when
    USE_SECTOR_RELATIVE_MEAN_REVERSION is on and only for mean_reversion
    entries specifically. None means the gate is off, the symbol's sector
    ETF is unknown, or its bars couldn't be fetched -- any of those fails
    open (mean_reversion entries proceed exactly as if the filter were
    off), same fail-open philosophy as every other sector lookup in this
    project.

    Returns a list of completed trades.
    """
    trades = []
    position = None  # None, or a dict describing the open simulated position
    enriched = add_indicators(bars)
    n = len(enriched)
    equity = starting_equity

    spy_bearish_now = None
    if USE_SPY_REGIME_GATE and spy_regime_bars is not None and not spy_regime_bars.empty:
        spy_bearish_now = compute_spy_bearish_at_bars(spy_regime_bars, enriched["timestamp"])

    # Sector-relative mean-reversion filter -- both series precomputed
    # ONCE up front (own-return via a plain pct_change, ETF-return via the
    # as-of join), same vectorize-before-the-loop pattern as spy_bearish_now
    # above, rather than recomputing either on every bar.
    sector_relative_blocks_now = None
    if (USE_SECTOR_RELATIVE_MEAN_REVERSION and sector_etf_bars is not None
            and not sector_etf_bars.empty):
        own_return_pct = (enriched["close"].pct_change(periods=SECTOR_RELATIVE_LOOKBACK_BARS) * 100).to_numpy()
        etf_return_pct = compute_sector_relative_return_at_bars(sector_etf_bars, enriched["timestamp"])
        underperformance_pct = etf_return_pct - own_return_pct
        # NaN on either side (either series still warming up) makes the
        # "<" comparison False by numpy's own NaN-propagation rule, which
        # already reads as "don't block" here -- the exact fail-open
        # behavior wanted, achieved for free rather than needing an
        # explicit isnan check (contrast spy_bearish_now's fillna(False),
        # a different mechanism reaching the same fail-open outcome).
        with np.errstate(invalid="ignore"):
            sector_relative_blocks_now = underperformance_pct < SECTOR_RELATIVE_MIN_UNDERPERFORMANCE_PCT

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
                elif (USE_BREAKOUT_INVALIDATION_EXIT and position["entry_reason"] == "breakout"
                      and position["invalidation_level"] is not None
                      and breakout_invalidated_at(enriched, i, position["invalidation_level"])):
                    # See strategy.breakout_invalidated_at's docstring for
                    # the real-evidence case: without this, a breakout
                    # position only ever exits via its bracket, whichever
                    # regime signal is active RIGHT NOW (unrelated to why
                    # it was opened), or the EOD flatten -- never because
                    # the breakout thesis itself specifically failed.
                    exit_price = current_bar["close"]
                    exit_reason = "breakout invalidated"
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
                    "high_vol_tercile": position["high_vol_tercile"],
                    "conviction_boosted": position["conviction_boosted"],
                })
                position = None
                continue  # don't also open a fresh position on the same bar we just exited

        # --- Look for a new entry ---
        if position is None:
            if minutes_until_close(current_bar["timestamp"]) <= STOP_NEW_ENTRIES_MINUTES_BEFORE_CLOSE:
                continue  # too close to the bell to open something new today
            if ENTRY_BLACKOUT_START_MINUTES <= current_bar["minutes_since_open"] < ENTRY_BLACKOUT_END_MINUTES:
                continue  # historically weak entry window (see strategy.py)
            if daily_trend_map is not None:
                # Multi-timeframe confirmation -- mirrors trading_bot.py's
                # daily_trend_confirms_entry exactly. None here (vs. an
                # empty dict) means "this symbol is exempt" (an S&P 500
                # name), so this branch only runs for symbols the filter
                # actually applies to.
                today = current_bar["timestamp"].tz_convert(MARKET_TZ).date()
                if not daily_trend_map.get(today, False):
                    continue
            if spy_bearish_now is not None and spy_bearish_now[i]:
                # Broad-market regime gate: SPY itself is trending down on
                # this same timeframe right now -- veto a NEW long here
                # regardless of what this symbol's own signal says.
                # Existing positions are untouched (this block only runs
                # when position is None).
                continue

            signal, reason_key, reason = decide_signal_at(enriched, i)
            if (signal == "BUY" and apply_vwap_volume_filter and reason_key == "vwap_reversion"
                    and not vwap_reversion_volume_confirms(enriched, i)):
                # Scanner-picks-only volume confirmation -- mirrors
                # trading_bot.py's check_symbol exactly. Caller only
                # passes apply_vwap_volume_filter=True for non-S&P-500
                # symbols (see USE_VWAP_VOLUME_CONFIRMATION above).
                continue
            if (signal == "BUY" and reason_key == "mean_reversion"
                    and sector_relative_blocks_now is not None and bool(sector_relative_blocks_now[i])):
                # Sector-relative mean-reversion filter -- mirrors
                # trading_bot.py's check_symbol exactly (only ever
                # consulted for a mean_reversion entry specifically). See
                # USE_SECTOR_RELATIVE_MEAN_REVERSION's comment there.
                continue
            if signal == "BUY":
                entry_price = current_bar["close"]
                stop_price, take_profit_price = compute_stop_and_target(entry_price, current_bar["atr"])
                # Same volatility guard the live bot applies -- a stop
                # inside the stock's own bar-to-bar noise isn't a stop.
                # Mirrored here so backtests can't flatter a setup the
                # live bot would refuse to take.
                if not stop_is_wider_than_noise(entry_price, current_bar["atr"], stop_price):
                    continue
                high_vol_tercile = bool(current_bar["high_vol_tercile"])
                # Never boosted under risk-based sizing -- mirrors
                # place_buy_order's identical reasoning in trading_bot.py;
                # set here so it's always defined for the position dict
                # below regardless of which sizing branch runs.
                conviction_boosted = False
                if USE_RISK_BASED_SIZING:
                    qty = compute_position_size(equity, entry_price, stop_price)
                else:
                    # See USE_VOLATILITY_SCALED_SIZING in strategy.py --
                    # only ever swaps in ANOTHER flat dollar figure for
                    # the high-vol tercile, never a fraction of equity or
                    # of TRADE_AMOUNT_USD, and strategy.py's own guard
                    # makes it impossible for this to exceed
                    # TRADE_AMOUNT_USD even via a bad env var.
                    trade_amount = TRADE_AMOUNT_USD
                    if USE_VOLATILITY_SCALED_SIZING and high_vol_tercile:
                        trade_amount = VOLATILITY_SCALED_REDUCED_USD
                    # elif, not a second independent if: see
                    # USE_CONVICTION_SIZING in strategy.py. EXPLICIT
                    # PRECEDENCE RULE, mirroring trading_bot.py's
                    # place_buy_order exactly: a trade that is BOTH in a
                    # high-conviction strategy AND in its own high-vol
                    # tercile gets the REDUCED (volatility) amount above,
                    # never the boosted one below -- safety (size down on
                    # real, measured noise) always wins over a return-
                    # chasing boost (size up on an unconfirmed strategy-
                    # level edge), every time, no exceptions.
                    elif USE_CONVICTION_SIZING and reason_key in HIGH_CONVICTION_STRATEGIES:
                        trade_amount = CONVICTION_BOOST_USD
                        conviction_boosted = True
                    qty = int(trade_amount // entry_price)
                if qty < 1:
                    continue  # not enough simulated capital/risk budget for even 1 share
                # Freeze the SAME level breakout_at() itself used to confirm
                # this entry (see USE_CLOSE_BEYOND_LEVEL_CONFIRMATION), so
                # breakout_invalidated_at() later checks against the level
                # that was actually true at entry, not a rolling window that
                # has since drifted. None for non-breakout entries -- the
                # exit check below never fires without a real level.
                invalidation_level = None
                if reason_key == "breakout":
                    level_col = ("breakout_recent_high_wick" if USE_CLOSE_BEYOND_LEVEL_CONFIRMATION
                                 else "breakout_recent_high")
                    level_value = current_bar[level_col]
                    if not pd.isna(level_value):
                        invalidation_level = float(level_value)
                position = {
                    "entry_time": current_bar["timestamp"],
                    "entry_price": entry_price,
                    "entry_reason": reason_key,
                    "qty": qty,
                    "stop_price": stop_price,
                    "take_profit_price": take_profit_price,
                    "high_vol_tercile": high_vol_tercile,
                    "conviction_boosted": conviction_boosted,
                    "invalidation_level": invalidation_level,
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
    if USE_VOLATILITY_SCALED_SIZING:
        print(f"Volatility-scaled sizing: ON -- a symbol's own high realized-vol tercile trades at "
              f"${VOLATILITY_SCALED_REDUCED_USD:.0f} instead of ${TRADE_AMOUNT_USD:.0f} "
              f"(only applies under flat sizing, i.e. USE_RISK_BASED_SIZING=false)\n")
    if USE_CONVICTION_SIZING:
        if HIGH_CONVICTION_STRATEGIES:
            print(f"Conviction-boosted sizing: ON -- {', '.join(sorted(HIGH_CONVICTION_STRATEGIES))} "
                  f"trade at ${CONVICTION_BOOST_USD:.0f} instead of ${TRADE_AMOUNT_USD:.0f} "
                  f"(only applies under flat sizing; volatility-based size-down always wins if a "
                  f"trade qualifies for both)\n")
        else:
            print("Conviction-boosted sizing: ON but HIGH_CONVICTION_STRATEGIES is empty -- "
                  "structurally a no-op right now (see strategy.HIGH_CONVICTION_STRATEGIES)\n")
    print("=" * 70)

    needs_sp500_list = USE_MULTI_TIMEFRAME_FILTER or USE_VWAP_VOLUME_CONFIRMATION
    sp500_symbols = fetch_sp500_symbols_once() if needs_sp500_list else set()
    if USE_MULTI_TIMEFRAME_FILTER:
        print(f"Multi-timeframe filter: ON, scanner-picks-only ({len(sp500_symbols)} S&P 500 "
              f"symbol(s) exempt)")
    if USE_VWAP_VOLUME_CONFIRMATION:
        print(f"vwap_reversion volume confirmation: ON, scanner-picks-only")
    if USE_MULTI_TIMEFRAME_FILTER or USE_VWAP_VOLUME_CONFIRMATION:
        print()

    spy_regime_bars = None
    if USE_SPY_REGIME_GATE:
        print(f"SPY regime gate: ON (blocks new entries in every symbol while "
              f"{SPY_REGIME_GATE_SYMBOL}'s own {BAR_MINUTES}-min ADX >= {ADX_TREND_THRESHOLD:.0f} "
              f"and its fast EMA < slow EMA)")
        spy_regime_bars = fetch_spy_regime_bars(args.days)
        print()

    sector_etf_bars_by_etf: dict[str, pd.DataFrame] = {}
    if USE_SECTOR_RELATIVE_MEAN_REVERSION:
        print(f"Sector-relative mean reversion filter: ON (mean_reversion entries require the "
              f"candidate to trail its own sector ETF by >= {SECTOR_RELATIVE_MIN_UNDERPERFORMANCE_PCT:.1f}pp "
              f"over {SECTOR_RELATIVE_LOOKBACK_BARS} bars; symbols with no known sector ETF are exempt)")
        sector_etf_bars_by_etf = fetch_sector_etf_bars(symbols, args.days)
        print()

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

        is_scanner_pick = symbol not in sp500_symbols

        daily_trend_map = None
        if USE_MULTI_TIMEFRAME_FILTER and is_scanner_pick:
            daily_trend_map = fetch_daily_trend_map(symbol, args.days)
            if not daily_trend_map:
                print(f"  (multi-timeframe filter: no daily data for {symbol} -- entries blocked all period)")

        apply_vwap_volume_filter = USE_VWAP_VOLUME_CONFIRMATION and is_scanner_pick

        symbol_sector_etf_bars = None
        if USE_SECTOR_RELATIVE_MEAN_REVERSION:
            etf_symbol = get_sector_etf(symbol)
            symbol_sector_etf_bars = sector_etf_bars_by_etf.get(etf_symbol) if etf_symbol else None

        trades = simulate(symbol, bars, starting_equity, daily_trend_map, apply_vwap_volume_filter,
                           spy_regime_bars, symbol_sector_etf_bars)
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
