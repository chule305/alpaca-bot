"""
Adaptive Intraday Trading Bot
------------------------------
Watches multiple stocks while the US market is open and trades them
automatically using several well-documented strategies (breakout, gap
continuation, Smash Day, Ross Hook, opening-range breakout, relative-
volume spike, VWAP reversion, trend-following, mean-reversion), chosen
automatically based on current conditions. See strategy.py for the
actual decision logic -- this file is the "runtime": connecting to
Alpaca, scanning for stocks, sizing and placing orders, and looping on
a schedule.

Every buy now includes an automatic stop-loss and take-profit (a
"bracket" order), so positions are protected even if the bot isn't
actively running at the exact moment the price moves. Position SIZE is
risk-based by default (see strategy.compute_position_size) rather than
a flat dollar amount, and a portfolio-level risk cap bounds aggregate
exposure across all open positions at once.

This script only trades Alpaca's PAPER TRADING account (fake money).
It runs continuously, but only takes action while the US stock market
is open -- it sleeps the rest of the time and wakes back up automatically.
"""

import os
import re
import time
import json
import argparse
import logging
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, GetOrdersRequest, TakeProfitRequest, StopLossRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus, OrderType
from alpaca.data.historical import StockHistoricalDataClient, ScreenerClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import StockBarsRequest, MostActivesRequest, MarketMoversRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import MostActivesBy, MarketType

from trade_recorder import record_trade, extract_context
from strategy import (
    add_indicators, decide_signal_at, compute_stop_and_target, compute_position_size,
    BAR_MINUTES, TRADE_AMOUNT_USD, FLATTEN_BEFORE_CLOSE, FLATTEN_MINUTES_BEFORE_CLOSE,
    STOP_NEW_ENTRIES_MINUTES_BEFORE_CLOSE, ENTRY_BLACKOUT_START_MINUTES, ENTRY_BLACKOUT_END_MINUTES,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, USE_ATR_STOPS, ATR_STOP_MULTIPLIER, ATR_TARGET_MULTIPLIER,
    USE_RISK_BASED_SIZING, RISK_PER_TRADE_PCT, MAX_POSITION_PCT_OF_EQUITY,
    FAST_MA, SLOW_MA, RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    ADX_PERIOD, ADX_TREND_THRESHOLD,
    USE_BREAKOUT, USE_SMASH_DAY_PATTERN, USE_GAP_PATTERN, USE_ROSS_HOOK, USE_ORB,
    USE_VWAP_REVERSION, USE_RVOL_SPIKE,
)

# ---------------------------------------------------------------------------
# SETUP / CONFIG
# ---------------------------------------------------------------------------

load_dotenv()

# .strip() defends against a trailing newline/whitespace sneaking into a
# pasted credential (e.g. via a GitHub Actions secret) -- that produces
# a cryptic "Invalid header value" error from the requests library with
# no obvious link back to "check your secret for a stray newline", so
# it's worth stripping unconditionally rather than relying on every
# credential being pasted perfectly clean everywhere this runs.
API_KEY = (os.getenv("ALPACA_API_KEY") or "").strip()
SECRET_KEY = (os.getenv("ALPACA_SECRET_KEY") or "").strip()

# Widened from the original TSLA/NVDA/COIN after the first backtest showed
# those three are behaviorally correlated (high-beta growth/risk-sentiment
# names). AMD and PLTR added real diversity and performed well in a live-
# validated run; MSTR was tried too but dropped -- worst symbol (-18.6%)
# with a 42% max drawdown, by far the riskiest thing tested. See CLAUDE.md.
# This is only the FALLBACK list used when USE_SCANNER=false or the scan
# fails; live trading normally uses the scanner's own picks instead.
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "TSLA,NVDA,COIN,AMD,PLTR").split(",") if s.strip()]
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", 5))

USE_SCANNER = os.getenv("USE_SCANNER", "true").strip().lower() in ("1", "true", "yes")
# Widened/faster defaults from the original 5 stocks refreshed once a day --
# batched bar-fetching (see get_recent_bars_batch) is what makes checking a
# much bigger watchlist every cycle cheap on API calls, so the scanner can
# now afford to look harder and more often.
SCANNER_WATCHLIST_SIZE = int(os.getenv("SCANNER_WATCHLIST_SIZE", 12))
# Floor on the watchlist size: if a scan legitimately succeeds but finds
# only a couple of qualifying names (common -- Alpaca caps the candidate
# pool at 50 gainers + 50 losers, and most extreme movers are either
# penny stocks or leveraged ETFs that get filtered out), top the list up
# with the known-liquid SYMBOLS names rather than letting the bot go a
# whole day with almost nothing to look at. Set to 0 to disable.
SCANNER_MIN_WATCHLIST_SIZE = int(os.getenv("SCANNER_MIN_WATCHLIST_SIZE", 5))
# Alpaca's screener endpoints hard-cap the "top" parameter at 50 server-side
# ("invalid top: should not be larger than 50") -- found live on 2026-07-23
# when this was set to 100. Clamped here (not just defaulted lower) so a
# misconfigured env var can't reintroduce the same failure.
SCANNER_CANDIDATE_POOL = min(int(os.getenv("SCANNER_CANDIDATE_POOL", 50)), 50)
SCANNER_MIN_PRICE = float(os.getenv("SCANNER_MIN_PRICE", 10.0))
# Minimum average DOLLAR volume (price x shares) for a candidate to count
# as liquid enough to trade. This replaced the original liquidity gate,
# which required a candidate to also appear in Alpaca's "most actives by
# volume" top 50 -- that gate was structurally broken and silently
# returned an EMPTY watchlist every scan for two days (2026-07-22 to
# 2026-07-24), so the bot ran on the fallback SYMBOLS list the whole
# time. Reason: "most active by SHARE volume" is dominated by cheap
# stocks (a dollar buys more shares), so the only names appearing in both
# the biggest-movers list AND the most-actives list were sub-$10 penny
# stocks -- exactly what SCANNER_MIN_PRICE then rejected. The two filters
# were mutually exclusive by construction. Measured live: 100 movers ->
# 5 in both lists -> 0 survived the price filter. Dollar volume is the
# correct liquidity measure and doesn't fight the price filter.
SCANNER_MIN_DOLLAR_VOLUME = float(os.getenv("SCANNER_MIN_DOLLAR_VOLUME", 10_000_000))
SCANNER_REFRESH_HOURS = float(os.getenv("SCANNER_REFRESH_HOURS", 0.5))
EXCLUDE_LEVERAGED_ETFS = os.getenv("EXCLUDE_LEVERAGED_ETFS", "true").strip().lower() in ("1", "true", "yes")
# The scanner ranks by size of today's move, which means it structurally
# favors stocks that have ALREADY moved a lot -- it's a momentum-chasing
# scanner by construction, not an early-catch one (there's no cheap way
# to catch a move before it starts on Alpaca's free data). This doesn't
# fix that, but it does avoid the worst version of it: a candidate
# already up/down more than this % today is more likely near-exhausted
# than just getting started, so it's excluded rather than ranked #1.
SCANNER_MAX_EXTENSION_PCT = float(os.getenv("SCANNER_MAX_EXTENSION_PCT", 50.0))

# Lightweight news-catalyst filter (presence/frequency only, no AI/NLP
# sentiment scoring -- that would add per-symbol latency and cost inside
# a scan loop). A mover with zero recent news is more likely thin-volume
# noise than a mover with actual news behind it.
USE_NEWS_FILTER = os.getenv("USE_NEWS_FILTER", "true").strip().lower() in ("1", "true", "yes")
NEWS_LOOKBACK_HOURS = float(os.getenv("NEWS_LOOKBACK_HOURS", 24))
MIN_NEWS_ITEMS = int(os.getenv("MIN_NEWS_ITEMS", 1))

# Loss-reduction: cap how many positions can be open at once, and a daily
# circuit breaker that pauses NEW entries (existing positions still get
# watched/exited normally) once the account is down more than this % on
# the day. MAX_PORTFOLIO_RISK_PCT is the more important of the three --
# it caps the sum of $-at-risk across every open position at once, so a
# handful of correlated positions (the scanner tends to find similar
# high-beta names) can't quietly stack up more aggregate risk than
# intended just because each one individually looked fine.
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", 5))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", 3))
MAX_PORTFOLIO_RISK_PCT = float(os.getenv("MAX_PORTFOLIO_RISK_PCT", 5.0))

# Best-effort list of common leveraged/inverse ETFs. These are structurally
# built to move 2-3x their underlying index, so they show up constantly in
# "biggest movers" scans without anything unusual actually happening.
# Not exhaustive -- covers the most common ones from major issuers.
LEVERAGED_ETF_DENYLIST = {
    "SOXL", "SOXS", "TQQQ", "SQQQ", "SPXL", "SPXS", "UPRO", "SPXU",
    "TNA", "TZA", "FAS", "FAZ", "LABU", "LABD", "YINN", "YANG",
    "NUGT", "DUST", "JNUG", "JDST", "GUSH", "DRIP", "UVXY", "SVXY",
    "BOIL", "KOLD", "TMF", "TMV", "UDOW", "SDOW", "URTY", "SRTY",
}

LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TIMEZONE", "Europe/Zagreb"))
WATCHLIST_STATE_FILE = "watchlist_state.json"
DAILY_RISK_STATE_FILE = "daily_risk_state.json"

# Mutable run state. Initialized here (not only in the __main__ block) so
# importing this module -- which the tests do -- gives a consistent
# starting state instead of AttributeError on first access. The __main__
# block still re-initializes these and then loads the saved state files
# over the top, so live behavior is unchanged.
active_watchlist: list[str] = list(SYMBOLS)
last_scan_time: datetime | None = None
last_flatten_date: date | None = None
current_trading_day: date | None = None
day_start_equity: float | None = None
daily_loss_breaker_tripped = False

if not API_KEY or not SECRET_KEY or "your_paper" in API_KEY:
    raise SystemExit(
        "ERROR: Please fill in your Alpaca PAPER API keys in the .env file "
        "before running this bot. See README.md for instructions."
    )

if not SYMBOLS:
    raise SystemExit("ERROR: SYMBOLS in your .env file is empty. Add at least one stock ticker "
                      "(used as a fallback if the scanner is off or fails).")

# Logs are written per TRADING DAY into logs/, not to one growing file.
#
# The reason is GitHub Actions: every scheduled run gets a fresh checkout
# and the runner is destroyed afterwards, so anything not committed back
# is gone forever. That's why diagnosing the 2026-07-24 failures meant
# reproducing them locally instead of just reading what the bot had
# logged -- there was nothing left to read. A dated file per day means
# every run that day appends to the same file, the workflow commits it
# back, and the whole session's terminal output stays available
# afterwards for exactly that kind of post-mortem.
LOG_DIR = os.getenv("LOG_DIR", "logs").strip()
LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", 30))
MARKET_TZ_FOR_LOGS = ZoneInfo("America/New_York")


def _todays_log_path() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    day = datetime.now(MARKET_TZ_FOR_LOGS).strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"{day}.log")


def _prune_old_logs() -> None:
    """Keeps the repo from growing without bound, since these get committed."""
    try:
        cutoff = datetime.now(MARKET_TZ_FOR_LOGS).date() - timedelta(days=LOG_RETENTION_DAYS)
        for name in os.listdir(LOG_DIR):
            if not name.endswith(".log"):
                continue
            try:
                file_day = datetime.strptime(name[:-4], "%Y-%m-%d").date()
            except ValueError:
                continue
            if file_day < cutoff:
                os.remove(os.path.join(LOG_DIR, name))
    except Exception:
        pass  # log housekeeping must never stop the bot from trading


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        # Append, so every run in the same trading day adds to one file
        # rather than overwriting the earlier runs' output.
        logging.FileHandler(_todays_log_path(), mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger()
_prune_old_logs()

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
screener_client = ScreenerClient(API_KEY, SECRET_KEY)
news_client = NewsClient(API_KEY, SECRET_KEY)


# ---------------------------------------------------------------------------
# TIME HELPERS
# ---------------------------------------------------------------------------

def to_local(dt: datetime) -> datetime:
    """Converts any timestamp to your local timezone for readable logs."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(LOCAL_TZ)


def fmt(dt: datetime) -> str:
    return to_local(dt).strftime("%Y-%m-%d %H:%M:%S %Z")


# ---------------------------------------------------------------------------
# PRICE DATA
# ---------------------------------------------------------------------------

def get_recent_bars_batch(symbols: list[str], lookback_days: int = 10) -> dict[str, pd.DataFrame]:
    """
    Fetches recent intraday price bars for ALL given symbols in a SINGLE
    API call instead of one call per symbol. This is what keeps checking
    a much bigger watchlist cheap on API rate limits regardless of how
    many symbols the scanner is tracking.
    lookback_days is in CALENDAR days, so 10 calendar days comfortably
    covers enough actual trading sessions to fill our indicator windows.
    Returns a dict of symbol -> DataFrame (missing/empty for any symbol
    with no data returned).
    """
    if not symbols:
        return {}
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
        start=datetime.now(timezone.utc) - timedelta(days=lookback_days),
    )
    bars = data_client.get_stock_bars(request).df
    if bars.empty:
        return {}
    bars = bars.reset_index()
    return {
        symbol: group.drop(columns="symbol").reset_index(drop=True)
        for symbol, group in bars.groupby("symbol")
    }


# ---------------------------------------------------------------------------
# NEWS -- lightweight catalyst check (presence/frequency, not sentiment)
# ---------------------------------------------------------------------------

def fetch_news_counts(symbols: list[str]) -> dict[str, int]:
    """
    Counts recent news articles per symbol over NEWS_LOOKBACK_HOURS, in
    one batched request. Returns an empty dict (meaning "no news data
    available") on any failure -- callers should treat that as
    fail-open, not as "definitely zero news."
    """
    if not symbols:
        return {}
    try:
        request = NewsRequest(
            symbols=",".join(symbols),
            start=datetime.now(timezone.utc) - timedelta(hours=NEWS_LOOKBACK_HOURS),
            limit=50,
        )
        news_set = news_client.get_news(request)
    except Exception as e:
        log.warning(f"SCANNER: could not fetch news data: {e}")
        return {}

    symbol_set = set(symbols)
    counts: dict[str, int] = {}
    for article in news_set.data.get("news", []):
        for sym in article.symbols:
            if sym in symbol_set:
                counts[sym] = counts.get(sym, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# WATCHLIST SCANNER -- picks today's volatile-stock candidates automatically
# ---------------------------------------------------------------------------

# Matches the way leveraged/inverse ETFs name themselves: a multiplier
# token ("2X", "3x", "1.5X"), or an explicit leverage/direction word.
# Checked against Alpaca's own asset name, NOT a ticker list, because a
# ticker denylist is unmaintainable here -- single-stock leveraged ETFs
# launch constantly under new symbols. Found live on 2026-07-24: 14 of
# the scanner's 21 surviving candidates were 2x single-stock ETFs
# ("Tradr 2X Short NBIS Daily ETF", "GraniteShares 2x Long NBIS Daily
# ETF", ...), none of which the ticker denylist knew about. These are
# especially dangerous for this bot because its stop-loss/take-profit
# percentages assume ordinary single-stock volatility, and a 2x/3x
# product blows through them twice as fast.
LEVERAGED_ETF_NAME_PATTERN = re.compile(
    r"(\b\d+(\.\d+)?\s*X\b|\bLEVERAGE|\bINVERSE\b|\bULTRA|\bBULL\b|\bBEAR\b)",
    re.IGNORECASE,
)

# Asset names never change, so cache them -- this keeps repeated scans
# inside one long-running job from re-fetching the same metadata.
_asset_name_cache: dict[str, str] = {}


def is_leveraged_etf(symbol: str) -> bool | None:
    """
    True if Alpaca's asset name for this symbol looks like a leveraged or
    inverse ETF, False if it clearly doesn't, None if it couldn't be
    determined (lookup failed). Callers treat None as "skip it" -- see
    scan_for_volatile_stocks for why that's the safe direction.
    """
    name = _asset_name_cache.get(symbol)
    if name is None:
        try:
            name = trading_client.get_asset(symbol).name or ""
        except Exception:
            return None
        _asset_name_cache[symbol] = name
    return bool(LEVERAGED_ETF_NAME_PATTERN.search(name))


def scan_for_volatile_stocks() -> list[str]:
    """
    Builds a watchlist automatically using Alpaca's screener data instead
    of a fixed symbol list:
      1. Pulls today's biggest % gainers and losers (candidates for
         volatility -- big moves happening right now).
      2. Keeps only candidates above SCANNER_MIN_PRICE that haven't
         already moved more than SCANNER_MAX_EXTENSION_PCT today (likely
         near-exhausted rather than just getting started).
      3. Optionally excludes known leveraged/inverse ETFs (see denylist
         above) -- these move a lot structurally, not because anything
         unusual is actually happening.
      4. Liquidity check: requires average DOLLAR volume (price x shares)
         of at least SCANNER_MIN_DOLLAR_VOLUME, measured from real recent
         bars. Alpaca's "most actives" list is used only as a free fast
         path (anything in it is already known liquid) and as a fallback
         if the bar fetch fails -- deliberately NOT as a hard gate; see
         SCANNER_MIN_DOLLAR_VOLUME's comment for the outage that caused.
      5. If USE_NEWS_FILTER is on: drops candidates with fewer than
         MIN_NEWS_ITEMS recent articles (a mover with no news behind it
         is more likely noise/thin-volume than a real catalyst) -- unless
         that would eliminate every candidate this cycle, in which case
         the filter is skipped for now rather than returning nothing.
      6. Ranks the survivors by size of move and returns the top N.

    Returns an empty list if the scan fails or nothing qualifies --
    callers should fall back to the manual SYMBOLS list in that case.
    """
    try:
        movers = screener_client.get_market_movers(
            MarketMoversRequest(top=SCANNER_CANDIDATE_POOL, market_type=MarketType.STOCKS)
        )
    except Exception as e:
        log.error(f"SCANNER: could not fetch screener data: {e}")
        return []

    # Best-effort only -- a failure here must not stop the scan, since
    # this is now just a fast path, not a requirement.
    known_liquid: set[str] = set()
    try:
        actives = screener_client.get_most_actives(
            MostActivesRequest(by=MostActivesBy.VOLUME, top=SCANNER_CANDIDATE_POOL)
        )
        known_liquid = {a.symbol for a in actives.most_actives}
    except Exception as e:
        log.warning(f"SCANNER: could not fetch most-actives list ({e}) -- "
                     f"falling back to dollar-volume checks alone.")

    candidates = list(movers.gainers) + list(movers.losers)

    # Cheap filters first, so the bar fetch below only covers survivors.
    prefiltered = []
    for m in candidates:
        if m.price < SCANNER_MIN_PRICE:
            continue
        if abs(m.percent_change) > SCANNER_MAX_EXTENSION_PCT:
            continue
        if EXCLUDE_LEVERAGED_ETFS and m.symbol in LEVERAGED_ETF_DENYLIST:
            continue
        prefiltered.append(m)

    if not prefiltered:
        log.warning("SCANNER: no candidates survived the price/extension filters.")
        return []

    # Structural leveraged/inverse ETF check against Alpaca's asset names.
    # Deliberately fail-CLOSED (an unverifiable symbol is skipped): the
    # cost of skipping one candidate is ~zero since others are always
    # available, while the cost of accidentally trading a 2x product with
    # stops tuned for ordinary stocks is real. If the asset endpoint were
    # down entirely this returns nothing and the caller falls back to the
    # known-liquid SYMBOLS list, which is a safe degradation.
    if EXCLUDE_LEVERAGED_ETFS:
        kept = []
        for m in prefiltered:
            verdict = is_leveraged_etf(m.symbol)
            if verdict is None:
                log.info(f"SCANNER: skipping {m.symbol} -- could not verify what kind of asset it is.")
                continue
            if verdict:
                continue
            kept.append(m)
        if len(kept) < len(prefiltered):
            log.info(f"SCANNER: excluded {len(prefiltered) - len(kept)} leveraged/inverse ETF(s) "
                      f"from {len(prefiltered)} candidate(s).")
        prefiltered = kept
        if not prefiltered:
            log.warning("SCANNER: every candidate this cycle was a leveraged/inverse ETF.")
            return []

    # Dollar-volume liquidity check on whatever isn't already known liquid.
    needs_check = [m.symbol for m in prefiltered if m.symbol not in known_liquid]
    liquid_enough = set(known_liquid)
    if needs_check:
        try:
            bars = get_recent_bars_batch(needs_check, lookback_days=5)
            for symbol, df in bars.items():
                if df is None or df.empty:
                    continue
                avg_dollar_volume = float((df["close"] * df["volume"]).mean())
                # Bars are BAR_MINUTES long, so scale one bar's average up
                # to a full 6.5-hour session for a comparable daily figure.
                bars_per_session = (6.5 * 60) / BAR_MINUTES
                if avg_dollar_volume * bars_per_session >= SCANNER_MIN_DOLLAR_VOLUME:
                    liquid_enough.add(symbol)
        except Exception as e:
            # Degrade to the old behavior rather than returning nothing.
            log.warning(f"SCANNER: dollar-volume check failed ({e}) -- falling back to "
                         f"the most-actives list alone for liquidity this cycle.")

    qualified = [m for m in prefiltered if m.symbol in liquid_enough]
    if not qualified:
        log.warning(f"SCANNER: {len(prefiltered)} candidate(s) passed price/extension filters but "
                     f"none met the ${SCANNER_MIN_DOLLAR_VOLUME:,.0f} average dollar-volume bar.")
        return []

    news_counts: dict[str, int] = {}
    if USE_NEWS_FILTER and qualified:
        news_counts = fetch_news_counts([m.symbol for m in qualified])
        with_news = [m for m in qualified if news_counts.get(m.symbol, 0) >= MIN_NEWS_ITEMS]
        if with_news:
            qualified = with_news
        else:
            log.info(f"SCANNER: news filter (min {MIN_NEWS_ITEMS} article(s) in "
                      f"{NEWS_LOOKBACK_HOURS:.0f}h) would eliminate every candidate this cycle -- "
                      f"skipping it for now.")

    qualified.sort(key=lambda m: abs(m.percent_change), reverse=True)

    picked = []
    for m in qualified:
        if m.symbol not in picked:
            news_note = f", {news_counts.get(m.symbol, 0)} recent news item(s)" if USE_NEWS_FILTER else ""
            log.info(f"SCANNER: candidate {m.symbol} -- {m.percent_change:+.1f}% today, ${m.price:.2f}, "
                      f"liquid{news_note}")
            picked.append(m.symbol)
        if len(picked) >= SCANNER_WATCHLIST_SIZE:
            break

    return picked


def save_watchlist_state() -> None:
    """Persists the current watchlist so a restart doesn't force an immediate re-scan."""
    try:
        with open(WATCHLIST_STATE_FILE, "w") as f:
            json.dump({
                "active_watchlist": active_watchlist,
                "last_scan_time": last_scan_time.isoformat() if last_scan_time else None,
            }, f)
    except Exception as e:
        log.warning(f"Could not save watchlist state: {e}")


def load_watchlist_state() -> None:
    """
    Loads a previously saved watchlist, if one exists, so restarting the
    bot doesn't immediately trigger a fresh scan (which could produce a
    different watchlist than the one that was already active).
    """
    global active_watchlist, last_scan_time
    try:
        with open(WATCHLIST_STATE_FILE, "r") as f:
            data = json.load(f)
        saved_watchlist = data.get("active_watchlist")
        saved_scan_time = data.get("last_scan_time")
        if saved_watchlist:
            active_watchlist = saved_watchlist
        if saved_scan_time:
            last_scan_time = datetime.fromisoformat(saved_scan_time)
        log.info(f"Loaded saved watchlist from previous run: {', '.join(active_watchlist)}")
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"Could not load saved watchlist state ({e}), starting fresh.")


def refresh_watchlist_if_needed(clock_timestamp: datetime) -> None:
    """
    Re-runs the scanner if enough time has passed (SCANNER_REFRESH_HOURS).
    Updates the module-level active_watchlist and saves it to disk so a
    restart won't immediately trigger another scan. Falls back to the
    manual SYMBOLS list if the scan fails or finds nothing -- and in
    that case, does NOT mark the scan as done, so it retries again next
    check instead of waiting a full cycle.
    """
    global active_watchlist, last_scan_time

    if not USE_SCANNER:
        return

    needs_refresh = (
        last_scan_time is None
        or (clock_timestamp - last_scan_time).total_seconds() >= SCANNER_REFRESH_HOURS * 3600
    )
    if not needs_refresh:
        return

    log.info("SCANNER: refreshing watchlist...")
    picked = scan_for_volatile_stocks()

    if picked:
        if len(picked) < SCANNER_MIN_WATCHLIST_SIZE:
            topped_up = list(picked)
            for s in SYMBOLS:
                if len(topped_up) >= SCANNER_MIN_WATCHLIST_SIZE:
                    break
                if s not in topped_up:
                    topped_up.append(s)
            log.info(f"SCANNER: only {len(picked)} name(s) qualified -- topping the watchlist up to "
                      f"{len(topped_up)} with fallback symbols so there's still enough to work with.")
            picked = topped_up
        active_watchlist = picked
        last_scan_time = clock_timestamp
        save_watchlist_state()
        log.info(f"SCANNER: new watchlist -> {', '.join(active_watchlist)}")
    else:
        active_watchlist = list(SYMBOLS)
        log.warning(f"SCANNER: scan failed or found no qualifying stocks -- using fallback list "
                     f"({', '.join(SYMBOLS)}) for now, will retry next check.")


# ---------------------------------------------------------------------------
# DAILY RISK STATE -- persisted so a crash-and-restart mid-day doesn't
# silently un-trip the circuit breaker or lose track of the day's
# starting equity.
# ---------------------------------------------------------------------------

def save_daily_risk_state() -> None:
    try:
        with open(DAILY_RISK_STATE_FILE, "w") as f:
            json.dump({
                "day_start_equity": day_start_equity,
                "current_trading_day": current_trading_day.isoformat() if current_trading_day else None,
                "daily_loss_breaker_tripped": daily_loss_breaker_tripped,
            }, f)
    except Exception as e:
        log.warning(f"Could not save daily risk state: {e}")


def load_daily_risk_state() -> None:
    """
    Restores the circuit breaker's state after a restart. If the saved
    state is from a previous day, the main loop's normal day-rollover
    check will detect that (today's real date won't match the saved
    one) and reinitialize everything fresh -- no special-casing needed
    here for stale state.
    """
    global day_start_equity, current_trading_day, daily_loss_breaker_tripped
    try:
        with open(DAILY_RISK_STATE_FILE, "r") as f:
            data = json.load(f)
        saved_day = data.get("current_trading_day")
        if saved_day:
            current_trading_day = date.fromisoformat(saved_day)
            day_start_equity = data.get("day_start_equity")
            daily_loss_breaker_tripped = bool(data.get("daily_loss_breaker_tripped", False))
            log.info(f"Restored daily risk state from previous run: day {current_trading_day}, "
                      f"breaker tripped: {daily_loss_breaker_tripped}")
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"Could not load saved daily risk state ({e}), starting fresh.")


# ---------------------------------------------------------------------------
# ORDERS / POSITIONS
# ---------------------------------------------------------------------------

def get_all_open_positions() -> dict[str, dict]:
    """
    Every open position in one API call, keyed by symbol -- {"qty":
    float, "avg_entry_price": float}. Reused for the open-symbol set,
    per-symbol qty lookups, and (with open stop-loss orders) the
    portfolio risk cap, instead of a separate get_open_position() call
    per symbol every cycle.
    """
    try:
        positions = trading_client.get_all_positions()
        return {p.symbol: {"qty": float(p.qty), "avg_entry_price": float(p.avg_entry_price)} for p in positions}
    except Exception as e:
        log.error(f"Could not fetch open positions: {e}")
        return {}


def get_current_portfolio_risk_usd(open_positions: dict[str, dict]) -> float:
    """
    Sums $-at-risk across all open positions: for each one, (entry price
    - current stop-loss price) * qty, read from the still-open bracket
    stop-loss orders. This bounds AGGREGATE exposure directly -- even a
    handful of highly-correlated positions (the scanner tends to find
    similar high-beta names) can't blow past MAX_PORTFOLIO_RISK_PCT,
    without needing to compute correlation between them. Positions with
    no matching open stop order are skipped (shouldn't normally happen,
    since every buy submits a bracket order with one).
    """
    if not open_positions:
        return 0.0
    try:
        open_orders = trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=list(open_positions.keys()))
        )
    except Exception as e:
        log.warning(f"Could not fetch open orders for portfolio risk check: {e}")
        return 0.0

    stop_price_by_symbol = {}
    for o in open_orders:
        if o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and o.stop_price is not None:
            stop_price_by_symbol[o.symbol] = float(o.stop_price)

    total_risk = 0.0
    for symbol, pos in open_positions.items():
        stop_price = stop_price_by_symbol.get(symbol)
        if stop_price is None:
            continue
        risk_per_share = pos["avg_entry_price"] - stop_price
        total_risk += max(risk_per_share, 0.0) * pos["qty"]
    return total_risk


def would_exceed_portfolio_risk_cap(equity: float | None, portfolio_risk_estimate: float) -> bool:
    """
    Whether opening one more risk-based-sized position would push total
    portfolio risk over MAX_PORTFOLIO_RISK_PCT of equity. Only
    meaningful when risk-based sizing is on and equity is known -- with
    flat-dollar sizing there's no clean way to estimate a new trade's
    risk before actually placing it, so this doesn't block in that case.
    """
    if not USE_RISK_BASED_SIZING or equity is None or equity <= 0:
        return False
    projected_new_risk = equity * RISK_PER_TRADE_PCT / 100
    cap = equity * MAX_PORTFOLIO_RISK_PCT / 100
    return (portfolio_risk_estimate + projected_new_risk) > cap


def place_buy_order(symbol: str, last_price: float, atr_value: float | None, equity: float | None,
                     reason_key: str = "unknown"):
    """
    Buys a stop-loss/take-profit-protected position ("bracket" order).

    Position SIZE comes from strategy.compute_position_size when
    USE_RISK_BASED_SIZING is on and equity is known: sized off how much
    you're willing to LOSE if the stop is hit (RISK_PER_TRADE_PCT of
    equity), not a flat dollar amount -- a volatile stock with a wide
    stop gets fewer shares than a calm one with a tight stop for the
    same dollar risk. Falls back to flat TRADE_AMOUNT_USD sizing if
    risk-based sizing is off or equity couldn't be fetched this cycle.

    Bracket orders on Alpaca don't reliably support fractional
    quantities (unlike plain market orders), regardless of whether the
    underlying stock itself is normally fractionable -- so this always
    buys a whole number of shares.

    Stop/target levels come from strategy.compute_stop_and_target, the
    same helper the backtester uses, so live and backtest can't drift
    apart on this either.

    The order's client_order_id is tagged with reason_key (which
    strategy triggered it) -- Alpaca has no concept of "why" an order
    was placed, and trading_log.txt doesn't persist across GitHub
    Actions' fresh-checkout-per-run model, so this is the only place
    that information can survive to be read back later (e.g. by
    daily_summary.py) without adding a whole separate state file.

    Returns (order, notional_usd) -- order is None if the trade was
    skipped (computed size was 0 shares), in which case notional_usd is 0.
    """
    stop_price, take_profit_price = compute_stop_and_target(last_price, atr_value)
    stop_price = round(stop_price, 2)
    take_profit_price = round(take_profit_price, 2)

    if USE_RISK_BASED_SIZING and equity is not None and equity > 0:
        qty = compute_position_size(equity, last_price, stop_price)
        sizing_style = f"risk-based, {RISK_PER_TRADE_PCT:.1f}% of ${equity:,.0f}"
    else:
        qty = int(TRADE_AMOUNT_USD // last_price)
        sizing_style = "flat $"

    if qty < 1:
        log.warning(f"[{symbol}] Computed position size is 0 shares at ${last_price:.2f} "
                     f"({sizing_style} sizing) -- skipping this trade.")
        return None, 0.0

    stop_style = "ATR-based" if (USE_ATR_STOPS and atr_value is not None and not pd.isna(atr_value)) else "fixed %"

    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_profit_price),
        stop_loss=StopLossRequest(stop_price=stop_price),
        client_order_id=f"{reason_key}-{int(datetime.now(timezone.utc).timestamp())}",
    )
    notional = qty * last_price
    log.info(f"[{symbol}] Buying {qty} share(s) (~${notional:.0f}, {sizing_style} sizing) | "
              f"stop-loss ${stop_price:.2f} | take-profit ${take_profit_price:.2f} ({stop_style})")
    order = trading_client.submit_order(order_request)
    place_buy_order.last_details = {
        "qty": qty,
        "price": last_price,
        "notional": notional,
        "stop_loss": stop_price,
        "take_profit": take_profit_price,
        "sizing_style": sizing_style,
        "equity": equity,
    }
    return order, notional


# Populated by place_buy_order so check_symbol can record the sizing and
# bracket levels it chose without place_buy_order needing to know about
# indicator context (which it has no access to) -- keeps the recording
# concern out of the order-placement signature.
place_buy_order.last_details = {}


def place_sell_order(symbol: str):
    """
    Closes the entire open position in this symbol. First cancels any
    outstanding bracket legs (stop-loss/take-profit orders) still open
    for this symbol -- Alpaca won't let a new sell through while shares
    are already reserved by those resting orders.
    """
    try:
        open_orders = trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        )
        for o in open_orders:
            trading_client.cancel_order_by_id(o.id)
    except Exception as e:
        log.warning(f"[{symbol}] Could not cancel outstanding bracket orders before selling: {e}")

    return trading_client.close_position(symbol)


# ---------------------------------------------------------------------------
# PER-SYMBOL CHECK
# ---------------------------------------------------------------------------

def check_symbol(symbol: str, df: pd.DataFrame, entries_paused_reason: str | None, at_position_cap: bool,
                  current_qty: float, equity: float | None, portfolio_risk_estimate: float) -> float:
    """
    Checks one symbol and acts on its signal. Returns the notional $
    amount of a newly opened position (0.0 if none), so the caller can
    track MAX_CONCURRENT_POSITIONS, the portfolio risk cap, and the
    running equity estimate live across a single cycle without an extra
    API call per symbol.
    """
    if df is None or df.empty:
        log.warning(f"[{symbol}] No price data returned, skipping this check.")
        return 0.0

    enriched = add_indicators(df)
    i = len(enriched) - 1
    signal, reason_key, reason = decide_signal_at(enriched, i)
    last_price = enriched["close"].iat[i]
    atr_value = enriched["atr"].iat[i]
    minutes_since_open_now = enriched["minutes_since_open"].iat[i]
    in_lunch_blackout = ENTRY_BLACKOUT_START_MINUTES <= minutes_since_open_now < ENTRY_BLACKOUT_END_MINUTES

    log.info(f"[{symbol}] {reason} | Signal: {signal} | Shares held: {current_qty} | Last price: ${last_price:.2f}")

    notional_opened = 0.0
    try:
        if signal == "BUY" and current_qty == 0:
            if entries_paused_reason:
                log.info(f"[{symbol}] ACTION: No trade ({entries_paused_reason}).")
            elif in_lunch_blackout:
                log.info(f"[{symbol}] ACTION: No trade (within the historically weak "
                          f"{ENTRY_BLACKOUT_START_MINUTES}-{ENTRY_BLACKOUT_END_MINUTES} min-since-open entry window).")
            elif at_position_cap:
                log.info(f"[{symbol}] ACTION: No trade (at MAX_CONCURRENT_POSITIONS={MAX_CONCURRENT_POSITIONS} cap).")
            elif would_exceed_portfolio_risk_cap(equity, portfolio_risk_estimate):
                log.info(f"[{symbol}] ACTION: No trade (would exceed MAX_PORTFOLIO_RISK_PCT="
                          f"{MAX_PORTFOLIO_RISK_PCT:.1f}% aggregate risk cap).")
            else:
                order, notional = place_buy_order(symbol, last_price, atr_value, equity, reason_key)
                if order is not None:
                    log.info(f"[{symbol}] ACTION: BUY (order id {order.id}, strategy: {reason_key})")
                    notional_opened = notional
                    record_trade(
                        "BUY", symbol,
                        trading_day_et=_trading_day_et(),
                        strategy=reason_key, reason=reason,
                        order_id=str(order.id), client_order_id=order.client_order_id,
                        context=extract_context(enriched, i),
                        details=place_buy_order.last_details,
                    )
        elif signal == "SELL" and current_qty > 0:
            order = place_sell_order(symbol)
            log.info(f"[{symbol}] ACTION: SELL - closing position (order id {order.id})")
            record_trade(
                "SELL", symbol,
                trading_day_et=_trading_day_et(),
                strategy=reason_key, reason=reason,
                qty=current_qty, price=last_price, notional=current_qty * last_price,
                equity=equity, order_id=str(order.id),
                context=extract_context(enriched, i),
            )
        else:
            log.info(f"[{symbol}] ACTION: No trade.")
    except Exception as e:
        error_text = str(e)
        if "trading halt" in error_text.lower():
            log.warning(f"[{symbol}] Trading is currently halted (common during extreme moves) -- "
                         f"will automatically try again next check.")
        else:
            log.error(f"[{symbol}] Order failed: {e}")

    return notional_opened


# ---------------------------------------------------------------------------
# DAILY LOSS CIRCUIT BREAKER
# ---------------------------------------------------------------------------

def get_account_equity() -> float | None:
    try:
        account = trading_client.get_account()
        return float(account.equity)
    except Exception as e:
        log.warning(f"Could not fetch account equity: {e}")
        return None


# ---------------------------------------------------------------------------
# MAIN LOOP -- runs continuously, only trades while the market is open
# ---------------------------------------------------------------------------

def seconds_until(target: datetime, now: datetime) -> float:
    return max((target - now).total_seconds(), 0)


def _trading_day_et() -> str:
    """The ET calendar day, so trades group the same way the logs do."""
    return datetime.now(MARKET_TZ_FOR_LOGS).strftime("%Y-%m-%d")


def flatten_all_positions() -> None:
    """Closes every open position and cancels any pending orders."""
    # Read positions BEFORE closing them -- afterwards there's nothing
    # left to describe, and an end-of-day exit is exactly the kind of
    # trade worth being able to analyze separately from a signal-driven
    # one (it's an exit the strategy didn't choose).
    try:
        closing = get_all_open_positions()
    except Exception:
        closing = {}

    try:
        trading_client.close_all_positions(cancel_orders=True)
        log.info("EOD FLATTEN: all positions closed, all pending orders cancelled.")
    except Exception as e:
        log.error(f"EOD FLATTEN failed: {e}")
        return

    day = _trading_day_et()
    for symbol, details in closing.items():
        record_trade("FLATTEN", symbol, trading_day_et=day,
                      strategy="end_of_day", reason="Flattened before market close",
                      qty=details.get("qty"))


def run_one_cycle() -> float:
    """
    Runs exactly one check cycle -- clock check, EOD flatten if needed,
    watchlist refresh, circuit breaker check, per-symbol checks -- and
    returns how many seconds the caller should wait before running
    again. Used by the continuous local loop below AND by --once mode
    (e.g. GitHub Actions, where an external scheduler decides when the
    next run happens instead of this process sleeping in between).
    All state that needs to survive between calls (or between separate
    --once processes) is read from/written to module-level globals and
    the two state JSON files, exactly as before -- this function is a
    straight extraction of what used to be the body of the main loop,
    not a behavior change.
    """
    global active_watchlist, last_scan_time, current_trading_day, day_start_equity, \
        daily_loss_breaker_tripped, last_flatten_date

    try:
        clock = trading_client.get_clock()
    except Exception as e:
        log.error(f"Could not reach Alpaca to check market clock: {e}. Retrying in 60s...")
        return 60

    if not clock.is_open:
        wait_seconds = seconds_until(clock.next_open, clock.timestamp)
        log.info(f"Market is closed. Next open: {fmt(clock.next_open)} "
                  f"(your local time) -- in {wait_seconds / 3600:.1f} hours.")
        # Sleep in at-most-1-hour chunks so the log gets periodic heartbeats
        # and so we recover gracefully if the computer was briefly asleep.
        return min(wait_seconds, 3600) + 5

    today_key = clock.timestamp.date()
    if current_trading_day != today_key:
        equity_now_for_day_start = get_account_equity()
        if equity_now_for_day_start is not None:
            day_start_equity = equity_now_for_day_start
            current_trading_day = today_key
            daily_loss_breaker_tripped = False
            save_daily_risk_state()
            log.info(f"Daily loss circuit breaker: tracking from starting equity "
                      f"${day_start_equity:,.2f} (limit -{MAX_DAILY_LOSS_PCT:.0f}%).")
        # else: leave current_trading_day unset so this retries next check.

    seconds_left_today = seconds_until(clock.next_close, clock.timestamp)

    # --- End-of-day window: stop opening new positions, flatten once ---
    if FLATTEN_BEFORE_CLOSE and seconds_left_today <= FLATTEN_MINUTES_BEFORE_CLOSE * 60:
        if last_flatten_date != today_key:
            log.info(f"Within {FLATTEN_MINUTES_BEFORE_CLOSE} min of close -- flattening all positions for the day.")
            flatten_all_positions()
            last_flatten_date = today_key
        return min(seconds_left_today, 60) + 2

    refresh_watchlist_if_needed(clock.timestamp)

    # --- Daily loss circuit breaker check (also doubles as the
    # equity fetch used for position sizing below, so this cycle
    # doesn't need a second account call for that) ---
    entries_paused_reason = None
    equity_now = None
    if seconds_left_today <= STOP_NEW_ENTRIES_MINUTES_BEFORE_CLOSE * 60:
        entries_paused_reason = (f"within {STOP_NEW_ENTRIES_MINUTES_BEFORE_CLOSE} min of close -- "
                                  f"not enough runway left today for a new position")
    else:
        equity_now = get_account_equity()
        if day_start_equity and equity_now is not None:
            daily_pnl_pct = (equity_now - day_start_equity) / day_start_equity * 100
            if daily_pnl_pct <= -MAX_DAILY_LOSS_PCT:
                if not daily_loss_breaker_tripped:
                    log.warning(f"DAILY LOSS CIRCUIT BREAKER: equity down {daily_pnl_pct:.1f}% today "
                                 f"(limit -{MAX_DAILY_LOSS_PCT:.0f}%) -- pausing new entries for the "
                                 f"rest of the day. Existing positions still managed normally.")
                    daily_loss_breaker_tripped = True
                    save_daily_risk_state()
        if daily_loss_breaker_tripped:
            entries_paused_reason = "daily loss circuit breaker tripped"

    open_positions = get_all_open_positions()
    open_position_symbols = set(open_positions.keys())
    open_count = len(open_position_symbols)
    portfolio_risk_estimate = get_current_portfolio_risk_usd(open_positions) if USE_RISK_BASED_SIZING else 0.0
    equity_estimate = equity_now

    symbols_to_check = sorted(set(active_watchlist) | open_position_symbols)

    log.info(f"Market is open (closes {fmt(clock.next_close)} your local time). "
              f"Checking {len(symbols_to_check)} symbols: {', '.join(symbols_to_check)}"
              + (f" ({entries_paused_reason})" if entries_paused_reason else ""))

    bars_by_symbol = {}
    try:
        bars_by_symbol = get_recent_bars_batch(symbols_to_check)
    except Exception as e:
        log.error(f"Could not fetch price data for this cycle's watchlist: {e}")

    for symbol in symbols_to_check:
        at_position_cap = open_count >= MAX_CONCURRENT_POSITIONS
        current_qty = open_positions.get(symbol, {}).get("qty", 0.0)
        notional = check_symbol(symbol, bars_by_symbol.get(symbol), entries_paused_reason,
                                 at_position_cap, current_qty, equity_estimate, portfolio_risk_estimate)
        if notional > 0:
            open_count += 1
            if equity_estimate is not None:
                portfolio_risk_estimate += equity_estimate * RISK_PER_TRADE_PCT / 100
                equity_estimate -= notional

    try:
        clock = trading_client.get_clock()
    except Exception:
        pass

    seconds_left_today = seconds_until(clock.next_close, clock.timestamp) if clock.is_open else 0
    sleep_seconds = min(CHECK_INTERVAL_MINUTES * 60, seconds_left_today)

    if sleep_seconds <= 1:
        log.info("Market closing very soon -- pausing checks until the next session.")
        return 30
    return sleep_seconds


# In --duration-minutes mode, a cycle asking to idle at least this long
# is worth a quick "is the session actually over?" check before we commit
# to waiting it out.
LONG_IDLE_SECONDS = 15 * 60
# ...and if the next open is further away than this, the trading day is
# done (rather than us just sitting in a pre-open gap), so the run can
# end early instead of holding a CI runner all night.
SESSION_OVER_GAP_SECONDS = 2 * 3600


def market_done_for_day() -> bool:
    """True if the market is closed and the next open is hours away."""
    try:
        clock = trading_client.get_clock()
    except Exception:
        return False  # can't tell -- keep going rather than exiting early
    if clock.is_open:
        return False
    return seconds_until(clock.next_open, clock.timestamp) > SESSION_OVER_GAP_SECONDS


def run_for_duration(duration_minutes: float) -> bool:
    """
    Runs cycles continuously for a wall-clock window, then returns.
    Behaves like the continuous local loop, but bounded so a CI job ends
    cleanly and the next scheduled one can take over. Returns True if any
    cycle raised, so the caller can still exit non-zero -- a run that
    failed every cycle must not report success.

    A function rather than inline in __main__ specifically so it can be
    tested against a simulated open market; the market-closed path was
    the only one exercisable live outside trading hours.
    """
    log.info(f"Running in --duration-minutes mode: cycling for {duration_minutes:.0f} "
              f"minutes, then exiting.")
    deadline = time.monotonic() + duration_minutes * 60
    had_error = False
    while time.monotonic() < deadline:
        try:
            sleep_seconds = run_one_cycle()
        except Exception as e:
            # Keep the window alive through a bad cycle, but remember it
            # so the process still exits non-zero -- otherwise a run that
            # failed every single cycle would show up green.
            log.error(f"Unexpected error in cycle, continuing after a short pause: {e}", exc_info=True)
            had_error = True
            sleep_seconds = 30
        if sleep_seconds >= LONG_IDLE_SECONDS and market_done_for_day():
            log.info("Market is closed for the day -- exiting early instead of idling.")
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(sleep_seconds, remaining))
    log.info("Duration window finished.")
    return had_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive intraday trading bot.")
    parser.add_argument("--once", action="store_true",
                         help="Run a single check cycle and exit, instead of looping forever. "
                              "For external schedulers (e.g. GitHub Actions) that decide when the "
                              "next run happens, rather than this process sleeping in between.")
    parser.add_argument("--duration-minutes", type=float, default=0,
                         help="Run cycles continuously for this many minutes, then exit. Bridges "
                              "unreliable external schedulers: GitHub Actions silently drops most "
                              "high-frequency cron ticks (measured 2026-07-24: 5 runs fired out of "
                              "~96 scheduled, and the first one landed 88 minutes after the open), "
                              "so one run has to cover a WINDOW rather than a single instant.")
    args = parser.parse_args()

    active_strategies = []
    if USE_BREAKOUT:
        active_strategies.append("breakout")
    if USE_GAP_PATTERN:
        active_strategies.append("gap_continuation")
    if USE_SMASH_DAY_PATTERN:
        active_strategies.append("smash_day")
    if USE_ROSS_HOOK:
        active_strategies.append("ross_hook")
    if USE_ORB:
        active_strategies.append("orb")
    if USE_RVOL_SPIKE:
        active_strategies.append("rvol_spike")
    if USE_VWAP_REVERSION:
        active_strategies.append("vwap_reversion")
    active_strategies += ["trend_following", "mean_reversion (regime-switched by ADX)"]

    log.info("=== Adaptive Intraday Trading Bot starting (PAPER TRADING MODE) ===")
    if USE_SCANNER:
        log.info(f"Watchlist mode: AUTOMATIC SCANNER (refreshes every {SCANNER_REFRESH_HOURS:.1f}h, "
                  f"top {SCANNER_WATCHLIST_SIZE} movers of {SCANNER_CANDIDATE_POOL} candidates, "
                  f"min price ${SCANNER_MIN_PRICE:.0f}, max extension {SCANNER_MAX_EXTENSION_PCT:.0f}%, "
                  f"leveraged ETFs excluded: {EXCLUDE_LEVERAGED_ETFS}, news filter: {USE_NEWS_FILTER}). "
                  f"Fallback list: {', '.join(SYMBOLS)}")
    else:
        log.info(f"Watchlist mode: MANUAL. Symbols: {', '.join(SYMBOLS)}")
    log.info(f"Bar size: {BAR_MINUTES}min | Check interval while market open: {CHECK_INTERVAL_MINUTES}min")
    log.info(f"Active strategies: {', '.join(active_strategies)}")
    log.info(f"Trend strategy: EMA {FAST_MA}/{SLOW_MA} crossover | Range strategy: RSI {RSI_PERIOD} "
              f"({RSI_OVERSOLD}/{RSI_OVERBOUGHT}) | Switched by ADX {ADX_PERIOD} (threshold {ADX_TREND_THRESHOLD})")
    log.info(f"No new entries between {ENTRY_BLACKOUT_START_MINUTES}-{ENTRY_BLACKOUT_END_MINUTES} min after "
              f"the open (historically weak window in backtesting)")
    if USE_ATR_STOPS:
        log.info(f"Risk management: ATR-based stops on every buy (stop {ATR_STOP_MULTIPLIER:.1f}x ATR / "
                  f"target {ATR_TARGET_MULTIPLIER:.1f}x ATR)")
    else:
        log.info(f"Risk management: fixed stop-loss -{STOP_LOSS_PCT:.0f}% / take-profit +{TAKE_PROFIT_PCT:.0f}% "
                  f"on every buy")
    if USE_RISK_BASED_SIZING:
        log.info(f"Position sizing: risk-based ({RISK_PER_TRADE_PCT:.1f}% of equity per trade, capped at "
                  f"{MAX_POSITION_PCT_OF_EQUITY:.0f}% of equity notional)")
    else:
        log.info(f"Position sizing: flat ${TRADE_AMOUNT_USD:.0f} per trade")
    log.info(f"Position limits: max {MAX_CONCURRENT_POSITIONS} concurrent positions | "
              f"max {MAX_PORTFOLIO_RISK_PCT:.1f}% aggregate portfolio risk | "
              f"daily loss circuit breaker at -{MAX_DAILY_LOSS_PCT:.0f}% (pauses new entries only)")
    if FLATTEN_BEFORE_CLOSE:
        log.info(f"End-of-day mode: positions will be auto-closed {FLATTEN_MINUTES_BEFORE_CLOSE} min before market close.")
    else:
        log.info("End-of-day mode: positions may be held overnight (FLATTEN_BEFORE_CLOSE=false). "
                  "Note: stop-loss/take-profit orders are DAY orders and expire at end of day too -- "
                  "an overnight position temporarily loses that protection until the next session's checks resume.")

    last_flatten_date = None
    active_watchlist = list(SYMBOLS)
    last_scan_time = None
    load_watchlist_state()

    current_trading_day = None
    day_start_equity = None
    daily_loss_breaker_tripped = False
    load_daily_risk_state()

    if args.once:
        # Single-shot mode: one check cycle, then exit. Exceptions are
        # NOT caught here -- letting the process exit non-zero on a real
        # error is what makes it visible as a failed run in an external
        # scheduler (e.g. GitHub Actions), instead of silently vanishing.
        log.info("Running in --once mode (single check cycle, for an external scheduler).")
        run_one_cycle()
    elif args.duration_minutes > 0:
        if run_for_duration(args.duration_minutes):
            raise SystemExit(1)
    else:
        while True:
            try:
                sleep_seconds = run_one_cycle()
            except Exception as e:
                # Last-resort safety net: everything inside run_one_cycle()
                # already has its own specific error handling, but this
                # catches anything unanticipated so one bad cycle doesn't
                # kill the whole process (and with it, every portfolio-
                # level protection that only runs while this loop is alive).
                log.error(f"Unexpected error in main loop, continuing after a short pause: {e}", exc_info=True)
                sleep_seconds = 30
            time.sleep(sleep_seconds)
