"""
Strategy Logic (shared between trading_bot.py and backtest.py)
-----------------------------------------------------------------
This file contains ONLY pure calculation logic -- indicators and the
buy/sell/hold decision rules. It does not connect to Alpaca, place any
orders, or do any logging.

Why this file exists separately: for a backtest to actually mean
anything, it has to run the EXACT same decision logic the live bot
uses. If the live bot and the backtester each had their own copy of
this logic, they'd drift apart over time as one got edited and the
other didn't, and the backtest would quietly stop being a valid test
of what's actually running live. Keeping it in one shared file makes
that impossible.

Indicators vs. decisions (why this file is split this way): every
indicator here (EMA, RSI, ADX, ATR, rolling breakout levels, VWAP,
opening range, pivots) only ever looks backward from a given bar --
none of them peek at future bars. That means computing them ONCE over
an entire price history gives the exact same value at bar i as
recomputing them from scratch on a slice that ends at bar i. The old
version of this file didn't take advantage of that: the backtester
called the decision function on a growing window every single bar,
silently recomputing every indicator from bar zero each time (O(n^2)
for no reason). add_indicators() now does the expensive part once;
decide_signal() and decide_signal_at() just read already-computed
columns. Same math, much less of it.
"""

import os
from datetime import timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MARKET_TZ = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# STRATEGY CONFIG (shared by live trading and backtesting)
# ---------------------------------------------------------------------------

FAST_MA = int(os.getenv("FAST_MA", 9))
SLOW_MA = int(os.getenv("SLOW_MA", 21))

RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
# Tightened from 30/70 after backtesting found mean_reversion had a 44%
# win rate with roughly symmetric win/loss size -- genuinely negative
# expectancy, not just weak. Requiring a more extreme reading raises the
# conviction bar for fewer, hopefully higher-quality entries. See also
# mean_reversion_at()'s new "must already be turning up" requirement.
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", 25))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", 75))

ADX_PERIOD = int(os.getenv("ADX_PERIOD", 14))
ADX_TREND_THRESHOLD = float(os.getenv("ADX_TREND_THRESHOLD", 25))

BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", 20))
# The only entry strategy that didn't have an on/off toggle -- added for
# consistency with every other strategy below, not because of a problem.
USE_BREAKOUT = os.getenv("USE_BREAKOUT", "true").strip().lower() in ("1", "true", "yes")
# Raised from 1.5 to 2.0 after the first real backtest: breakout fired on
# 77 of 239 trades (by far the most of any strategy, since it's a loose
# "new 20-bar high + some extra volume" trigger) but netted only +0.07%/
# trade on average -- essentially a coin flip. Firing less often but more
# selectively is meant to stop it grabbing bars a stronger strategy
# (rvol_spike, vwap_reversion) checked later might otherwise have taken.
BREAKOUT_VOLUME_MULTIPLIER = float(os.getenv("BREAKOUT_VOLUME_MULTIPLIER", 2.0))

# Smash Day Pattern (Type B) -- a Larry Williams reversal pattern, sourced
# from Oxford Capital Strategies' public strategy research (B-rated in
# their own testing). Long side only -- this bot doesn't take short
# positions. See smash_day_signal() docstring for the exact rule and how
# it's adapted from an intrabar stop order to this bot's bar-close checks.
# Default FLIPPED TO OFF after the first real backtest (TSLA/NVDA/COIN,
# ~3 months): 78 trades, net -$35.23, and 82% of them just drifted to the
# end-of-day flatten instead of exiting on a real signal -- the worst
# combination (no edge, no exit discipline either). Kept in code, not
# deleted, since 78 trades on 3 correlated symbols over one window is
# still a thin sample -- re-enable and re-test if you want a second look.
USE_SMASH_DAY_PATTERN = os.getenv("USE_SMASH_DAY_PATTERN", "false").strip().lower() in ("1", "true", "yes")

# Gap Pattern (Type A) -- the other Oxford B-rated pattern, previously
# deferred (see gap_continuation_signal() docstring for sourcing caveat).
# Fired ZERO times in the first backtest (TSLA/NVDA/COIN didn't gap >
# GAP_MIN_PCT in that window) -- left on since that's "unproven," not
# "proven bad," unlike smash_day/ross_hook below.
USE_GAP_PATTERN = os.getenv("USE_GAP_PATTERN", "true").strip().lower() in ("1", "true", "yes")
GAP_MIN_PCT = float(os.getenv("GAP_MIN_PCT", 2.0))

# Ross Hook -- previously deferred pending swing-pivot detection (see
# ross_hook_signal() docstring for the simplified adaptation used here).
# Default FLIPPED TO OFF after the first real backtest: worst average
# per-trade return of any strategy (-0.58%), net -$28.83 on 10 trades.
# Small sample (10 trades), so this is a real but not overwhelming
# signal -- kept in code and toggleable, same reasoning as smash_day above.
USE_ROSS_HOOK = os.getenv("USE_ROSS_HOOK", "false").strip().lower() in ("1", "true", "yes")
ROSS_HOOK_PIVOT_LOOKBACK = int(os.getenv("ROSS_HOOK_PIVOT_LOOKBACK", 3))
ROSS_HOOK_SCAN_BARS = int(os.getenv("ROSS_HOOK_SCAN_BARS", 40))

# Opening Range Breakout -- standard, well-documented intraday pattern.
USE_ORB = os.getenv("USE_ORB", "true").strip().lower() in ("1", "true", "yes")
ORB_MINUTES = int(os.getenv("ORB_MINUTES", 15))

# VWAP mean-reversion -- BUY when price is stretched below session VWAP
# and starting to turn back up.
USE_VWAP_REVERSION = os.getenv("USE_VWAP_REVERSION", "true").strip().lower() in ("1", "true", "yes")
VWAP_REVERSION_PCT = float(os.getenv("VWAP_REVERSION_PCT", 1.5))

# Relative volume (RVOL) spike -- an unusual volume surge on a green bar,
# independent of the breakout strategy's "new high" requirement.
# Default FLIPPED TO OFF after a SECOND backtest still showed it net
# negative (-$3,047.66 across 101 trades, 50% win rate) despite adding
# the RVOL_MIN_CLOSE_STRENGTH confirmation below specifically to fix
# this -- that fix reduced trade count (~155 -> 101) but didn't actually
# improve the win rate or flip it positive. Being straight about it: the
# attempted fix didn't work. Kept in code and toggleable, same as
# smash_day/ross_hook, in case a larger sample or a different fix (e.g.
# a higher RVOL_MULTIPLIER) tells a different story.
USE_RVOL_SPIKE = os.getenv("USE_RVOL_SPIKE", "false").strip().lower() in ("1", "true", "yes")
RVOL_LOOKBACK = int(os.getenv("RVOL_LOOKBACK", 20))
RVOL_MULTIPLIER = float(os.getenv("RVOL_MULTIPLIER", 3.0))
# Added after backtesting showed rvol_spike accounted for 13 of 23 stop-
# loss hits across the whole strategy set (57% of all of them) and had a
# worse average loss (-2.15%) than average win (+1.89%) despite a >50%
# win rate -- a volume spike alone wasn't enough confirmation, too many
# false starts. Requires the spike bar to ALSO close in the upper portion
# of its own high-low range (a strong close, not just barely green).
RVOL_MIN_CLOSE_STRENGTH = float(os.getenv("RVOL_MIN_CLOSE_STRENGTH", 0.66))

BAR_MINUTES = int(os.getenv("BAR_MINUTES", 5))
TRADE_AMOUNT_USD = float(os.getenv("TRADE_AMOUNT_USD", 500))

FLATTEN_BEFORE_CLOSE = os.getenv("FLATTEN_BEFORE_CLOSE", "true").strip().lower() in ("1", "true", "yes")
FLATTEN_MINUTES_BEFORE_CLOSE = int(os.getenv("FLATTEN_MINUTES_BEFORE_CLOSE", 10))

# Stop opening NEW positions this many minutes before close (existing
# positions are still monitored and exited normally -- this only blocks
# fresh entries). Backtesting showed the last ~90 minutes of the session
# performing meaningfully worse than the rest of the day, likely because
# positions opened that late don't have enough runway before being
# forced closed by the end-of-day flatten.
STOP_NEW_ENTRIES_MINUTES_BEFORE_CLOSE = int(os.getenv("STOP_NEW_ENTRIES_MINUTES_BEFORE_CLOSE", 90))

# Backtesting found entries between 1-2pm ET were a net drag (negative
# combined P&L, mediocre win rate) while 11am-12pm ET was the best-
# performing window despite getting far fewer trades -- most entries
# cluster right at the 9:30 open instead. Blocks NEW entries during this
# historically weak window (existing positions still managed normally),
# expressed as minutes since the 9:30 session open, same style as
# STOP_NEW_ENTRIES_MINUTES_BEFORE_CLOSE above. Enforced by the caller
# (trading_bot.py / backtest.py), not inside decide_signal, matching how
# the close-proximity cutoff already works.
ENTRY_BLACKOUT_START_MINUTES = int(os.getenv("ENTRY_BLACKOUT_START_MINUTES", 210))  # 1:00pm ET
ENTRY_BLACKOUT_END_MINUTES = int(os.getenv("ENTRY_BLACKOUT_END_MINUTES", 270))  # 2:00pm ET

# Risk management: fixed-% exit levels (the default), as a % away from entry.
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", 5))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", 10))

# Risk management: optional volatility-adjusted exit levels instead of a
# flat % for every stock. Off by default -- flip on and backtest first.
USE_ATR_STOPS = os.getenv("USE_ATR_STOPS", "false").strip().lower() in ("1", "true", "yes")
ATR_PERIOD = int(os.getenv("ATR_PERIOD", 14))
ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER", 1.5))
ATR_TARGET_MULTIPLIER = float(os.getenv("ATR_TARGET_MULTIPLIER", 3.0))

# Risk management: position SIZE based on how much you're willing to
# lose if the stop is hit, not a flat dollar amount. A volatile stock
# with a wide stop naturally gets fewer shares than a calm one with a
# tight stop for the same $ risk, and size scales automatically as
# account equity grows or shrinks -- unlike a flat TRADE_AMOUNT_USD,
# which stops meaning the same thing over time and treats every stock as
# equally risky. On by default: this was identified as a real weakness
# (not cosmetic), directly evidenced by MSTR's 42% single-symbol
# drawdown in backtesting under the old flat-dollar sizing.
USE_RISK_BASED_SIZING = os.getenv("USE_RISK_BASED_SIZING", "true").strip().lower() in ("1", "true", "yes")
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", 1.0))
# Hard cap on notional position size regardless of stop distance, so a
# very tight stop (e.g. a calm stock under ATR stops) can't imply an
# absurdly large position.
MAX_POSITION_PCT_OF_EQUITY = float(os.getenv("MAX_POSITION_PCT_OF_EQUITY", 25.0))


# ---------------------------------------------------------------------------
# INDICATORS -- each one only ever looks backward from a given row, so
# computing them once over a full price history is equivalent to (and much
# cheaper than) recomputing them on every growing slice of that history.
# ---------------------------------------------------------------------------

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Standard RSI (Wilder's smoothing). Ranges 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """Shared by ADX and ATR so it's only ever computed once."""
    high, low, close = df["high"], df["low"], df["close"]
    return pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)


def compute_adx(df: pd.DataFrame, period: int, true_range: pd.Series | None = None) -> pd.Series:
    """Standard ADX (Wilder's smoothing). Ranges roughly 0-100; higher = stronger trend."""
    high, low = df["high"], df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = true_range if true_range is not None else compute_true_range(df)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx


def compute_atr(true_range: pd.Series, period: int) -> pd.Series:
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _session_date(df: pd.DataFrame) -> pd.Series:
    ts = df["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(timezone.utc)
    return ts.dt.tz_convert(MARKET_TZ).dt.date


def _minutes_since_open(df: pd.DataFrame) -> pd.Series:
    ts = df["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(timezone.utc)
    et_time = ts.dt.tz_convert(MARKET_TZ)
    session_open = et_time.apply(lambda t: t.replace(hour=9, minute=30, second=0, microsecond=0))
    return (et_time - session_open).dt.total_seconds() / 60


def compute_vwap(df: pd.DataFrame, session_date: pd.Series) -> pd.Series:
    """Session-cumulative volume-weighted average price, resetting each trading day."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]
    cum_pv = pv.groupby(session_date).cumsum()
    cum_vol = df["volume"].groupby(session_date).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def compute_opening_range(df: pd.DataFrame, session_date: pd.Series, minutes_since_open: pd.Series,
                           orb_minutes: int) -> tuple[pd.Series, pd.Series]:
    """
    Per-session opening range high/low, built from only the first
    orb_minutes of each session and broadcast across that whole session's
    rows -- so every bar in a session sees the same opening range, and it
    only reflects bars that had actually printed by the time it formed.
    """
    in_opening_range = minutes_since_open < orb_minutes
    high_in_range = df["high"].where(in_opening_range)
    low_in_range = df["low"].where(in_opening_range)
    orb_high = high_in_range.groupby(session_date).cummax().groupby(session_date).ffill()
    orb_low = low_in_range.groupby(session_date).cummin().groupby(session_date).ffill()
    return orb_high, orb_low


def compute_prior_session_close(df: pd.DataFrame, session_date: pd.Series) -> pd.Series:
    """Each session's rows get the PREVIOUS session's final close (for gap %)."""
    last_close_by_session = df.groupby(session_date)["close"].last()
    prior_close_by_session = last_close_by_session.shift(1)
    return session_date.map(prior_close_by_session)


def compute_pivots(df: pd.DataFrame, lookback: int) -> tuple[pd.Series, pd.Series]:
    """
    A bar is a pivot low/high if its low/high is the most extreme within
    `lookback` bars on EITHER side. Uses a centered rolling window over the
    full history (safe to precompute once -- see module docstring), but
    callers must only treat a pivot at index j as "known" once the current
    index is at least j + lookback, since that's the earliest point the
    bars needed to confirm it have actually happened.
    """
    window = 2 * lookback + 1
    is_pivot_low = df["low"] == df["low"].rolling(window, center=True, min_periods=window).min()
    is_pivot_high = df["high"] == df["high"].rolling(window, center=True, min_periods=window).max()
    return is_pivot_low.fillna(False), is_pivot_high.fillna(False)


# ---------------------------------------------------------------------------
# INDICATOR PRECOMPUTE -- runs once per dataframe, adds all columns the
# decision functions below need. Cheap on the live bot's small per-check
# dataframe; the real payoff is the backtester calling this ONCE per symbol
# instead of once per bar.
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema_fast"] = compute_ema(df["close"], FAST_MA)
    df["ema_slow"] = compute_ema(df["close"], SLOW_MA)
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)

    true_range = compute_true_range(df)
    df["adx"] = compute_adx(df, ADX_PERIOD, true_range=true_range)
    df["atr"] = compute_atr(true_range, ATR_PERIOD)

    # Breakout: prior BREAKOUT_LOOKBACK bars' high/avg-volume, excluding
    # the current bar (shift(1) before the rolling window).
    df["breakout_recent_high"] = df["close"].shift(1).rolling(BREAKOUT_LOOKBACK).max()
    df["breakout_avg_volume"] = df["volume"].shift(1).rolling(BREAKOUT_LOOKBACK).mean()

    # Smash Day setup/trigger, vectorized from the original bar-relative rule.
    df["smash_day_setup"] = df["close"].shift(1) < df["low"].shift(2)
    df["smash_day_trigger"] = df["high"] > df["high"].shift(1)

    # Relative volume vs. its own recent average, excluding the current bar.
    df["rvol_avg_volume"] = df["volume"].shift(1).rolling(RVOL_LOOKBACK).mean()

    session_date = _session_date(df)
    df["_session_date"] = session_date
    df["minutes_since_open"] = _minutes_since_open(df)
    df["vwap"] = compute_vwap(df, session_date)

    orb_high, orb_low = compute_opening_range(df, session_date, df["minutes_since_open"], ORB_MINUTES)
    df["orb_high"] = orb_high
    df["orb_low"] = orb_low

    df["prior_session_close"] = compute_prior_session_close(df, session_date)
    df["session_open_price"] = df.groupby(session_date)["open"].transform("first")
    df["session_first_bar_high"] = df.groupby(session_date)["high"].transform("first")

    is_pivot_low, is_pivot_high = compute_pivots(df, ROSS_HOOK_PIVOT_LOOKBACK)
    df["is_pivot_low"] = is_pivot_low
    df["is_pivot_high"] = is_pivot_high

    return df


# ---------------------------------------------------------------------------
# STRATEGIES -- each takes the indicator-enriched df and a row index i,
# and returns True/False for whether it fires a BUY at that bar.
# ---------------------------------------------------------------------------

def breakout_at(df: pd.DataFrame, i: int) -> bool:
    """
    Volume-confirmed breakout above the recent BREAKOUT_LOOKBACK-bar
    high, requiring a confirmation bar: the prior bar must have already
    broken out (above ITS OWN reference level), and the current bar must
    still be pushing higher, not just the very first bar to touch the
    level. Same "wait for the move to actually hold" requirement already
    used by vwap_reversion/mean_reversion.
    """
    if i < 1:
        return False
    recent_high = df["breakout_recent_high"].iat[i]
    avg_volume = df["breakout_avg_volume"].iat[i]
    prev_recent_high = df["breakout_recent_high"].iat[i - 1]
    if pd.isna(recent_high) or pd.isna(avg_volume) or avg_volume == 0 or pd.isna(prev_recent_high):
        return False

    close_now = df["close"].iat[i]
    close_prev = df["close"].iat[i - 1]
    broke_out_last_bar = close_prev > prev_recent_high
    still_pushing = close_now > recent_high and close_now > close_prev
    volume_confirmed = df["volume"].iat[i] > avg_volume * BREAKOUT_VOLUME_MULTIPLIER
    return broke_out_last_bar and still_pushing and volume_confirmed


def smash_day_at(df: pd.DataFrame, i: int) -> bool:
    """
    Smash Day Pattern (Type B) -- developed by Larry Williams, sourced
    from Oxford Capital Strategies' public research (graded B in their
    own testing). Long side only -- this bot doesn't take short
    positions, so the short-side rule from the original pattern is
    omitted.

    Concept: a "smash day" is a bar that closes below the low of two
    bars ago -- an unusually sharp, likely-overdone move down. If price
    then pushes back above that smash bar's own high, it's read as a
    failed breakdown / reversal, and triggers a long entry.

    Original rule (Oxford's notation, i = the entry bar):
      Setup:  Close[i-1] < Low[i-2]
      Entry:  a buy stop one tick above High[i-1]
    Adaptation: this bot checks signals once per bar close rather than
    watching for an intrabar stop order, so "entry" here means the
    CURRENT bar's high has already closed above the smash bar's high --
    one bar later than a live intrabar stop would trigger, which is a
    real (if usually small) difference worth knowing about.
    """
    if i < 2:
        return False
    return bool(df["smash_day_setup"].iat[i] and df["smash_day_trigger"].iat[i])


def gap_continuation_at(df: pd.DataFrame, i: int) -> bool:
    """
    Gap Pattern (Type A) -- the other Oxford Capital Strategies B-rated
    pattern (see CLAUDE.md), previously deferred because gaps are a
    daily-bar concept that needed day-boundary detection first.

    Sourcing caveat, same honesty standard as smash_day_at(): the exact
    Oxford Type A rule text was never fully sourced, so this is a
    standard, well-documented "gap and go" continuation implementation
    rather than a verified reproduction of their exact rule -- gap up
    at least GAP_MIN_PCT vs. the prior session's close, then a later
    bar in that same session breaks above the opening bar's high
    (confirming the gap is extending rather than filling).
    """
    prior_close = df["prior_session_close"].iat[i]
    session_open = df["session_open_price"].iat[i]
    first_bar_high = df["session_first_bar_high"].iat[i]
    if pd.isna(prior_close) or prior_close == 0 or pd.isna(first_bar_high):
        return False
    if df["minutes_since_open"].iat[i] <= 0:
        return False  # this bar IS the opening bar -- nothing to confirm yet

    gap_pct = (session_open - prior_close) / prior_close * 100
    if gap_pct < GAP_MIN_PCT:
        return False

    close_now = df["close"].iat[i]
    close_prev = df["close"].iat[i - 1]
    # Fire on the FIRST bar that breaks the opening bar's high, not every
    # bar afterward that happens to still be above it.
    return close_now > first_bar_high and close_prev <= first_bar_high


def ross_hook_at(df: pd.DataFrame, i: int) -> bool:
    """
    Ross Hook -- the other previously-deferred Oxford strategy (needed
    swing-pivot detection, per CLAUDE.md). Simplified adaptation of the
    classic 1-2-3 reversal + hook entry:
      Point 1: a confirmed pivot low
      Point 2: the next confirmed pivot high after it (the bounce)
      Point 3: the next confirmed pivot low after that, HIGHER than
               point 1's low (confirms the pullback is holding, not
               making a new low -- the structural "3")
      Hook entry: price breaks above point 3's own bar's high for the
               first time (the "hook").
    A pivot at index j only counts as "confirmed" once we're at least
    ROSS_HOOK_PIVOT_LOOKBACK bars past it (see compute_pivots) --
    otherwise this would be using bars that hadn't happened yet at
    decision time i.
    """
    k = ROSS_HOOK_PIVOT_LOOKBACK
    confirmed_cutoff = i - k
    if confirmed_cutoff < 0:
        return False

    start = max(0, i - ROSS_HOOK_SCAN_BARS)
    lows = df["low"].values
    highs = df["high"].values
    is_pivot_low = df["is_pivot_low"].values
    is_pivot_high = df["is_pivot_high"].values

    # Walk backward from "now" to find the most recent completed 1-2-3:
    # point 3 (nearest confirmed pivot low) -> point 2 (nearest confirmed
    # pivot high before that) -> point 1 (nearest confirmed pivot low
    # before THAT). Scanning backward for the nearest point 3 first (not
    # forward from an arbitrary point 1) is what makes this find the
    # structure that's actually still relevant to the current bar.
    p3 = None
    for j in range(confirmed_cutoff, start - 1, -1):
        if is_pivot_low[j]:
            p3 = j
            break
    if p3 is None:
        return False

    p2 = None
    for j in range(p3 - 1, start - 1, -1):
        if is_pivot_high[j]:
            p2 = j
            break
    if p2 is None:
        return False

    p1 = None
    for j in range(p2 - 1, start - 1, -1):
        if is_pivot_low[j]:
            p1 = j
            break
    if p1 is None:
        return False

    if not (lows[p3] > lows[p1]):
        return False

    hook_high = highs[p3]
    close_now = df["close"].iat[i]
    close_prev = df["close"].iat[i - 1] if i > 0 else -np.inf
    return close_now > hook_high and close_prev <= hook_high


def orb_at(df: pd.DataFrame, i: int) -> bool:
    """
    Opening Range Breakout, requiring a confirmation bar: the prior bar
    must have already closed above the opening range high, and the
    current bar must still be pushing higher -- not firing on the very
    first bar to cross it. Same "wait for the move to actually hold"
    requirement already used by vwap_reversion/mean_reversion.
    """
    if i < 1:
        return False
    if df["minutes_since_open"].iat[i] < ORB_MINUTES or df["minutes_since_open"].iat[i - 1] < ORB_MINUTES:
        return False
    orb_high = df["orb_high"].iat[i]
    if pd.isna(orb_high):
        return False
    close_now = df["close"].iat[i]
    close_prev = df["close"].iat[i - 1]
    broke_out_last_bar = close_prev > orb_high
    still_pushing = close_now > orb_high and close_now > close_prev
    return broke_out_last_bar and still_pushing


def vwap_reversion_at(df: pd.DataFrame, i: int) -> bool:
    """BUY when price is stretched VWAP_REVERSION_PCT% below session VWAP and just turned back up."""
    vwap = df["vwap"].iat[i]
    if pd.isna(vwap) or vwap == 0 or i < 1:
        return False
    close_now = df["close"].iat[i]
    close_prev = df["close"].iat[i - 1]
    stretched_below = close_now < vwap * (1 - VWAP_REVERSION_PCT / 100)
    turning_up = close_now > close_prev
    return bool(stretched_below and turning_up)


def rvol_spike_at(df: pd.DataFrame, i: int) -> bool:
    """
    Unusual volume surge (independent of the breakout strategy's 'new
    high' requirement) on a green bar that ALSO closes in the upper
    portion of its own high-low range (RVOL_MIN_CLOSE_STRENGTH) --
    added after backtesting showed a volume spike + merely-green close
    wasn't enough confirmation on its own (see RVOL_MIN_CLOSE_STRENGTH
    docstring above for the numbers).
    """
    avg_volume = df["rvol_avg_volume"].iat[i]
    if pd.isna(avg_volume) or avg_volume == 0:
        return False
    volume_now = df["volume"].iat[i]
    close_now = df["close"].iat[i]
    open_now = df["open"].iat[i]
    high_now = df["high"].iat[i]
    low_now = df["low"].iat[i]
    bar_range = high_now - low_now
    close_strength = (close_now - low_now) / bar_range if bar_range > 0 else 1.0
    return (volume_now > avg_volume * RVOL_MULTIPLIER
            and close_now > open_now
            and close_strength >= RVOL_MIN_CLOSE_STRENGTH)


def trend_following_at(df: pd.DataFrame, i: int) -> str:
    """Fast/slow EMA crossover. BUY on golden cross, SELL on death cross."""
    if i < 1:
        return "HOLD"
    prev_fast, prev_slow = df["ema_fast"].iat[i - 1], df["ema_slow"].iat[i - 1]
    curr_fast, curr_slow = df["ema_fast"].iat[i], df["ema_slow"].iat[i]

    if prev_fast <= prev_slow and curr_fast > curr_slow:
        return "BUY"
    elif prev_fast >= prev_slow and curr_fast < curr_slow:
        return "SELL"
    return "HOLD"


def mean_reversion_at(df: pd.DataFrame, i: int) -> str:
    """
    RSI-based. BUY when oversold AND price has already started turning
    back up -- added after backtesting showed a 44% win rate buying on
    RSI-oversold alone, with no confirmation (catching a still-falling
    knife). SELL side (exits) deliberately left as pure RSI overbought,
    unchanged: this same function also governs exits for ANY position
    held during a choppy regime, not just entries it opened itself, and
    the evidence was specifically about entry quality, not exit timing.
    """
    current_rsi = df["rsi"].iat[i]
    if pd.isna(current_rsi):
        return "HOLD"
    if current_rsi < RSI_OVERSOLD:
        if i >= 1 and df["close"].iat[i] > df["close"].iat[i - 1]:
            return "BUY"
        return "HOLD"
    elif current_rsi > RSI_OVERBOUGHT:
        return "SELL"
    return "HOLD"


# ---------------------------------------------------------------------------
# DECISION -- entry strategies are checked in order (first BUY wins); exits
# for an already-open position are governed by whichever regime strategy
# (trend-following or mean-reversion) is currently active, same as before.
# ---------------------------------------------------------------------------

def _decide_from_indicators(df: pd.DataFrame, i: int) -> tuple[str, str, str]:
    """
    Core decision logic, reading only already-computed indicator columns
    at row i. Returns (signal, reason_key, reason_text).

    Entry strategies are checked in this order (first BUY wins).
    Reordered after the first real backtest (TSLA/NVDA/COIN, ~3 months,
    239 trades -- see CLAUDE.md) to put strategies with a demonstrated
    edge ahead of breakout, which originally fired on ~1/3 of all trades
    but netted almost nothing per trade -- letting it fire first was
    likely grabbing bars a stronger signal would otherwise have taken.
    Priority position itself is left alone even for strategies later
    turned off by default (rvol_spike, smash_day, ross_hook) -- if you
    re-enable one, it fires in the slot this history explains, not at
    the front by default:
      1. rvol_spike   -- OFF BY DEFAULT (see USE_RVOL_SPIKE above): net
                        negative in TWO separate backtests, including
                        one with a targeted fix that didn't help.
      2. vwap_reversion (consistently the best strategy across three
                        backtests now -- 65-80% win rate each time)
      3. orb          (positive but regressing toward breakeven with
                        more data -- watch, don't over-trust yet)
      4. gap_continuation (still nearly unproven -- only 1-3 trades total
                        across all backtests so far)
      5. breakout     (originally weak, turned solidly positive after
                        BREAKOUT_VOLUME_MULTIPLIER was tightened -- see
                        that constant's comment)
      6/7. smash_day, ross_hook -- both net losers, off by default (see
                        their USE_ flags above), placed last so if
                        re-enabled they only fire as a last resort
      8. ADX-gated regime switch (trend-following / mean-reversion) --
         also what governs exits for a position already held.
    """
    min_bars_needed = max(SLOW_MA, RSI_PERIOD, ADX_PERIOD, BREAKOUT_LOOKBACK) + 5
    if i + 1 < min_bars_needed:
        return "HOLD", "warmup", f"warming up ({i + 1}/{min_bars_needed} bars collected)"

    if USE_RVOL_SPIKE and rvol_spike_at(df, i):
        return "BUY", "rvol_spike", f"relative volume spike ({RVOL_MULTIPLIER:.1f}x average) on a green bar"

    if USE_VWAP_REVERSION and vwap_reversion_at(df, i):
        return "BUY", "vwap_reversion", "stretched below VWAP and turning back up"

    if USE_ORB and orb_at(df, i):
        return "BUY", "orb", f"Opening Range Breakout (broke the first {ORB_MINUTES}-min range high)"

    if USE_GAP_PATTERN and gap_continuation_at(df, i):
        return "BUY", "gap_continuation", "Gap Pattern (Type A) continuation, gap held and extended"

    if USE_BREAKOUT and breakout_at(df, i):
        return "BUY", "breakout", f"volume-confirmed breakout above {BREAKOUT_LOOKBACK}-bar high"

    if USE_SMASH_DAY_PATTERN and smash_day_at(df, i):
        return "BUY", "smash_day", "Smash Day reversal (failed breakdown, price reclaimed the smash bar's high)"

    if USE_ROSS_HOOK and ross_hook_at(df, i):
        return "BUY", "ross_hook", "Ross Hook (1-2-3 reversal, price broke the hook bar's high)"

    current_adx = df["adx"].iat[i]
    if pd.isna(current_adx):
        return "HOLD", "warmup", "warming up (ADX not ready yet)"

    if current_adx >= ADX_TREND_THRESHOLD:
        signal = trend_following_at(df, i)
        reason = f"trending (ADX {current_adx:.1f}) -> trend-following strategy"
        reason_key = "trend_following"
    else:
        signal = mean_reversion_at(df, i)
        reason = f"choppy/sideways (ADX {current_adx:.1f}) -> mean-reversion strategy"
        reason_key = "mean_reversion"

    return signal, reason_key, reason


def decide_signal(df: pd.DataFrame) -> tuple[str, str]:
    """
    Live-bot entry point: takes a raw OHLCV dataframe (no indicators yet),
    computes them, and decides a signal for the LAST row. Returns
    (signal, reason_text) -- kept as a 2-tuple for backward compatibility
    with existing callers that only care about the human-readable reason.
    """
    enriched = add_indicators(df)
    signal, _reason_key, reason_text = _decide_from_indicators(enriched, len(enriched) - 1)
    return signal, reason_text


def decide_signal_at(df: pd.DataFrame, i: int) -> tuple[str, str, str]:
    """
    Backtest entry point: df must already have gone through add_indicators()
    ONCE (by the caller, outside any per-bar loop). Returns
    (signal, reason_key, reason_text) for row i.
    """
    return _decide_from_indicators(df, i)


# ---------------------------------------------------------------------------
# RISK MANAGEMENT -- stop-loss / take-profit levels for a new position.
# Shared by the live bot (place_buy_order) and the backtester (simulate)
# so the two can't drift apart on this either.
# ---------------------------------------------------------------------------

def compute_stop_and_target(entry_price: float, atr_value: float | None = None) -> tuple[float, float]:
    """
    Returns (stop_price, take_profit_price) for a new long position.
    Uses ATR-scaled distances if USE_ATR_STOPS is on AND a usable ATR
    value was passed in; otherwise falls back to the fixed
    STOP_LOSS_PCT / TAKE_PROFIT_PCT (the default behavior).
    """
    if USE_ATR_STOPS and atr_value is not None and not pd.isna(atr_value) and atr_value > 0:
        stop_price = entry_price - ATR_STOP_MULTIPLIER * atr_value
        take_profit_price = entry_price + ATR_TARGET_MULTIPLIER * atr_value
    else:
        stop_price = entry_price * (1 - STOP_LOSS_PCT / 100)
        take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT / 100)
    return stop_price, take_profit_price


def compute_position_size(equity: float, entry_price: float, stop_price: float) -> int:
    """
    Returns how many whole shares to buy, sized off how much you're
    willing to LOSE if the stop is hit (RISK_PER_TRADE_PCT of current
    equity) rather than a flat dollar amount. A wide stop (a volatile
    stock, or an ATR stop on one) naturally gets fewer shares than a
    tight stop (a calm stock) for the same dollar risk, and the whole
    thing scales automatically as equity grows or shrinks.
    MAX_POSITION_PCT_OF_EQUITY caps the notional size regardless, so a
    very tight stop can't imply an absurdly large position.
    Returns 0 if risk-based sizing can't be computed (invalid inputs) --
    callers should fall back to flat-dollar sizing in that case.
    """
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0 or equity <= 0:
        return 0
    risk_amount = equity * RISK_PER_TRADE_PCT / 100
    qty_by_risk = risk_amount / risk_per_share
    max_notional = equity * MAX_POSITION_PCT_OF_EQUITY / 100
    qty_by_cap = max_notional / entry_price
    return int(min(qty_by_risk, qty_by_cap))
