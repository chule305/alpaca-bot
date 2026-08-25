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
# Hysteresis band around ADX_TREND_THRESHOLD, OFF by default -- a single
# hard cutoff means a stock chopping around ADX 25 (which happens exactly
# at the boundary between "trending" and "choppy", i.e. often) can flip
# trend_following <-> mean_reversion bar to bar. Each flip isn't free: it
# swaps in a strategy with a different exit rule (trend_following_at's
# death-cross vs. mean_reversion_at's RSI-overbought) mid-position, so a
# whipsaw at the ADX line can cut a real trend short on a one-bar dip back
# under 25, or hold a losing mean-reversion trade past where its own rule
# would have exited because ADX ticked up first. Plausible, not yet
# proven -- ships toggleable so the backtest gets the final word rather
# than a theory. See add_indicators()'s adx_trend_regime column for the
# actual band-crossing computation (must be a precomputed column, not a
# loop variable, per this file's compute-once pattern).
USE_ADX_HYSTERESIS = os.getenv("USE_ADX_HYSTERESIS", "false").strip().lower() in ("1", "true", "yes")
ADX_HYSTERESIS_BAND = float(os.getenv("ADX_HYSTERESIS_BAND", 3))

# Second regime axis, independent of ADX (Moreira & Muir 2017,
# "Volatility-Managed Portfolios", J. Finance): everything above measures
# DIRECTIONAL PERSISTENCE (trending vs. choppy) -- it says nothing about
# MAGNITUDE. A stock can have low ADX while its realized volatility is
# either very high (violent whipsaws -- e.g. VEEE's 6.0-7.3%/bar ATR
# days, see stop_is_wider_than_noise's docstring) or very low (quiet
# drift). Moreira & Muir found that scaling exposure DOWN when trailing
# realized vol is high (and up when low) improves risk-adjusted returns
# through a mechanism that's orthogonal to trend strength -- this bot
# only acts on the DOWN half (see USE_VOLATILITY_SCALED_SIZING near
# TRADE_AMOUNT_USD below for why, and why it's a fixed-dollar reduction,
# never a %-of-equity one).
#
# Reuses the ATR column computed a few lines below in add_indicators()
# rather than adding a second realized-vol indicator -- ATR is already
# this bot's validated per-symbol noise measure (MIN_STOP_TO_ATR_RATIO,
# STOP_SLIPPAGE_ATR_FRACTION in backtest.py both lean on it). Expressed
# as ATR-as-%-of-price (not raw dollars) so a $5 stock and a $500 stock
# aren't compared on different absolute scales, then ranked with a
# TRAILING rolling window against ONLY that SAME symbol's own history --
# never a cross-symbol comparison. VEEE's "high vol" and NVDA's "high
# vol" are different absolute ATR% by design; this is each stock's own
# regime relative to itself, not a cross-sectional ranking.
VOL_PERCENTILE_LOOKBACK = int(os.getenv("VOL_PERCENTILE_LOOKBACK", 90))
# Terciles split at 1/3 and 2/3 by definition -- not exposed as an env
# var (unlike VOL_PERCENTILE_LOOKBACK above) because moving this number
# would silently redefine what "tercile" means rather than tune a
# behavior. The levers meant to be tuned are whether the top bucket
# affects sizing at all (USE_VOLATILITY_SCALED_SIZING) and by how much
# (VOLATILITY_SCALED_REDUCED_USD), both below.
HIGH_VOL_TERCILE_CUTOFF = 2.0 / 3.0

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

# Same-time-of-day historical volume floor instead of a flat trailing
# average, e.g. a 9:30-10:00am bar only gets compared against OTHER
# 9:30-10:00am bars from prior sessions, not the whole day's flat
# BREAKOUT_LOOKBACK-bar average. Motivation: intraday volume follows the
# well-documented "U-shape" (heaviest right at the open, thinnest around
# midday, picking back up into the close), so a bar's volume being
# "unusual" should be judged against what's normal for THAT time of day,
# not a blended average that doesn't know what time it is.
#
# The actual direction of the mispricing this fixes, measured on real data
# (TSLA, 90 days), is the OPPOSITE of the naive guess: BREAKOUT_LOOKBACK=20
# bars at BAR_MINUTES=15 is only a 5-hour trailing window, and because it's
# a continuous rolling window (not reset per session) the early bars of a
# session are still mostly averaging in the QUIET back half of the PRIOR
# session, not today's own open. Measured avg volume by 30-min bucket vs.
# the flat trailing average it's compared against:
#   9:30-10:00  actual ~3.24M/bar  vs. flat trailing avg ~1.31M  (2.5x)
#   12:30-1:00  actual ~0.98M/bar  vs. flat trailing avg ~1.69M  (0.6x)
# So the flat average was actually the LOOSER bar at the open (real volume
# already runs ~2.5x it before any breakout-specific surge) and the
# STRICTER one at midday (real volume runs well under it) -- backwards
# from "elevated morning volume should make the bar easier to read as
# unusual," but the same root cause: a trailing average blind to time of
# day misprices "unusual" in both directions. See
# compute_time_of_day_avg_volume() for how the corrected comparison
# average is built without lookahead.
USE_TIME_OF_DAY_VOLUME_NORM = os.getenv("USE_TIME_OF_DAY_VOLUME_NORM", "true").strip().lower() in ("1", "true", "yes")
# Width of the minutes-since-open bucket used for the historical average
# above. 30 (not 15, i.e. not 1:1 with BAR_MINUTES) trades a little
# precision for faster warmup: at BAR_MINUTES=15 a 15-wide bucket gets
# exactly one bar of history per prior session added to its average, while
# a 30-wide bucket gets two, roughly halving how many sessions it takes
# before a bucket's average is built on a meaningful sample.
#
# 90-day backtest, both universes, same-symbol-set A/B, everything else at
# defaults (2026-08-06, see CLAUDE.md):
#   megacap (TSLA/NVDA/COIN/AMD/PLTR): 99 trades -> 78, win rate 60% -> 62%,
#     profit factor 1.78 -> 2.37, return +7.5% -> +8.6%, max DD 1.7% -> 1.4%
#   scanner (FBRX/VEEE/PN/TRAX/QBTS/SMCI/SAFT/RNG/INHD/FCUV/ATKR/SRAD):
#     96 trades -> 99, win rate 49% -> 48%, profit factor 1.15 -> 1.22,
#     return +1.3% -> +1.8%, max DD 1.8% -> 1.7%
# Both universes improve on profit factor, return, and drawdown; megacap
# trades far less often but at meaningfully higher quality (breakout firing
# drops from 32 trades to 6 there specifically, per the mispricing above --
# most of what used to pass as a "breakout" at the open was really just
# ordinary open-of-day volume, not a real surge). Scanner names are lower-
# float and don't show the same clean session-boundary effect, so breakout
# firing there is roughly unchanged (29 -> 32 trades) and the combined
# improvement comes from elsewhere in the chain reacting to which bars
# breakout no longer claims first.
#
# Turned ON by default 2026-08-06, same day it was built -- the user's own
# call after seeing the numbers, ahead of this project's usual "let it sit
# a while first" convention (see the isolated-then-combined re-verification
# above; both runs agreed). Worth remembering if this one doesn't hold up:
# it shipped on a single 90-day window on two small symbol sets, not a long
# track record, and that trade-off was made consciously, not by default.
TIME_OF_DAY_VOLUME_BUCKET_MINUTES = int(os.getenv("TIME_OF_DAY_VOLUME_BUCKET_MINUTES", 30))

# breakout_recent_high (below) has always been a rolling max of CLOSES,
# not of HIGHS -- so the "20-bar high" breakout already fires off closes,
# never off a bare intrabar wick. What it does NOT do by default is use
# the real chart high (wicks included) as the level to clear -- a rolling
# max of closes sits at or below that real high, so the level being
# tested against is softer than "the 20-bar high" actually implies. This
# toggle swaps the level itself to a rolling max of HIGHS (the level a
# chart reader would actually draw) while still requiring the bar to
# CLOSE beyond it, not just wick above it -- i.e. the market has to
# accept the real breakout level, not fade a level that only looked hard
# because it was built from closes.
#
# OFF by default after backtesting both ways (2026-08-06, TSLA/NVDA/COIN/
# AMD/PLTR and the FBRX/... scanner universe, 90 days each): it fires
# fewer-but-stricter breakout trades as intended (megacap 32->21, scanner
# 29->20), but that didn't translate into a better bot on either universe.
#   megacap: return +7.6% -> +6.4%, win rate 60% -> 57%, profit factor
#            1.79 -> 1.66, max drawdown 1.7% -> 1.8% -- worse across the
#            board, because the trades that stopped being "breakout"
#            didn't just disappear, they mostly landed on gap_continuation
#            instead (9 trades/$54 -> 12 trades/$43) at lower quality.
#   scanner: return +1.3% -> +1.2%, win rate 49% -> 51%, profit factor
#            1.15 -> 1.16, max drawdown 1.8% -> 2.1% -- essentially flat,
#            not a clear win either way.
# Kept toggleable rather than deleted since "the real level should be
# wick-based" is still a reasonable hypothesis, just not one this data
# supports turning on.
USE_CLOSE_BEYOND_LEVEL_CONFIRMATION = os.getenv("USE_CLOSE_BEYOND_LEVEL_CONFIRMATION", "false").strip().lower() in ("1", "true", "yes")

# Breakout INVALIDATION exit -- see breakout_invalidated_at()'s docstring
# (right after breakout_at() below) for the full real-evidence case this
# is built on. Symmetric with the entry side: breakout_at() requires a
# CLOSE above breakout_recent_high (or breakout_recent_high_wick, if
# USE_CLOSE_BEYOND_LEVEL_CONFIRMATION is on) to get in; this exits the
# instant a LATER bar closes back BELOW that SAME level -- frozen at the
# moment of entry, never re-read from the rolling window as it drifts --
# instead of waiting for an unrelated regime signal (trend_following's
# death-cross or mean_reversion's RSI-overbought, whichever happens to
# be active right now, regardless of what actually opened the position)
# or the mandatory end-of-day flatten.
#
# Default OFF -- the mechanism itself (breakout_invalidated_at()) is real,
# well-tested, and NOT in question; only whether it should be ON BY
# DEFAULT is. History:
#   2026-08-12 (commit c774afb): enabled live (.env/trade.yml set to
#     "true", overriding this code default) on real, independently-
#     verified 90-day scanner-universe backtest evidence: avg breakout
#     loss shrank ~20% (-1.96% -> -1.57%), combined portfolio profit
#     factor 1.29 -> 1.35, max drawdown 1.6% -> 1.4%.
#   2026-08-23: reverted back to OFF (this code default was never
#     actually flipped to "true" -- only .env/trade.yml were, and those
#     are reverted alongside this comment) after a 180-day robustness
#     re-test with a per-symbol P&L breakdown found the ENTIRE claimed
#     90-day/180-day improvement was driven by ONE symbol, SMCI
#     (85-100%+ of the delta on both windows). Excluding SMCI, every
#     metric on the 180-day scanner backtest REVERSES: portfolio pnl
#     $125.53 (ON) vs $148.20 (OFF), profit factor 1.245 vs 1.279, max
#     drawdown 2.30% vs 1.91% -- the feature makes things WORSE once the
#     one outlier is removed. Separately, a scanner-selection audit found
#     SMCI appeared in the REAL live watchlist exactly ONCE in the last 5
#     weeks (2026-07-22, before live trading even started), so this
#     feature had no real expected benefit and plausible real harm on the
#     population the bot actually trades now. Ships toggleable (not
#     deleted) so a future backtest across a wider, less outlier-prone
#     symbol set can still give this a fair re-test.
USE_BREAKOUT_INVALIDATION_EXIT = os.getenv("USE_BREAKOUT_INVALIDATION_EXIT", "false").strip().lower() in ("1", "true", "yes")

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

# Gap QUALITY gate -- separate from GAP_MIN_PCT above, which only checks
# gap SIZE. Overnight/intraday return-decomposition research (Lou, Polk &
# Skouras, "A Tug of War: Overnight Versus Intraday Expected Returns";
# similarly Cooper, Cliff & Gulen on overnight/intraday return patterns)
# finds that gaps driven by thin, sentiment/attention-style flow tend to
# partially fade back during the day, while gaps accompanied by real
# volume are more likely to hold or extend. gap_continuation_at() as
# written has no way to tell those apart -- it only ever checks PRICE
# (gap size vs. prior close, then a break of the opening bar's high),
# never whether anyone was actually there trading it.
#
# Reuses the same time-of-day volume normalization
# USE_TIME_OF_DAY_VOLUME_NORM already built for breakout (see
# compute_time_of_day_avg_volume) rather than inventing a second one:
# gap_quality_confirmed_at() compares the gap day's OPENING bar volume
# against this symbol's own historical volume for that same
# minutes-since-open bucket. A flat trailing average would be the wrong
# comparison here for the same reason documented on that flag -- 9:30 is
# the single heaviest-volume bucket of the day, so a flat average
# understates what's "normal" right at the open and would make ordinary
# opens look like volume surges.
#
# Default OFF -- ships toggleable so the backtest gets the final word,
# same convention as every other filter in this file. gap_continuation
# is already this bot's least-tested strategy (single digits to low
# teens of trades per symbol in a 90-day window) -- see CLAUDE.md for the
# specific numbers this shipped with, and treat any one backtest run on
# top of an already-thin sample as a hypothesis test, not a settled
# result.
USE_GAP_QUALITY_FILTER = os.getenv("USE_GAP_QUALITY_FILTER", "false").strip().lower() in ("1", "true", "yes")
# How many multiples of the symbol's own normal same-time-of-day opening
# volume the gap day's opening bar must clear. 1.5x chosen as a middle
# ground between BREAKOUT_VOLUME_MULTIPLIER (2.0, a "new high" breakout
# where a bigger surge is expected) and VWAP_REVERSION_MIN_VOLUME_MULT
# (1.2, just "not on dead volume") -- a gap is already a large price
# event even at 1x normal volume, so it doesn't need as high a bar as a
# breakout does to be worth trusting, but should still show MORE than
# ordinary opening-bar volume, not just any positive reading.
GAP_QUALITY_VOLUME_MULT = float(os.getenv("GAP_QUALITY_VOLUME_MULT", 1.5))

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
# OFF by default as of 2026-07-28. ORB is the highest-volume strategy in
# the system and it does not pay for itself once stop slippage is modelled:
#
#   scanner-picked stocks: 183 trades, 37% wins, -$474.56
#   megacaps:               78 trades, 44% wins,   +$0.73  (noise)
#
# Turning it off improves BOTH universes, which is why this is a default
# change and not a per-universe tweak:
#   scanner: profit factor 0.95 -> 1.08, max drawdown 12.4% -> 8.8%
#   megacap: profit factor 1.37 -> 1.52, max drawdown  3.0% -> 2.1%
# It also frees capital for breakout, which improves from $166 to $233.
#
# It looked flat rather than bad for a long time because the backtester
# used to assume stops filled exactly at the stop price; ORB's losers are
# concentrated in fast-moving names where that assumption is worst.
USE_ORB = os.getenv("USE_ORB", "false").strip().lower() in ("1", "true", "yes")
ORB_MINUTES = int(os.getenv("ORB_MINUTES", 15))

# VWAP mean-reversion -- BUY when price is stretched below session VWAP
# and starting to turn back up.
USE_VWAP_REVERSION = os.getenv("USE_VWAP_REVERSION", "true").strip().lower() in ("1", "true", "yes")
VWAP_REVERSION_PCT = float(os.getenv("VWAP_REVERSION_PCT", 1.5))
# Whether vwap_reversion_at additionally requires the CURRENT bar to have
# already closed above the PRIOR bar's close ("already turning back up")
# before firing, on top of just being stretched below VWAP.
#
# DEFAULT FLIPPED TO FALSE (evidence says REMOVE the requirement) after a
# rigorous sensitivity sweep -- 90d AND 180d, BOTH universes, with a
# per-symbol breakdown specifically to rule out one outlier symbol driving
# the result (this project's own recent history has burned it on exactly
# that mistake at least 3 times, one still live at the time of this
# writing -- see USE_MULTI_TIMEFRAME_FILTER's comment in trading_bot.py).
# This one wasn't outlier-driven: 7-8 of the 12 scanner symbols improved
# individually, not one name carrying the whole average. Every metric
# tested improved, broadly, by removing the requirement:
#
#   90d megacap:  82->117 trades, PF 1.57->1.43 (down slightly), return +4.75%->+5.79%
#   90d scanner: 102->149 trades, PF 0.97->1.20,                  return -0.22%->+2.49%
#   180d megacap:      PF 1.50->1.51 (tie),                       return +7.85%->+12.24%
#   180d scanner:      PF 1.17->1.37,                              return +2.25%->+7.75%
#
# The one metric that moved the "wrong" way (90d megacap PF, 1.57->1.43)
# is a real tradeoff, not a red flag: nearly 1.5x the trade count (82->117)
# at a still-solidly-profitable profit factor, on a strategy/universe pair
# that was already the smallest sample here. Every OTHER cell -- both
# windows, both universes, return, trade count, and 3 of 4 profit-factor
# readings -- points the same direction, which is why this ships as the
# new default rather than staying a per-universe or per-window tweak.
#
# Why "already turning up" was hurting more than helping: waiting for
# close_now > close_prev doesn't distinguish a genuine bounce off support
# from a stock still falling that merely ticked up for one bar inside a
# larger downswing -- ADX (VWAP_REVERSION_MAX_ADX) is already the real
# trend/knife-catching guard here (see vwap_reversion_at's docstring for
# the 2026-07-27 postmortem that added it), so the turn-up check was
# mostly just adding a one-bar entry LAG on top of a filter that already
# does the actual safety job: fewer, later entries on the same setups,
# not meaningfully safer ones.
#
# Old stricter behavior (require the turn-up) is still fully available --
# set this true to restore it, same toggle-for-symmetry convention as
# every other filter in this file.
USE_VWAP_REVERSION_TURN_UP_CONFIRMATION = os.getenv(
    "USE_VWAP_REVERSION_TURN_UP_CONFIRMATION", "false").strip().lower() in ("1", "true", "yes")
# Refuse VWAP reversion entries above this ADX -- an extreme-trend guard,
# NOT the fix for the 2026-07-27 losses.
#
# The tempting story that day was "every loser was entered at ADX >= 25,
# so gate on 25". A 90-day backtest killed that idea: at 25 the strategy
# drops from 56 trades / 66% wins / +$76.61 to 14 trades / 50% / -$12.54,
# and total return halves (+11.1% -> +5.9%). High ADX is just as common in
# this strategy's WINNERS. The one-day correlation was real and the
# conclusion drawn from it was wrong -- the actual cause was volatility
# (see MIN_STOP_TO_ATR_RATIO), not trend strength.
#
# 50 is set as a cheap safety rail: backtest-neutral (+10.9% vs +11.1%,
# and vwap_reversion is fractionally BETTER at +$81.60), while still
# refusing genuinely absurd cases. Deliberately not tighter, because the
# evidence says tighter costs money.
VWAP_REVERSION_MAX_ADX = float(os.getenv("VWAP_REVERSION_MAX_ADX", 50))
# Requires volume at least this many times the recent average on the
# entry bar -- vwap_reversion was the only enabled strategy with no
# volume check at all. Added 2026-07-31, SCANNER PICKS ONLY (see
# vwap_reversion_volume_confirms below for why this can't just live
# inside vwap_reversion_at itself).
#
# First tested vwap_reversion in ISOLATION (as the only active
# strategy): looked like a clean win everywhere, roughly doubling its
# own profit factor on liquid names (1.74 -> 3.53). Tested again in the
# FULL priority chain, same 90 days, and the isolated test turned out to
# be misleading: tightening vwap_reversion's entries means fewer of its
# signals fire, so more bars fall through to trend_following (next in
# priority) -- which is weaker on megacaps, dragging the combined result
# down even though vwap_reversion itself was better. Same-moment A/B on
# the full system: megacap PF 1.60 -> 1.56 (worse), scanner PF 1.18 ->
# 1.30 (better). Same split shape as the multi-timeframe filter, same
# fix: scanner picks only, S&P 500 names exempt.
VWAP_REVERSION_MIN_VOLUME_MULT = float(os.getenv("VWAP_REVERSION_MIN_VOLUME_MULT", 1.2))

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

# Raised from 5 to 15 on 2026-07-31 after a 90-day backtest: fewer,
# higher-quality decision points beats reacting to every 5-minute wiggle.
# Improved BOTH universes together (rare enough to be worth noting, most
# tuning changes trade one off against the other):
#   megacap: win rate 55% -> 61%, profit factor 1.60 -> 1.87, drawdown 2.0% -> 1.6%
#   scanner: win rate 46% -> 53%, profit factor 1.18 -> 1.24, drawdown 5.7% -> 3.0% (nearly halved)
# Verified before shipping that nothing downstream assumes 5-minute bars:
# the entry blackout window and EOD flatten/stop-new-entries timing are
# all wall-clock based (Alpaca's real clock, not bar boundaries), so they
# needed no changes. The one real interaction: MIN_STOP_TO_ATR_RATIO's
# guard gets meaningfully stricter at a coarser timeframe (15-min ATR
# runs ~1.8x the 5-min value), which likely explains PART of the
# improvement on volatile scanner picks -- filtering out more marginal
# trades, not a side effect fighting the result. See CHECK_INTERVAL_MINUTES
# in trading_bot.py for the paired change this needs.
BAR_MINUTES = int(os.getenv("BAR_MINUTES", 15))
TRADE_AMOUNT_USD = float(os.getenv("TRADE_AMOUNT_USD", 500))

# Volatility-scaled sizing: reduces the FLAT TRADE_AMOUNT_USD position to
# VOLATILITY_SCALED_REDUCED_USD for the HIGH realized-vol tercile only
# (see high_vol_tercile in add_indicators() and the second-regime-axis
# comment above) -- low/mid tercile trades are untouched, still exactly
# TRADE_AMOUNT_USD. Applied externally in trading_bot.py's
# place_buy_order() and backtest.py's simulate(), not here -- the same
# "strategy.py computes, callers decide sizing" split TRADE_AMOUNT_USD /
# USE_RISK_BASED_SIZING above already use (position sizing has never
# lived inside this file's pure decision functions).
#
# MUST be a fixed dollar number, never a percentage of equity or of
# TRADE_AMOUNT_USD. See the 2026-08-05 CLAUDE.md entry: a %-of-equity
# sizing cap (MAX_POSITION_PCT_OF_EQUITY) that looked fine in backtest
# metrics -- win rate, profit factor, and drawdown-as-%-of-equity all
# genuinely supported it -- produced real $15,700-19,900 positions on a
# ~$99,645 account instead of the intended $500, because none of those
# metrics surface the ABSOLUTE DOLLAR size of a single position.
# VOLATILITY_SCALED_REDUCED_USD is deliberately just another flat-dollar
# figure like TRADE_AMOUNT_USD itself, not a multiplier or fraction of
# it, so it cannot scale with account size at all. The guard right below
# additionally makes it impossible for this to exceed TRADE_AMOUNT_USD
# even by misconfiguration -- this lever exists to shrink the high-vol
# bucket's size, never to grow anything past the existing baseline.
USE_VOLATILITY_SCALED_SIZING = os.getenv("USE_VOLATILITY_SCALED_SIZING", "false").strip().lower() in ("1", "true", "yes")
# $350 -- a 30% cut from the $500 baseline. Walked through explicitly:
# a HIGH-vol-tercile trade buys $350 of stock instead of $500 (e.g. 7
# shares instead of 10 at a $50 entry) -- fewer shares, a smaller
# absolute dollar loss if the stop is hit, nothing about account equity
# enters this number anywhere. 30% is a deliberately moderate first cut
# (not the "wipe it out" instinct a bigger number might invite): the
# high tercile is exactly where STOP_SLIPPAGE_ATR_FRACTION-modeled stop
# slippage bites hardest (see backtest.py), so this reduction stacks
# with the existing ATR-based noise guard (MIN_STOP_TO_ATR_RATIO)
# instead of trying to replace it.
#
# Known, MEASURED side effect on the megacap universe (90-day backtest,
# 2026-08-11, TSLA/NVDA/COIN/AMD/PLTR): int(350 // price) rounds to 0
# shares whenever price > $350, which AMD ($397-495 in this window) and
# TSLA ($326-422) cross often -- 17 of 45 high-vol-tercile trades across
# those two symbols were skipped entirely, not just downsized (`qty < 1`
# in place_buy_order/simulate() already treats that as "no trade", same
# as it always has for TRADE_AMOUNT_USD itself on any >$500 stock -- this
# candidate just lowers that same rounding floor from $500 to $350, so it
# now bites on the $350-500 price band too). This is still SAFE by this
# candidate's one hard rule (0 exposure is <= $500, never >), and
# arguably the most conservative possible response to a high-vol regime
# -- but it means "reduced size" becomes "skipped trade" for higher-
# priced names, not a smooth downsize, and the megacap backtest numbers
# below reflect that mix, not a pure size-only comparison. The scanner
# universe (all sub-$100 names here) never hit this floor -- see
# CLAUDE.md for the clean, composition-identical read of the mechanism.
VOLATILITY_SCALED_REDUCED_USD = float(os.getenv("VOLATILITY_SCALED_REDUCED_USD", 350))
if VOLATILITY_SCALED_REDUCED_USD > TRADE_AMOUNT_USD:
    # Fail loudly and immediately at import time rather than silently
    # sizing UP on the high-vol tercile -- exactly the shape of mistake
    # the 2026-08-05 incident (comment above) cost real money on, just
    # via a different constant. This candidate's one job is to never be
    # able to repeat that, even under a bad env-var edit.
    raise ValueError(
        f"VOLATILITY_SCALED_REDUCED_USD (${VOLATILITY_SCALED_REDUCED_USD:.0f}) must not exceed "
        f"TRADE_AMOUNT_USD (${TRADE_AMOUNT_USD:.0f}) -- this lever only ever REDUCES size on the "
        f"high-vol tercile, it must never raise it above the existing flat-sizing baseline."
    )

# Capped-boost sizing: the SAFE alternative to an earlier, REJECTED idea
# from the same conversation as the 2026-08-05 incident cited in
# VOLATILITY_SCALED_REDUCED_USD's comment above -- staking up to ~$90k on
# a "high-conviction" trade based on a vague, unvalidated confidence
# read. That idea was declined for the exact same underlying reason the
# %-of-equity sizing mode was: this bot has no real confidence score, and
# a lever that's allowed to scale position size off anything other than
# a flat dollar figure has already, once, for real, on 2026-08-05,
# produced $15,700-19,900 single positions on a ~$99,645 account instead
# of the intended $500 -- from a mode whose backtest metrics (win rate,
# profit factor, drawdown-as-%-of-equity) all looked fine, because none
# of those metrics surface the ABSOLUTE DOLLAR size of a single position.
# What follows is the approved, structurally-bounded alternative: a
# strategy with REAL evidence of a meaningfully better win rate trades a
# bit bigger (one fixed, capped bump), everything else stays at
# TRADE_AMOUNT_USD. Applied externally in trading_bot.py's
# place_buy_order()/estimate_new_position_risk_usd() and backtest.py's
# simulate() -- same "strategy.py computes, callers decide sizing" split
# TRADE_AMOUNT_USD / VOLATILITY_SCALED_REDUCED_USD / USE_RISK_BASED_SIZING
# above already use.
USE_CONVICTION_SIZING = os.getenv("USE_CONVICTION_SIZING", "false").strip().lower() in ("1", "true", "yes")

# Comma-separated reason_key strings (e.g. "trend_following,rvol_spike")
# naming which strategies currently have real evidence of a meaningfully
# better win rate and therefore qualify for CONVICTION_BOOST_USD below.
# Parsed once at import time like every other env-driven constant in
# this file: split on comma, strip whitespace, drop empty entries, build
# a set (membership-tested, not order-sensitive).
#
# DEFAULT IS DELIBERATELY EMPTY. This is not a stub left for later --
# it's the honest, currently-correct state of the evidence, and shipping
# it any other way would be shipping an opinion the data doesn't support.
# Per-strategy win rates were checked across THREE independent sources:
# two 90-day backtests (megacap: TSLA/NVDA/COIN/AMD/PLTR; scanner:
# FBRX/VEEE/PN/TRAX/QBTS/SMCI/SAFT/RNG/INHD/FCUV/ATKR/SRAD) and real
# reconstructed Alpaca live-trade history. NO strategy showed a
# consistent, real edge across all three:
#   - trend_following looked like a standout on megacap (64% win) but was
#     the WORST performer on scanner (33% win, losing money) -- a result
#     that flatters one 5-symbol universe and craters on a differently
#     composed one isn't "real evidence" of an edge, it's overfitting to
#     megacap's specific names.
#   - vwap_reversion looked solid on BOTH backtests but was the worst
#     REAL-MONEY performer of all (23% win, -$200 actual) -- precisely
#     the backtest-vs-live gap this project's testing culture exists to
#     catch before it costs more than $200.
# So the qualifying list is empty, on purpose, until some strategy
# someday shows a real, consistent edge across all three sources at
# once. Adding a name here should cite fresh evidence, never be a
# default.
HIGH_CONVICTION_STRATEGIES = {
    s.strip() for s in os.getenv("HIGH_CONVICTION_STRATEGIES", "").split(",") if s.strip()
}

# Flat dollar figure, same rule as TRADE_AMOUNT_USD and
# VOLATILITY_SCALED_REDUCED_USD -- never a percentage of equity, never a
# multiplier applied to equity. $750 against the $500 baseline is a
# deliberately modest bump (1.5x), chosen to be worth having without
# inviting the "why not push it further" instinct that produced the
# rejected $90k idea in the first place.
CONVICTION_BOOST_USD = float(os.getenv("CONVICTION_BOOST_USD", 750))
if CONVICTION_BOOST_USD > TRADE_AMOUNT_USD * 2:
    # Hard, structural ceiling: this feature must NEVER be able to more
    # than double the normal flat trade size, regardless of env var
    # misconfiguration. This guard is the direct, structural answer to
    # why the earlier "$90k on a strong signal" idea (see this block's
    # opening comment) was rejected -- that idea had no ceiling at all,
    # tied position size to a vague confidence read, and scaled with
    # equity. The 2026-08-05 incident (VOLATILITY_SCALED_REDUCED_USD's
    # comment above) is the real-money proof of what an unbounded sizing
    # lever costs: $15,700-19,900 single positions instead of $500, from
    # a mode whose own backtest metrics looked fine right up until it
    # was run for real. A 2x cap here means even a badly misconfigured
    # CONVICTION_BOOST_USD can only ever produce a $1,000 position on a
    # $500 baseline -- never anywhere close to a five-figure one.
    raise ValueError(
        f"CONVICTION_BOOST_USD (${CONVICTION_BOOST_USD:.0f}) must not exceed 2x "
        f"TRADE_AMOUNT_USD (${TRADE_AMOUNT_USD:.0f} x 2 = ${TRADE_AMOUNT_USD * 2:.0f}) -- "
        f"this lever must never be able to more than double the normal flat trade size. "
        f"This is the same structural lesson the 2026-08-05 %-of-equity incident cost real "
        f"money to learn, and the direct reason the earlier $90k 'high-conviction' sizing "
        f"idea was rejected instead of built: no sizing lever in this bot may be unbounded."
    )

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
# The stop must sit at least this many ATRs below the entry, or the trade
# is skipped entirely. See stop_is_wider_than_noise for the day this was
# learned the expensive way. 2.0 is the conventional "stop outside the
# noise" ratio; at STOP_LOSS_PCT=5 it refuses anything whose ATR exceeds
# 2.5% of price per bar, which excludes VEEE-class stocks while leaving
# ordinary movers (ATR 0.5-2.2%/bar) untouched. Set to 0 to disable.
MIN_STOP_TO_ATR_RATIO = float(os.getenv("MIN_STOP_TO_ATR_RATIO", 2.0))

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


def compute_vol_percentile(atr: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    """
    Each bar's ATR-as-%-of-price, ranked (0-1) against that SAME symbol's
    own trailing `lookback` bars -- see VOL_PERCENTILE_LOOKBACK's comment
    for why this is a second, ADX-independent regime axis.

    No-lookahead: pandas' rolling().rank(pct=True) at row i only ever
    considers rows [i-lookback+1, i], same backward-only guarantee as
    every other indicator in this file. NaN for the first `lookback - 1`
    bars of a symbol's history (not enough trailing data yet to rank
    against) -- callers comparing this against a threshold (e.g.
    HIGH_VOL_TERCILE_CUTOFF) get False for those bars for free, since
    NaN >= threshold is False in pandas/numpy, not NaN.
    """
    atr_pct = atr / close.replace(0, np.nan) * 100
    return atr_pct.rolling(lookback, min_periods=lookback).rank(pct=True)


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


def compute_time_of_day_avg_volume(df: pd.DataFrame, session_date: pd.Series,
                                    minutes_since_open: pd.Series,
                                    bucket_minutes: int) -> pd.Series:
    """
    For each bar, the historical average volume of OTHER sessions' bars
    that fall in the same minutes-since-open bucket -- e.g. a 9:30-10:00am
    bar is judged against prior sessions' 9:30-10:00am bars, not against a
    flat trailing average spanning every time of day. See
    USE_TIME_OF_DAY_VOLUME_NORM above for why that distinction matters.

    Lookahead safety: volume is first averaged WITHIN each (session,
    bucket) pair down to one number, then an EXPANDING mean of that
    per-session number is taken across sessions, shifted by one session.
    That shift drops the current session's own bucket entirely -- not just
    the current bar -- out of its own historical average, so a later bar
    in the same bucket the same session can never leak into an earlier
    bar's comparison (or vice versa). Expanding rather than a fixed
    trailing window because a single backtest run's history is short
    enough that a fixed N-day lookback would spend much of the window
    still warming up.
    """
    bucket = np.floor(minutes_since_open / bucket_minutes)
    key = pd.DataFrame({
        "session_date": session_date.values,
        "bucket": bucket.values,
        "volume": df["volume"].values,
    })
    per_session_bucket = (
        key.groupby(["bucket", "session_date"], sort=True)["volume"]
        .mean()
        .reset_index()
    )
    per_session_bucket["hist_avg"] = (
        per_session_bucket.groupby("bucket")["volume"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    lookup = per_session_bucket.set_index(["bucket", "session_date"])["hist_avg"]
    bar_key = pd.MultiIndex.from_arrays([bucket.values, session_date.values])
    return pd.Series(lookup.reindex(bar_key).values, index=df.index)


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

    # Second regime axis (see VOL_PERCENTILE_LOOKBACK above): each bar's
    # trailing realized-vol percentile, relative to this SAME symbol's
    # own history only. Always computed (cheap, and matches how every
    # other optional-strategy column here is unconditional) so flipping
    # USE_VOLATILITY_SCALED_SIZING on in trading_bot.py/backtest.py never
    # requires touching add_indicators.
    df["vol_percentile"] = compute_vol_percentile(df["atr"], df["close"], VOL_PERCENTILE_LOOKBACK)
    df["high_vol_tercile"] = df["vol_percentile"] >= HIGH_VOL_TERCILE_CUTOFF

    # Sticky ADX regime for USE_ADX_HYSTERESIS (see that constant's
    # comment for why): a proper Schmitt trigger, not just "ffill the
    # gaps". ADX has to clear the UPPER band edge to commit to
    # trend_following or drop below the LOWER edge to commit to
    # mean_reversion; while it's inside the band it keeps whatever regime
    # was already active, which is the entire point (the plain
    # ADX_TREND_THRESHOLD compare below has no memory of that). Computed
    # as: mark only the bars that actually cross a band edge as
    # True/False, leave everything inside the band as NaN, then
    # forward-fill -- a vectorized way of carrying the last DECIDED
    # regime forward over every bar in between, with no per-bar Python
    # loop and no mutable state (this file precomputes once, per its
    # module docstring, rather than tracking state across calls).
    # Computed unconditionally, same as every other optional-strategy
    # column here, so USE_ADX_HYSTERESIS is a pure read-time switch.
    adx_upper = ADX_TREND_THRESHOLD + ADX_HYSTERESIS_BAND
    adx_lower = ADX_TREND_THRESHOLD - ADX_HYSTERESIS_BAND
    regime_crossing = pd.Series(np.nan, index=df.index)
    regime_crossing[df["adx"] >= adx_upper] = 1.0
    regime_crossing[df["adx"] <= adx_lower] = 0.0
    sticky_regime = regime_crossing.ffill()
    # Bars before the FIRST band crossing (warmup, or a symbol whose ADX
    # never clears the band early on) have no prior regime to stick with
    # yet -- fall back to the plain single-cutoff read for just those
    # rows. min_bars_needed and the NaN-ADX warmup check in
    # _decide_from_indicators already keep these particular rows from
    # ever being traded on, so this fallback is only there to give the
    # column a defined value everywhere, not to influence real decisions.
    sticky_regime = sticky_regime.fillna((df["adx"] >= ADX_TREND_THRESHOLD).astype(float))
    df["adx_trend_regime"] = sticky_regime.astype(bool)

    # Breakout: prior BREAKOUT_LOOKBACK bars' high/avg-volume, excluding
    # the current bar (shift(1) before the rolling window).
    df["breakout_recent_high"] = df["close"].shift(1).rolling(BREAKOUT_LOOKBACK).max()
    df["breakout_avg_volume"] = df["volume"].shift(1).rolling(BREAKOUT_LOOKBACK).mean()
    # Wick-based counterpart of breakout_recent_high, only consulted when
    # USE_CLOSE_BEYOND_LEVEL_CONFIRMATION is on -- see that flag's comment
    # above and breakout_at() below for why the two differ.
    df["breakout_recent_high_wick"] = df["high"].shift(1).rolling(BREAKOUT_LOOKBACK).max()

    # Smash Day setup/trigger, vectorized from the original bar-relative rule.
    df["smash_day_setup"] = df["close"].shift(1) < df["low"].shift(2)
    df["smash_day_trigger"] = df["high"] > df["high"].shift(1)

    # Relative volume vs. its own recent average, excluding the current bar.
    df["rvol_avg_volume"] = df["volume"].shift(1).rolling(RVOL_LOOKBACK).mean()

    session_date = _session_date(df)
    df["_session_date"] = session_date
    df["minutes_since_open"] = _minutes_since_open(df)
    df["vwap"] = compute_vwap(df, session_date)

    # Always computed (cheap, and matches how orb_high/vwap/etc. above are
    # computed regardless of their own USE_* toggle) so flipping
    # USE_TIME_OF_DAY_VOLUME_NORM on doesn't require touching add_indicators.
    df["breakout_tod_avg_volume"] = compute_time_of_day_avg_volume(
        df, session_date, df["minutes_since_open"], TIME_OF_DAY_VOLUME_BUCKET_MINUTES
    )

    # Gap-quality gate columns (USE_GAP_QUALITY_FILTER above) -- the gap
    # day's own opening-bar volume, and what's normal for THIS symbol at
    # THAT specific time-of-day bucket, broadcast across the whole
    # session the same way session_open_price/session_first_bar_high
    # already are below. Reuses breakout_tod_avg_volume's per-bucket
    # historical average (already lookahead-safe, see that column's
    # comment) rather than a second normalization pass -- the opening
    # bar's own minutes_since_open bucket is bucket 0, so its
    # breakout_tod_avg_volume value already IS "this symbol's normal
    # opening-bar volume." Always computed, same reasoning as
    # breakout_tod_avg_volume: keeps flipping the toggle a pure read-time
    # switch rather than requiring changes here.
    df["session_first_bar_volume"] = df.groupby(session_date)["volume"].transform("first")
    df["session_first_bar_tod_avg_volume"] = df.groupby(session_date)["breakout_tod_avg_volume"].transform("first")

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

    The comparison is always against CLOSE, never the bar's HIGH, so
    this never fired off a bare intrabar wick to begin with. What
    USE_CLOSE_BEYOND_LEVEL_CONFIRMATION changes is which LEVEL gets
    cleared: off (default), the level is a rolling max of CLOSES, which
    sits at or below the real chart high; on, it's a rolling max of
    HIGHS -- the actual N-bar high, wicks included -- so a close above
    it means the market genuinely accepted that price, not just cleared
    a softer close-only approximation of it.
    """
    if i < 1:
        return False
    if USE_CLOSE_BEYOND_LEVEL_CONFIRMATION:
        recent_high = df["breakout_recent_high_wick"].iat[i]
        prev_recent_high = df["breakout_recent_high_wick"].iat[i - 1]
    else:
        recent_high = df["breakout_recent_high"].iat[i]
        prev_recent_high = df["breakout_recent_high"].iat[i - 1]
    # See USE_TIME_OF_DAY_VOLUME_NORM above: the flat trailing average is a
    # stricter bar in the morning and a looser one at lunch purely because
    # of the intraday U-shaped volume pattern, not because of anything
    # about the breakout itself.
    if USE_TIME_OF_DAY_VOLUME_NORM:
        avg_volume = df["breakout_tod_avg_volume"].iat[i]
    else:
        avg_volume = df["breakout_avg_volume"].iat[i]
    if pd.isna(recent_high) or pd.isna(avg_volume) or avg_volume == 0 or pd.isna(prev_recent_high):
        return False

    close_now = df["close"].iat[i]
    close_prev = df["close"].iat[i - 1]
    broke_out_last_bar = close_prev > prev_recent_high
    still_pushing = close_now > recent_high and close_now > close_prev
    volume_confirmed = df["volume"].iat[i] > avg_volume * BREAKOUT_VOLUME_MULTIPLIER
    return broke_out_last_bar and still_pushing and volume_confirmed


def breakout_invalidated_at(df: pd.DataFrame, i: int, invalidation_level: float | None) -> bool:
    """
    True if `invalidation_level` -- the SAME breakout_recent_high (or
    breakout_recent_high_wick, if USE_CLOSE_BEYOND_LEVEL_CONFIRMATION was
    on) level that justified this position's original breakout entry,
    frozen at the moment of entry, never re-read from a later, drifted
    rolling window -- is a real number AND this bar's close has dropped
    back BELOW it.

    This is breakout_at()'s own entry test, run in reverse against the
    same level: entry requires a CLOSE above the recent-high; this is
    "the thesis that got us in no longer holds," the exit that mirrors
    it. Without this, EVERY open position -- no matter which strategy
    opened it -- only ever exits via (a) its bracket stop-loss/take-
    profit filling on their own, (b) whichever regime strategy happens
    to be active RIGHT NOW (decide_signal_at() only ever reflects the
    CURRENT regime, not what originally justified the entry), or (c) the
    mandatory end-of-day flatten. There has never been a "the original
    breakout thesis specifically failed" exit.

    That gap is not a theory -- it's reconstructed from real Alpaca order
    history: ALL 12 real winning breakout trades exited via a plain
    MARKET order, never the bracket's own take-profit LIMIT leg, meaning
    every real win got cut short before reaching its 10% target by
    something unrelated to the breakout thesis. Only 2 of 9 real losses
    hit the actual stop-loss; the other 7 also just got market-sold. In a
    90-day scanner-universe backtest, "end-of-day flatten" was the exit
    reason for 28 of 37 breakout trades (76%) -- take-profit fired only 3
    times. Net effect: real breakout trades won MORE often (57%) than
    backtests predicted, but still lost money overall (-$182.99 across 21
    real trades), because winners averaged only +2% (cut short) while
    losers averaged -3.3% (never reliably capped). Giving breakout
    positions their own thesis-specific exit is meant to close that gap.

    Fails CLOSED like every other guard in this file: a missing level
    (None or NaN -- e.g. a non-breakout position, or a breakout position
    opened before this feature existed / before this field was tracked)
    means this never fires. It never guesses at a level it wasn't given.
    """
    if invalidation_level is None or pd.isna(invalidation_level):
        return False
    return bool(df["close"].iat[i] < invalidation_level)


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


def gap_quality_confirmed_at(df: pd.DataFrame, i: int) -> bool:
    """
    True if this gap day's OPENING bar traded on volume meaningfully
    above what's normal for this symbol at that same time-of-day bucket
    (see USE_GAP_QUALITY_FILTER above for the research this is drawing
    on: gaps backed by real volume are more likely to hold, thin/
    sentiment-driven gaps tend to fade). Only meaningful for a row that
    gap_continuation_at has already confirmed IS a gap day -- this
    function doesn't re-check gap size, only volume quality.

    Fails CLOSED like every other volume/noise guard in this file: not
    enough prior-session history yet to know what "normal" volume is for
    this bucket (early in a backtest window, or a low-history symbol)
    means no confirmation, not a free pass.
    """
    opening_volume = df["session_first_bar_volume"].iat[i]
    normal_volume = df["session_first_bar_tod_avg_volume"].iat[i]
    if pd.isna(opening_volume) or pd.isna(normal_volume) or normal_volume <= 0:
        return False
    return opening_volume >= normal_volume * GAP_QUALITY_VOLUME_MULT


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

    Both checks above are pure PRICE checks -- they can't tell a gap
    backed by real trading interest from a thin, low-volume gap that's
    statistically more likely to fade (see USE_GAP_QUALITY_FILTER).
    When that toggle is on, a qualifying price-based gap must ALSO clear
    gap_quality_confirmed_at's opening-volume check to fire.
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
    if not (close_now > first_bar_high and close_prev <= first_bar_high):
        return False

    if USE_GAP_QUALITY_FILTER and not gap_quality_confirmed_at(df, i):
        return False

    return True


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
    """
    BUY when price is stretched VWAP_REVERSION_PCT% below session VWAP --
    but ONLY when the stock isn't in a strong trend.

    That trend condition is the whole difference between mean reversion
    and catching a falling knife, and its absence cost real money on
    2026-07-27: this strategy took 8 of the day's 10 trades, lost 7 of
    them, and accounted for essentially the entire -$171 day while the
    other two strategies were both slightly green.

    Every one of those losses was entered while ADX said "strong trend":
    VEEE at ADX 44.6 / 40.7 / 30.8, TRAX at 25.0 twice, NVDA at 53.6 with
    RSI 17.7. "Stretched below VWAP" is a reversion signal in a RANGE; in
    a downtrend it's just a pause on the way down, and the bot bought
    every one of those pauses.

    ADX was already computed and already had a threshold constant -- it
    was simply never consulted here, because this strategy runs at
    priority 1, ahead of the regime switch that does check it.

    USE_VWAP_REVERSION_TURN_UP_CONFIRMATION (default FALSE): whether an
    entry ALSO requires the current bar to have already closed above the
    prior bar's close ("already turning back up") on top of being
    stretched below VWAP. This used to be a hardcoded, non-toggleable
    requirement. A 90d-AND-180d, both-universe sensitivity sweep with a
    per-symbol breakdown (see that constant's comment above) found
    removing it improves every metric tested, broadly across 7-8 of 12
    scanner symbols rather than one outlier -- ADX already does the real
    knife-catching-vs-reversion job here, so the turn-up check was mostly
    adding a one-bar entry lag rather than meaningfully more safety.
    """
    vwap = df["vwap"].iat[i]
    if pd.isna(vwap) or vwap == 0 or i < 1:
        return False
    adx = df["adx"].iat[i]
    # Unknown ADX fails CLOSED: without a trend reading there's no way to
    # tell reversion from knife-catching, and the evidence says guessing
    # wrong is expensive.
    if pd.isna(adx) or adx >= VWAP_REVERSION_MAX_ADX:
        return False
    close_now = df["close"].iat[i]
    stretched_below = close_now < vwap * (1 - VWAP_REVERSION_PCT / 100)
    if not stretched_below:
        return False
    if USE_VWAP_REVERSION_TURN_UP_CONFIRMATION:
        close_prev = df["close"].iat[i - 1]
        turning_up = close_now > close_prev
        return bool(turning_up)
    return True


def vwap_reversion_volume_confirms(df: pd.DataFrame, i: int) -> bool:
    """
    Standalone volume check for a vwap_reversion entry -- deliberately
    NOT folded into vwap_reversion_at itself. It only applies to scanner-
    picked (non-S&P-500) symbols (see VWAP_REVERSION_MIN_VOLUME_MULT's
    comment for the full evidence), and "is this symbol an S&P 500
    member" is an operational/live-data concept strategy.py has no
    access to by design -- it stays a pure, network-free decision file.
    So this check lives here as a separate, callable function, and
    trading_bot.py/backtest.py apply it externally (same architecture as
    the multi-timeframe filter) only for the symbols it's proven to help.

    Fails CLOSED like every other volume/noise guard in this file: no
    volume reading means no confirmation, not a free pass.
    """
    avg_volume = df["rvol_avg_volume"].iat[i]
    volume_now = df["volume"].iat[i]
    if pd.isna(avg_volume) or avg_volume == 0:
        return False
    return bool(volume_now >= avg_volume * VWAP_REVERSION_MIN_VOLUME_MULT)


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
        return "BUY", "vwap_reversion", (
            "stretched below VWAP and turning back up" if USE_VWAP_REVERSION_TURN_UP_CONFIRMATION
            else "stretched below VWAP")

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

    if USE_ADX_HYSTERESIS:
        is_trend = bool(df["adx_trend_regime"].iat[i])
    else:
        is_trend = current_adx >= ADX_TREND_THRESHOLD

    if is_trend:
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

def stop_is_wider_than_noise(entry_price: float, atr_value: float | None,
                              stop_price: float) -> bool:
    """
    True if the stop sits far enough below the entry to survive the
    stock's ordinary bar-to-bar movement.

    This is the real lesson of 2026-07-27. The bot bought VEEE three
    times with a 5% stop while VEEE's ATR was 6.0-7.3% PER FIVE-MINUTE
    BAR -- the stop was inside a single bar's normal range, so it was
    certain to be hit by noise and certain to slip badly when it was.
    Those three trades lost 8.78%, 8.32% and 8.26% against a "5%" stop
    and made up $127 of the day's $172 loss.

    The day's results line up almost perfectly with this one number:

        VEEE  ATR 6.0-7.3%/bar -> -8.78%, -8.32%, -8.26%
        TRAX  ATR 2.21%/bar    -> -4.21%, -4.16%, -3.28%
        NVDA  ATR 0.50%/bar    -> -1.19%
        QBTS  ATR 1.37%/bar    -> +0.47%
        COIN  ATR 0.63%/bar    -> +1.37%

    A stop tighter than the noise isn't risk management, it's a coin flip
    with a fee. The right response is to not trade the stock at all --
    widening the stop instead would just turn a 5% risk into a 15% one.

    Unlike a signal filter this protects EVERY strategy, and it's a
    property of the instrument rather than of one day's tape, which is
    why it generalizes where the ADX theory didn't.
    """
    if atr_value is None or pd.isna(atr_value) or atr_value <= 0:
        return True  # no ATR reading -- don't block on missing data
    stop_distance = entry_price - stop_price
    if stop_distance <= 0:
        return False
    return stop_distance >= MIN_STOP_TO_ATR_RATIO * atr_value


def compute_daily_trend_map(daily_bars: pd.DataFrame) -> dict:
    """
    Multi-timeframe confirmation building block: given DAILY bars (one
    row per trading day, NOT the intraday BAR_MINUTES bars everything
    else here works with), returns a {date: bool} map of whether that
    date's entries should be considered "trend-confirmed" -- meaning the
    PRIOR fully-closed day's daily EMA_fast > EMA_slow (an uptrend).

    Shifted by exactly one day on purpose: today's own daily bar is still
    forming while the market is open, so using it to gate today's
    intraday entries would be lookahead -- gating a 10am decision on
    data that only exists at 4pm the same day. Only a day that has
    already fully closed counts.

    Added 2026-07-31 after a 90-day backtest (see CLAUDE.md) found this
    filter is a clear win specifically for the kind of volatile, low-cap
    stocks the live scanner picks -- return +6.5% -> +9.8%, profit
    factor 1.10 -> 1.27, max drawdown 8.9% -> 4.6% over the same period.
    On liquid S&P 500-type names it cut trade count in half for a lower
    total dollar return (each trade taken was higher quality, just far
    fewer of them), so callers apply this ONLY to non-S&P-500 symbols --
    see trading_bot.py's USE_MULTI_TIMEFRAME_FILTER / fetch_sp500_symbols
    for how that scoping is done. This function itself has no opinion on
    which symbols it should run for -- it's a pure date -> bool mapping.

    A date with no prior-day data (warmup, or a data gap) maps to False,
    not missing -- callers should treat "unknown" and "downtrend" the
    same way (both refuse new entries), matching the fail-closed pattern
    already used by stop_is_wider_than_noise.
    """
    if daily_bars is None or daily_bars.empty:
        return {}
    df = daily_bars.copy()
    df["ema_fast"] = df["close"].ewm(span=FAST_MA, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=SLOW_MA, adjust=False).mean()
    df["trend_up"] = df["ema_fast"] > df["ema_slow"]
    df["date"] = _session_date(df)
    # The trend value ASSOCIATED WITH a given date is the trend as of the
    # PRIOR row -- i.e. what was already known and fully closed before
    # that date's own session began.
    df["trend_up_prior_day"] = df["trend_up"].shift(1)
    return dict(zip(df["date"], df["trend_up_prior_day"].fillna(False)))


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
