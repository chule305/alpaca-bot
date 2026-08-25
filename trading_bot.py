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
import io
import re
import time
import json
import argparse
import logging
import urllib.request
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, GetOrdersRequest, TakeProfitRequest, StopLossRequest,
    GetOrderByIdRequest, ReplaceOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus, OrderType, OrderStatus
from alpaca.common.enums import Sort
from alpaca.data.historical import StockHistoricalDataClient, ScreenerClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import (
    StockBarsRequest, MostActivesRequest, MarketMoversRequest, NewsRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import MostActivesBy, MarketType

from trade_recorder import record_trade, extract_context, extract_strategy
from strategy import (
    MIN_STOP_TO_ATR_RATIO, stop_is_wider_than_noise, compute_daily_trend_map,
    vwap_reversion_volume_confirms,
)
from strategy import (
    add_indicators, decide_signal_at, compute_stop_and_target, compute_position_size,
    breakout_invalidated_at,
    BAR_MINUTES, TRADE_AMOUNT_USD, FLATTEN_BEFORE_CLOSE, FLATTEN_MINUTES_BEFORE_CLOSE,
    STOP_NEW_ENTRIES_MINUTES_BEFORE_CLOSE, ENTRY_BLACKOUT_START_MINUTES, ENTRY_BLACKOUT_END_MINUTES,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, USE_ATR_STOPS, ATR_STOP_MULTIPLIER, ATR_TARGET_MULTIPLIER,
    USE_RISK_BASED_SIZING, RISK_PER_TRADE_PCT, MAX_POSITION_PCT_OF_EQUITY,
    USE_VOLATILITY_SCALED_SIZING, VOLATILITY_SCALED_REDUCED_USD,
    USE_CONVICTION_SIZING, HIGH_CONVICTION_STRATEGIES, CONVICTION_BOOST_USD,
    FAST_MA, SLOW_MA, RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    ADX_PERIOD, ADX_TREND_THRESHOLD,
    USE_BREAKOUT, USE_SMASH_DAY_PATTERN, USE_GAP_PATTERN, USE_ROSS_HOOK, USE_ORB,
    USE_VWAP_REVERSION, USE_RVOL_SPIKE,
    USE_BREAKOUT_INVALIDATION_EXIT, USE_CLOSE_BEYOND_LEVEL_CONFIRMATION,
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
# Raised from 5 to 15 alongside strategy.BAR_MINUTES on 2026-07-31 --
# checking every 5 minutes against a bar that only closes every 15
# achieves nothing but wasted API calls two-thirds of the time; reaction
# speed is gated by the bar close either way, not by how often we look.
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", 15))

USE_SCANNER = os.getenv("USE_SCANNER", "true").strip().lower() in ("1", "true", "yes")
# Widened/faster defaults from the original 5 stocks refreshed once a day --
# batched bar-fetching (see get_recent_bars_batch) is what makes checking a
# much bigger watchlist every cycle cheap on API calls, so the scanner can
# now afford to look harder and more often.
#
# Raised 12 -> 18 on 2026-08-02 after the user asked about widening the
# bot's search for setups (more stocks -> more chances for the SAME
# already-validated strategies to find a qualifying entry). Two other
# ideas from that conversation were checked and rejected first:
#   - A parallel NASDAQ-100 backstop (mirroring SP500_MIN_WATCHLIST_SLOTS):
#     checked the real overlap between NASDAQ-100 and S&P 500 constituent
#     lists -- 88 of 101 NASDAQ-100 names (87%) are already S&P 500
#     members, so this would add real machinery (fetch/cache/exempt-from-
#     filters, mirroring the whole SP500 backstop) for just 13 genuinely
#     new symbols, one of which (MSTR) was already deliberately dropped
#     from this bot's default universe once before. Not worth building.
#   - Raising SCANNER_CANDIDATE_POOL: already maxed. Alpaca's screener
#     endpoints hard-cap "top" at 50 server-side (see that constant's
#     comment) -- there's no more raw candidate pool to widen.
# This increase is NOT backtest-validated the way strategy/filter changes
# normally are here: backtest.py takes a fixed symbol list as input and
# has no way to replay what the scanner would historically have picked
# day-by-day, so "does a bigger watchlist actually raise win rate" isn't
# a backtestable question with the current tooling -- only a structural
# argument (more coverage of an already-filtered, already-liquid pool,
# same strategies, same per-symbol risk caps). Kept the increase modest
# (+50%, not a jump to 30+) for that reason, and scaled
# SP500_MIN_WATCHLIST_SLOTS alongside it (4 -> 6) so the liquidity
# backstop stays roughly the same ~1/3 share of the list instead of being
# diluted by the larger total.
SCANNER_WATCHLIST_SIZE = int(os.getenv("SCANNER_WATCHLIST_SIZE", 18))
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
# After a position in a symbol closes, refuse to buy that symbol again for
# this many minutes.
#
# 2026-07-27 is the case for it: of 10 trades, 7 were rapid re-entries into
# just TWO falling stocks. VEEE was bought four separate times as it fell
# 17.79 -> 15.33 (once only FIVE minutes after being stopped out), and TRAX
# three times on the way from 43.42 to 40.54. Six of those seven repeats
# lost money; together they are most of the day's -$171.
#
# One bad read on a stock became four bad trades purely because nothing
# stopped the bot from immediately trying again. This is what turns a
# normal losing trade into a bleed.
SYMBOL_COOLDOWN_MINUTES = float(os.getenv("SYMBOL_COOLDOWN_MINUTES", 60))
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
#
# 2026-08-23 scanner robustness investigation (see CLAUDE.md): tightened
# 50 -> 20 after real money confirmed 50 was far too loose to matter.
# "Momentum mover" picks averaged -$2.17/trade in REAL trading vs.
# +$51.56/trade for S&P 500 backstop picks -- same code, same risk rules,
# only the selected stock differed. Of the 5 real losing "mover" trades
# traced against real intraday bars (RGEN, CHYM, ALMR, NN, ATRO), 2
# entered well past this filter's reach: ALMR after +28.1% from the open,
# NN at the exact peak of a +23.5% two-hour run -- comfortably under the
# old 50% cutoff but caught cleanly by 20%. The other 3 (RGEN +8.9%,
# CHYM +10.1%, ATRO +7.3%) are all under ANY sane extension threshold --
# those are what SCANNER_OPENING_BLACKOUT_MINUTES below catches instead
# (all 3 entered 30-35 minutes since open, squarely in that filter's
# window). 20% is chosen to comfortably block the confirmed 23-28% range
# while still leaving room for genuine, early-stage moves -- see the
# scanner-universe before/after backtest in CLAUDE.md for the watchlist-
# size impact of this change.
SCANNER_MAX_EXTENSION_PCT = float(os.getenv("SCANNER_MAX_EXTENSION_PCT", 20.0))

# 2026-08-23 scanner robustness investigation, continued: no blackout
# existed anywhere on the dangerous OPENING-spike window -- the pre-
# existing ENTRY_BLACKOUT_START/END_MINUTES (see strategy.py) targets the
# low-volume MIDDAY chop window (~1:00-2:00pm ET) and says nothing about
# the opening period, which is exactly where real losses cluster: the
# 30-90-minutes-since-open bucket was the single worst in real trading,
# net -$55.70 across 12 real trades, with losses running ~1.6x the size
# of wins even at a 50% win rate. Scoped to scanner picks only (same
# S&P-500-exempt mechanism as USE_MULTI_TIMEFRAME_FILTER/
# USE_VWAP_VOLUME_CONFIRMATION above) because the evidence is specifically
# about scanner-picked movers entering into a still-running opening
# spike, not about S&P 500 names, and because this is DIFFERENT FROM and
# ADDITIVE TO the lunch-window blackout, not a replacement for it.
#
# Default 45. Checked directly against the 5 real losing trades this
# whole investigation traces (RGEN, CHYM, ALMR, NN, ATRO): RGEN entered
# at 30.4 min since open, CHYM at 34.2, ATRO at 32.4 -- a 30-minute
# cutoff would slip through all three by just a few minutes, right as
# each opening spike was already rolling over (see ATRO's case
# specifically). 45 comfortably covers all three with room to spare.
# (ALMR/NN entered ~222 min in, well outside either blackout window --
# those two are what SCANNER_MAX_EXTENSION_PCT above is for instead.)
#
# NOT widened further to 90 despite real trading's own "30-90-min-since-
# open bucket is the single worst" framing -- backtest.py's entry-time
# proxy (see that file) was used to sanity-check window sizes on the
# scanner-universe backtest, and 90 min backtests WORSE than 45 at both
# 90 and 180 days (180d: blackout alone swings the universe from +$98.47
# to -$98.19 at 90 min, vs. only +$46.57 at 45 min -- see CLAUDE.md for
# the full window-size sweep). That damage is dominated by a couple of
# single-symbol/single-trade effects (one QBTS breakout entered at
# literally 0 minutes since open and hit take-profit for +$48.27 pre-fix;
# delaying it past a wide blackout turned it into a loss), and the
# window-size response isn't even monotonic across the sizes tested --
# a classic small-sample overfitting signature, not a reason to trust 90
# over 45. 45 is the size directly justified by the concrete real cases
# without extrapolating past what they actually show. Defaults ON given
# the underlying real-money evidence for SOME opening-window block is
# real, direct-price-bar-confirmed, not backtest-only.
USE_SCANNER_OPENING_BLACKOUT = os.getenv("USE_SCANNER_OPENING_BLACKOUT", "true").strip().lower() in ("1", "true", "yes")
SCANNER_OPENING_BLACKOUT_MINUTES = int(os.getenv("SCANNER_OPENING_BLACKOUT_MINUTES", 45))

# 2026-08-23 scanner robustness investigation, continued: no listing-age
# filter existed anywhere, so a genuinely recent, no-track-record listing
# could reach the live watchlist with zero screening as long as it moved
# enough today. Confirmed concretely in real trading: ALMR (traded
# 2026-08-11, real loss -$25.48) had only ~80 days of Alpaca daily bar
# history at the time; EROC (on the 2026-08-12 watchlist) had only ~43.
# Requires at least SCANNER_MIN_LISTING_AGE_DAYS trading days of daily bar
# history via the same daily-bars endpoint refresh_daily_trend_maps_if_needed
# already uses (see meets_min_listing_age below) -- no new API dependency.
# Defaults ON given ALMR/EROC are concrete, confirmed real cases this
# would have caught.
USE_SCANNER_MIN_LISTING_AGE = os.getenv("USE_SCANNER_MIN_LISTING_AGE", "true").strip().lower() in ("1", "true", "yes")
SCANNER_MIN_LISTING_AGE_DAYS = int(os.getenv("SCANNER_MIN_LISTING_AGE_DAYS", 100))

# 2026-07-29 finding (see CLAUDE.md): the scanner's "top movers today"
# picks backtest at profit factor 1.08 over 90 days; the same strategies
# on liquid megacaps backtest at 1.52. The movers scan is a momentum
# filter by construction -- it ranks by SIZE OF MOVE, which structurally
# excludes megacaps (they rarely move enough in a day to place in the
# top 50 gainers/losers) even though this system trades them better.
# Rather than replace the movers scan, this reserves a minimum number of
# watchlist slots for S&P 500 names specifically, so liquidity is always
# represented regardless of whether anything in the index happens to be
# a big mover today. USE_SP500_UNIVERSE off returns to the pre-existing
# movers-only behavior exactly.
USE_SP500_UNIVERSE = os.getenv("USE_SP500_UNIVERSE", "true").strip().lower() in ("1", "true", "yes")
# Raised 6->10 (of 18) on 2026-08-23, the same day entry-fill slippage
# was finally modeled honestly in backtest.py (see CLAUDE.md's "backtest.py
# was assuming free entries" entry). That fix, combined with everything
# else merged the same day, revealed something this constant's OLD value
# was quietly working against: a full-system backtest on a fresh,
# real-watchlist-derived symbol set (NIQ/CDNL/ONON/VREX/HZO/QNST/DOCS/
# CRSR/TEAM/APPS/BLMN/SEDG) came back profit factor 0.39 at BOTH 90 and
# 180 days once realistic slippage was included -- consistent, not a
# thin-window fluke, and NOT explained by any single toggle (isolating
# every individual change tested that same day -- USE_MULTI_TIMEFRAME_
# FILTER on/off, USE_VWAP_REVERSION_TURN_UP_CONFIRMATION on/off, every
# combination -- all stayed in the 0.27-0.39 PF range with slippage on;
# only zeroing slippage itself flipped the picture positive). This lines
# up exactly with the SAME day's real-money finding (see
# scan_for_volatile_stocks' own comment above and CLAUDE.md): "momentum
# mover" picks averaged -$2.17/trade in REAL trading even with every
# known bug excluded, vs. +$51.56/trade for S&P 500 backstop picks --
# same code, same risk rules, only the selected stock differing. Two
# independent methods (real fills, and backtest once it stopped assuming
# free entries) now agree: the S&P 500 backstop population is the only
# part of this bot with a demonstrated positive edge, and the scanner's
# momentum-mover population currently is not, even after the same day's
# 4 new scanner-quality filters (SCANNER_MAX_EXTENSION_PCT tightened,
# USE_SCANNER_OPENING_BLACKOUT, USE_SCANNER_MIN_LISTING_AGE, the
# liquidity-window fix). Raised to 10 rather than all the way to
# SCANNER_WATCHLIST_SIZE (abandoning momentum-mover picks entirely) --
# that would be a bigger, more fundamental change than today's evidence
# demands on its own, and the 4 new scanner filters deserve a real chance
# to prove themselves in live trading now that they exist, just with a
# smaller, more conservative share of the watchlist while that evidence
# accumulates. This specific lever isn't independently backtestable
# (backtest.py takes a fixed symbol list, it doesn't simulate the
# scanner's own daily selection process) -- the justification rests on
# the asymmetric evidence on each sub-population above, not a direct
# before/after backtest of this exact change.
SP500_MIN_WATCHLIST_SLOTS = int(os.getenv("SP500_MIN_WATCHLIST_SLOTS", 10))
# The constituent list barely changes intraday -- a rare few times a year
# -- so this is a slow-moving reference list, not a scan result. Cached
# in-process rather than to disk: within one --duration-minutes job the
# same process runs many scan cycles, so this avoids refetching on every
# one of them, and a fresh job refetching once every ~2.5h is cheap
# enough that on-disk persistence isn't worth the extra state file.
SP500_REFRESH_HOURS = float(os.getenv("SP500_REFRESH_HOURS", 24))
# A community-maintained, versioned CSV mirror of the S&P 500 constituent
# list, served over plain HTTPS with no auth/rate limit -- chosen over
# scraping Wikipedia's table (fragile to markup changes) or a paid data
# API (unnecessary for a list that changes a handful of times a year).
SP500_LIST_URL = os.getenv(
    "SP500_LIST_URL",
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
)

# Multi-timeframe confirmation, SCANNER PICKS ONLY: requires the prior
# day's daily EMA(9) > EMA(21) before an intraday entry on a non-S&P-500
# name. Reuses the exact same fetch_sp500_symbols() membership check
# built for the S&P 500 liquidity backstop, since that backstop puts both
# kinds of symbol on the same watchlist every day -- this can't be "on
# for the whole watchlist," it has to know which symbols to skip.
#
# DEFAULT FLIPPED BACK TO FALSE 2026-08-23. Originally shipped TRUE on
# 2026-07-31 based on a single 90-day backtest called "a clear win... on
# liquid S&P 500-type names it cut trade count in half for a LOWER total
# dollar return" but "an unambiguous win" for scanner picks specifically
# (return +6.5%->+9.8%, PF 1.10->1.27, max drawdown 8.9%->4.6%). That
# characterization does not survive a real re-test with the mandatory
# per-symbol outlier check this project now requires for every backtest
# claim (see the project-wide methodology note this task was shipped
# alongside). Four separate ON-vs-OFF comparisons were run, each with a
# FULL leave-one-symbol-out sweep (12 exclusions) to check whether the
# verdict depends on any single name:
#
#   1. Fixed 12-symbol "scanner" list (FBRX/VEEE/PN/TRAX/QBTS/SMCI/SAFT/
#      RNG/INHD/FCUV/ATKR/SRAD -- the list used all week for
#      comparability), 90 days: ON wins ($149 vs $39, PF 1.20 vs 1.03,
#      DD 3.42% vs 3.89%). NOT robust on independent re-verification --
#      excluding either of two symbols (PN, or FCUV) flips it to favor
#      OFF instead, and the FCUV case is a textbook small-sample artifact
#      (4 trades total, profit factor 35.73 -- near-zero losses inflating
#      PF absurdly). This result doesn't survive its own leave-one-out
#      check and shouldn't be trusted as evidence either way.
#   2. SAME fixed list, 180 days: OFF wins ($752 vs $457, PF 1.32 vs
#      1.36 -- roughly tied on quality per trade, OFF just takes ~1.7x
#      more of them). ALSO robust to any single exclusion, including
#      VEEE -- excluding VEEE still leaves OFF winning by ~$118. The
#      90d/180d disagreement traces to a genuine regime split, not
#      noise: splitting the 180d run at its midpoint, the RECENT ~90
#      days inside it mildly favor ON (~+$51) while the OLDER 90-180
#      days back favor OFF heavily (~+$346) -- two different periods
#      behaving differently, confirmed by a strategy-level breakdown:
#      trend_following's blocked trades are net LOSERS in the recent
#      window (correctly refused) but net WINNERS in the older window
#      (wrongly refused), same reversal for gap_continuation.
#   3 & 4. A FRESH 12-symbol set (NIQ/CDNL/ONON/VREX/HZO/QNST/DOCS/CRSR/
#      TEAM/APPS/BLMN/SEDG), drawn from the ACTUAL live watchlist history
#      (2026-08-03 to 08-11) rather than the stale fixed list -- this
#      matters because the fixed list turned out to have ZERO overlap
#      with any symbol actually real-traded since this filter went live
#      on 2026-07-31 (checked against Alpaca's real order history). On
#      this genuinely current universe, OFF wins DECISIVELY on BOTH
#      windows, robust to every single exclusion (24/24 leave-one-out
#      checks, zero flips):
#        90d:  ON $-32 (PF 0.96, net LOSING) vs OFF $+251 (PF 1.20)
#        180d: ON $+128 (PF 1.08)            vs OFF $+674 (PF 1.26)
#      Drawdown also favors OFF in both (3.77%->3.58% at 90d,
#      4.10%->3.66% at 180d) -- not just a quantity/quality tradeoff
#      here, OFF is better on every axis.
#
#   Verdict: 3 of these 4 tests favor OFF and hold up under their own
#   leave-one-out check; only the fixed list's 90-day result favors ON,
#   and that one does NOT hold up under leave-one-out on independent
#   re-verification -- so it isn't real dissenting evidence, just noise
#   from a symbol set no longer representative of what the bot actually
#   trades. Combined with real-trading cross-checks: VEEE
#   (the single biggest per-symbol contributor to the fixed list's
#   180d OFF-favoring delta, ~60% of it) appeared on the REAL watchlist
#   exactly once across this repo's entire watchlist_state.json history
#   (2026-07-27) and has had zero real trades since 2026-07-28 -- three
#   days before this filter even went live -- so whatever happened to it
#   in these backtests has near-zero bearing on live trading either way.
#   VEEE's own daily-bar history shows real, extreme (300%+ single-day
#   return) volatility events, a plausible mechanism for why a ONE-
#   PRIOR-DAY EMA read would badly lag a violent reversal -- but VEEE
#   isn't uniquely volatile among its peers (INHD, PN show comparably
#   extreme days) and isn't the majority driver of any of the 4 tests
#   above, so this reads as a real but non-dominant contributing factor,
#   not the whole story.
#
#   None of this makes OFF a "clean, unambiguous win" the way the
#   2026-07-31 ON decision was originally described -- it's a real,
#   multi-test, leave-one-out-robust preponderance of evidence, on a
#   feature whose original validation turned out not to generalize.
#   Given this project's default posture (off unless evidence justifies
#   on) and that 3 of 4 independent robustness checks -- including both
#   tests on the currently-representative universe -- favor OFF, this
#   reverts to OFF. Re-test if re-enabling: this analysis is itself
#   sensitive to which universe/window you pick (that's the whole
#   finding), so don't trust one more single-window backtest as the
#   final word either.
USE_MULTI_TIMEFRAME_FILTER = os.getenv("USE_MULTI_TIMEFRAME_FILTER", "false").strip().lower() in ("1", "true", "yes")
# Daily trend only changes once a day (it's driven by daily closes), so
# this doesn't need refreshing anywhere near as often as the 5-minute
# intraday cycle -- cached like SP500_REFRESH_HOURS for the same reason.
DAILY_TREND_REFRESH_HOURS = float(os.getenv("DAILY_TREND_REFRESH_HOURS", 4))

# 2026-07-31: vwap_reversion volume confirmation, SCANNER PICKS ONLY --
# same reasoning and same S&P-500-exempt scoping as the multi-timeframe
# filter above. See VWAP_REVERSION_MIN_VOLUME_MULT's comment in
# strategy.py for the full evidence, including the isolated-test pitfall
# that made this look like a universal win before it was tested in the
# full priority chain.
USE_VWAP_VOLUME_CONFIRMATION = os.getenv("USE_VWAP_VOLUME_CONFIRMATION", "true").strip().lower() in ("1", "true", "yes")

# Broad-market regime gate: veto new long entries in EVERY symbol when
# SPY itself is in a confirmed downtrend on this same BAR_MINUTES
# timeframe. Reuses the exact ADX/trend machinery strategy.py already
# computes for every symbol (ADX_TREND_THRESHOLD, the ema_fast/ema_slow
# pair trend_following_at reads) rather than inventing a new indicator --
# "trending down" means precisely what it means for any other symbol's
# own trend_following regime: ADX >= ADX_TREND_THRESHOLD (a confirmed
# trend, not chop) AND ema_fast < ema_slow (that trend pointing down).
# Checked here, not inside strategy.py's decision functions, for the same
# reason the S&P-500-membership gates above (USE_MULTI_TIMEFRAME_FILTER,
# USE_VWAP_VOLUME_CONFIRMATION) live here: "is a DIFFERENT symbol's tape
# down" is operational/cross-symbol context, and strategy.py stays a
# pure, single-symbol, network-free decision file by design.
#
# Default FLIPPED TO OFF after a 90-day backtest.py run (2026-08-06) came
# back net negative on BOTH universes, on every combined metric at once
# (win rate, total return, profit factor, and max drawdown all worse --
# see USE_SPY_REGIME_GATE's comment in backtest.py for the full numbers).
# Vetoing entries whenever SPY's own ADX/EMA read "downtrend" cut trade
# count by roughly a third in both universes without the surviving
# trades being any higher quality -- it caught real winners along with
# the losers it was meant to filter. Kept in code and toggleable, same
# as USE_RVOL_SPIKE/USE_ROSS_HOOK in strategy.py -- re-test before
# re-enabling, ideally against a different threshold or index.
USE_SPY_REGIME_GATE = os.getenv("USE_SPY_REGIME_GATE", "false").strip().lower() in ("1", "true", "yes")
SPY_REGIME_GATE_SYMBOL = os.getenv("SPY_REGIME_GATE_SYMBOL", "SPY").strip().upper()

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
# Raised 5 -> 10 on 2026-08-09 at the user's explicit request to put more
# of the account to work -- deliberately as MORE $500 positions at once
# (more diversified, same risk per trade) rather than BIGGER individual
# ones. Bigger positions were the exact 2026-08-05 failure mode (see
# CLAUDE.md): one $19,883 position turned an ordinary -6% move into a
# $1,206.55 loss instead of the ~$30 it should have been. This lever is
# different in kind, not just degree -- MAX_PORTFOLIO_HEAT_USD below
# already caps how much AGGREGATE $-risk all of them can carry at once
# regardless of this number, so widening this doesn't widen the account's
# real worst-case exposure, it just lets more small, independent bets run
# at the same time.
#
# Raised again, 10 -> 18, on 2026-08-09 after the user clarified what
# "use the full $100k" actually meant: not a bigger single trade (already
# declined, see above), but not being artificially capped from putting
# the account's capital to work in aggregate either. Checked the real
# number first rather than guessing: this paper account's buying power is
# ~$398k (4x margin) against ~$99.6k equity -- capital was never actually
# the constraint, even at the old cap of 10. MAX_CONCURRENT_POSITIONS was
# always a deliberate RISK ceiling (concentration/correlation), not a
# capital one. Raised to 18 to match SCANNER_WATCHLIST_SIZE exactly --
# the position-count cap can no longer block a trade the scanner itself
# was willing to watch; the real risk ceiling is MAX_PORTFOLIO_HEAT_USD
# and MAX_POSITIONS_PER_SECTOR below, not this number.
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", 18))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", 3))
MAX_PORTFOLIO_RISK_PCT = float(os.getenv("MAX_PORTFOLIO_RISK_PCT", 5.0))

# MAX_PORTFOLIO_RISK_PCT above only ever applies when USE_RISK_BASED_SIZING
# is on -- would_exceed_portfolio_risk_cap() returns False unconditionally
# otherwise, because there's no clean way to turn a %-of-equity cap into a
# per-trade risk estimate when sizing isn't itself %-of-equity. Since the
# 2026-08-05 revert (see CLAUDE.md), this bot runs FIXED $500-per-position
# sizing by default, which means the aggregate-risk cap is currently a
# no-op in the bot's actual default configuration -- nothing bounds how
# much total $-risk a handful of same-cycle entries can stack up to.
#
# USE_PORTFOLIO_HEAT_CAP closes that gap with a FIXED DOLLAR ceiling
# instead of a percentage one, deliberately. A %-of-equity ceiling is
# exactly the shape of thing that caused the 2026-08-05 incident:
# MAX_POSITION_PCT_OF_EQUITY=25 was dormant and harmless for months while
# USE_RISK_BASED_SIZING was off, then silently became a real ~$25k-per-
# position cap the moment sizing mode flipped -- the SAME number meant
# wildly different dollar amounts depending on account size and sizing
# mode, and nothing surfaced that until real fills showed it. A fixed USD
# figure can't do that: $200 means $200 regardless of equity or which
# sizing mode is active, so this composes safely with flat-$ sizing
# instead of reintroducing a second equity-scaled cap alongside it.
#
# Raised twice on 2026-08-09, in lockstep with MAX_CONCURRENT_POSITIONS
# each time (200 -> 250 -> 450), so the heat cap never becomes the
# binding constraint before the position-count cap does. 450 is exactly
# 18 positions' worth of a $25 stop-loss risk each (the default $500
# flat size at a 5% stop: 500 * 0.05 = $25/position), matching
# MAX_CONCURRENT_POSITIONS=18 1:1. Still a hard, fixed-dollar ceiling
# either way -- at most 0.45% of this account's equity can be at risk
# across ALL open positions combined, regardless of how many slots are
# technically available. ON by default (unlike most new toggles here)
# since it is purely restrictive -- it can only ever block a trade,
# never add exposure, so there is no downside risk to leaving it active.
USE_PORTFOLIO_HEAT_CAP = os.getenv("USE_PORTFOLIO_HEAT_CAP", "true").strip().lower() in ("1", "true", "yes")
MAX_PORTFOLIO_HEAT_USD = float(os.getenv("MAX_PORTFOLIO_HEAT_USD", 450))

# Portfolio-CONSTRUCTION control: caps how many currently open positions
# may share the same sector, checked before a new entry (never affects
# closing/selling logic). MAX_PORTFOLIO_RISK_PCT/MAX_PORTFOLIO_HEAT_USD
# already bound aggregate $-at-risk, but say nothing about how
# concentrated that risk is -- five same-sector positions can each pass
# their own risk-cap check individually while the account is really
# making one large correlated bet five times over, and both the scanner
# and the S&P 500 backstop can independently gravitate toward the same
# crowded trade (e.g. several semiconductor names all showing up as
# "today's biggest movers" on the same news cycle).
#
# Investigated 2026-07-31 (see CLAUDE.md, "Idea 2") as a correlation-
# limiting idea and explicitly NOT built at the time: backtest.py runs
# each symbol in total isolation with its own equity curve, so there was
# no way to backtest a P&L before/after for a control that only means
# anything across a shared, concurrent position pool -- that's still
# true today and this doesn't change it. This is a portfolio-construction
# control verified via unit tests on its own gating logic (mirroring
# would_exceed_portfolio_risk_cap below), not a strategy change proven
# out via backtest P&L.
#
# ON by default, same reasoning as USE_PORTFOLIO_HEAT_CAP above -- this
# only ever blocks a trade, never adds exposure, so there's no downside
# to leaving it active even though the sector lookup itself (see
# get_symbol_sector) can be incomplete for unmapped symbols. Unmapped
# symbols fail OPEN (not blocked), so the worst case of an incomplete
# SECTOR_MAP is "the cap doesn't apply to this one symbol," never a
# false block.
USE_SECTOR_CONCENTRATION_CAP = os.getenv("USE_SECTOR_CONCENTRATION_CAP", "true").strip().lower() in ("1", "true", "yes")
MAX_POSITIONS_PER_SECTOR = int(os.getenv("MAX_POSITIONS_PER_SECTOR", 2))

# Best-effort GICS-style sector classification for symbols this bot is
# likely to actually hold: the default fallback list (TSLA/NVDA/COIN/
# AMD/PLTR), other common megacaps the S&P 500 backstop tends to surface
# (see fetch_sp500_candidates), and a handful of single-sector ETFs.
# NOT exhaustive -- the scanner's own picks are often small/micro-caps
# that simply won't be in here, and that's fine: get_symbol_sector
# treats an unmapped symbol as "unknown" and the cap fails OPEN for it
# (see that function's docstring), same fail-open philosophy as
# is_leveraged_etf's None case. Broad, multi-sector index ETFs (SPY,
# QQQ, DIA, IWM, ...) are deliberately left OUT of this map rather than
# assigned a sector -- there isn't a correct single sector to give them,
# and mapping one anyway would make the cap actively wrong for them
# instead of just not applying.
SECTOR_MAP: dict[str, str] = {
    # Information Technology / Semiconductors
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
    # Communication Services
    "GOOGL": "Communication Services", "GOOG": "Communication Services",
    "META": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services", "TMUS": "Communication Services",
    # Consumer Discretionary
    "TSLA": "Consumer Discretionary", "AMZN": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "BKNG": "Consumer Discretionary",
    # Financials
    "COIN": "Financials", "JPM": "Financials", "V": "Financials",
    "MA": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "AXP": "Financials", "SCHW": "Financials",
    # Health Care
    "UNH": "Health Care", "JNJ": "Health Care", "LLY": "Health Care",
    "PFE": "Health Care", "MRK": "Health Care", "ABBV": "Health Care",
    "ABT": "Health Care", "TMO": "Health Care", "DHR": "Health Care",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    # Industrials
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "MMM": "Industrials", "UPS": "Industrials", "HON": "Industrials",
    "RTX": "Industrials", "LMT": "Industrials",
    # Consumer Staples
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples", "PM": "Consumer Staples",
}

# One SPDR sector ETF per GICS sector name used in SECTOR_MAP -- the
# standard, most-liquid single-sector ETF for each (XLK/XLF/XLE/XLV/XLI/
# XLP/XLU/XLY/XLB/XLRE/XLC). Deliberately a SEPARATE map from SECTOR_MAP
# rather than folding ETF tickers directly into it: SECTOR_MAP's values
# are read as sector NAMES in several places (sector_concentration_blocks_entry's
# grouping, log messages), and conflating "sector name" with "this
# sector's benchmark ETF" would make that map do two jobs at once for no
# benefit -- a symbol's sector doesn't change depending on which feature
# is asking.
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

# Sector-relative mean-reversion filter: require a mean_reversion BUY
# candidate to be genuinely underperforming its OWN sector ETF over the
# same lookback window, not just reading RSI-oversold because the whole
# sector (or the whole market) had a soft day. Sourced from statistical-
# arbitrage / short-term-reversal research (Avellaneda & Lee; Quantpedia's
# short-term reversal writeups): a stock's move is a more meaningful
# reversion signal once it's been isolated from its peer group's own move
# over the same window -- "oversold in isolation" is weaker evidence than
# "oversold relative to how its own sector traded today."
#
# This needs a sector ETF's own price series, which is exactly the kind
# of operational/external data strategy.py deliberately has no access to
# (same reasoning as the S&P-500-membership gates and this file's own
# sector-concentration cap above) -- so, same as those, it's implemented
# here and in backtest.py's simulate(), gated externally, and only ever
# consulted for a mean_reversion entry specifically (checked inside
# check_symbol once reason_key is known, same pattern as
# USE_VWAP_VOLUME_CONFIRMATION).
#
# NOT the same idea as USE_SPY_REGIME_GATE above, even though both
# "compare against an external reference series" -- and that similarity
# is exactly why SPY_REGIME_GATE's rejection doesn't settle this one
# either way. SPY regime asks "is the WHOLE market trending down" and
# vetoes every symbol's entries on one binary market-wide read regardless
# of story; that one 90-day test came back net negative on both
# universes (see USE_SPY_REGIME_GATE's comment) because a single broad-
# index read is too blunt -- it vetoed real winners along with the
# losers it meant to catch. This filter is narrower and symbol-specific:
# it doesn't touch every trade, only mean_reversion entries, and it asks
# a comparative question (this stock vs. ITS OWN sector, same window)
# rather than a directional one about the whole tape. Different
# mechanism, different scope -- worth its own honest test rather than
# assuming the SPY result predicts this one.
#
# Backtested 2026-08-11, both universes, 90 days, ON vs. OFF with
# everything else at defaults (see CLAUDE.md for the full run):
#
#   megacap (TSLA/NVDA/COIN/AMD/PLTR): 0 mean_reversion trades either way
#     (ADX rarely drops into mean_reversion's regime on these names in
#     this window) -- combined 78 trades, +7.4%, PF 2.09 identical ON/OFF.
#   scanner (FBRX/VEEE/PN/TRAX/QBTS/SMCI/SAFT/RNG/INHD/FCUV/ATKR/SRAD):
#     2 mean_reversion trades total either way -- combined 104 trades,
#     49% win rate, +2.5% ($+150.62/$6,000), PF 1.30, max DD 1.6%,
#     byte-for-byte identical ON/OFF.
#
# Not a wash by coincidence -- checked WHY: one of the two trades (SRAD)
# is on a symbol with no SECTOR_MAP entry and no sector Alpaca's asset
# metadata can supply either (confirmed the filter fails open for it, per
# get_sector_etf's docstring), so it was never going to be affected by
# ANY threshold. The other (SMCI, -$25.41, IS in SECTOR_MAP -> XLK) DID
# get evaluated: its real underperformance vs. XLK over the 14-bar window
# was between 3 and 5pp (confirmed by sweeping the threshold: unaffected
# at 2.0pp, the shipped default; blocked at 3.0pp and every value tried
# above it) -- it just narrowly cleared the 2.0pp bar this run shipped
# with, so the mechanism is doing real, threshold-sensitive work, this
# sample is simply too small (one single evaluable trade) to say whether
# 2.0pp is calibrated well or not.
#
# Left OFF by default per this project's "default off unless backtest
# evidence clearly justifies on" convention -- a filter that only ever
# got to evaluate ONE real trade this run isn't evidence of anything,
# good or bad, whichever way that one trade happened to land. The
# scanner's own picks are mostly small/micro-caps that SECTOR_MAP
# deliberately doesn't cover (see that map's own docstring), so this
# filter's effective reach is narrower than "every mean_reversion entry"
# even once mean_reversion fires more often -- worth knowing going in,
# not a defect introduced here. Kept in code and toggleable, same as the
# rejected-but-plausible entries above -- re-test if mean_reversion ever
# becomes a bigger slice of trade volume AND SECTOR_MAP's coverage grows
# to actually reach more of the symbols that produce it.
USE_SECTOR_RELATIVE_MEAN_REVERSION = os.getenv("USE_SECTOR_RELATIVE_MEAN_REVERSION", "false").strip().lower() in ("1", "true", "yes")
# Bars over which both the candidate's own return and its sector ETF's
# return are measured. Defaults to RSI_PERIOD (not a separate arbitrary
# number) since that's the exact window mean_reversion's own "is this
# oversold" read is already judging the candidate's price over --
# comparing sector-relative performance across a DIFFERENT window than
# the one that produced the oversold signal in the first place would be
# answering a different question than the one this filter is meant to ask.
SECTOR_RELATIVE_LOOKBACK_BARS = int(os.getenv("SECTOR_RELATIVE_LOOKBACK_BARS", RSI_PERIOD))
# How many percentage points the candidate's own return must trail its
# sector ETF's return over that window before the entry is allowed
# (etf_return_pct - candidate_return_pct >= this). A small/zero threshold
# would let through candidates that merely moved a hair less than their
# sector on an ordinary day -- not the "genuinely underperforming its
# peers" bar the research motivation calls for.
SECTOR_RELATIVE_MIN_UNDERPERFORMANCE_PCT = float(os.getenv("SECTOR_RELATIVE_MIN_UNDERPERFORMANCE_PCT", 2.0))

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
# See USE_BREAKOUT_INVALIDATION_EXIT (strategy.py) and the OPEN POSITION
# CONTEXT section below for what this persists and why.
OPEN_POSITION_CONTEXT_FILE = "open_position_context.json"

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
# Symbols that recently had a position close, and the time they become
# eligible to buy again. See SYMBOL_COOLDOWN_MINUTES.
symbol_cooldown_until: dict[str, datetime] = {}
# What we held on the previous cycle, so a position closing (by stop, by
# target, or by our own sell) can be detected without an extra API call.
previously_held: set[str] = set()
# {symbol: {"strategy": reason_key, "invalidation_level": float | None}}
# for every symbol currently holding a position this bot opened. See the
# OPEN POSITION CONTEXT section below for how this is populated, cleared,
# and persisted.
open_position_context: dict[str, dict] = {}

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

def filter_to_regular_session(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Drops every pre-market/after-hours bar, keeping only the 9:30am-4:00pm
    ET regular session -- mirrors backtest.py's fetch_historical_bars
    filter EXACTLY (same boundary condition, same inclusive/exclusive
    edges), so a bar set that goes through this ends up on the same kind
    of data a backtest would have computed indicators on. MARKET_TZ_FOR_LOGS
    (defined above) is the same America/New_York zone used there -- it's
    used for this filter too now, not just log timestamps.

    2026-08-23: added because add_indicators's groupby(session_date)
    columns (session_open_price, gap %, session_first_bar_high/volume,
    VWAP) are all anchored to whichever bar prints FIRST each calendar
    day -- pre-market or not -- when the input includes extended-hours
    bars. Since backtest.py's fetch_historical_bars has always filtered
    to the regular session before computing indicators but the live path
    never did, live's gap_continuation/vwap_reversion/breakout signals
    (the 3 most-traded live strategies) were being computed on session
    anchors the backtest never saw: measured real divergence in
    session_open_price of +0.35% to +1.8% on ordinary days with modest
    pre-market activity. See get_recent_bars_batch's regular_session_only
    param -- this is applied ONLY on the specific paths that feed
    add_indicators, not to every bar fetch (the liquidity/dollar-volume
    checks in fetch_sp500_candidates and scan_for_volatile_stocks
    legitimately want raw bars, since pre/post-market volume is real
    tradable-liquidity signal for those, not indicator input).
    """
    if bars.empty:
        return bars
    et_time = bars["timestamp"].dt.tz_convert(MARKET_TZ_FOR_LOGS)
    is_regular_session = (
        ((et_time.dt.hour > 9) | ((et_time.dt.hour == 9) & (et_time.dt.minute >= 30)))
        & (et_time.dt.hour < 16)
    )
    return bars[is_regular_session].reset_index(drop=True)


def get_recent_bars_batch(symbols: list[str], lookback_days: int = 10,
                           regular_session_only: bool = False,
                           end: datetime | None = None) -> dict[str, pd.DataFrame]:
    """
    Fetches recent intraday price bars for ALL given symbols in a SINGLE
    API call instead of one call per symbol. This is what keeps checking
    a much bigger watchlist cheap on API rate limits regardless of how
    many symbols the scanner is tracking.
    lookback_days is in CALENDAR days, so 10 calendar days comfortably
    covers enough actual trading sessions to fill our indicator windows.
    Returns a dict of symbol -> DataFrame (missing/empty for any symbol
    with no data returned).

    regular_session_only: when True, applies filter_to_regular_session
    to the raw response before splitting it by symbol -- pass this ONLY
    from callers whose bars feed straight into add_indicators (see that
    function's docstring for why). Defaults to False so every OTHER
    caller (the liquidity/dollar-volume checks) keeps seeing the exact
    same raw, extended-hours-inclusive bars it always has -- this is an
    additive opt-in, not a change to this function's default behavior.

    `end`, if given, caps how recent the returned bars are (e.g. so a
    liquidity check can look at trailing history WITHOUT today -- see
    scan_for_volatile_stocks' dollar-volume check). Defaults to None,
    i.e. through the most recent available bar, same as before this
    parameter existed.
    """
    if not symbols:
        return {}
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
        start=datetime.now(timezone.utc) - timedelta(days=lookback_days),
        end=end,
    )
    bars = data_client.get_stock_bars(request).df
    if bars.empty:
        return {}
    bars = bars.reset_index()
    if regular_session_only:
        bars = filter_to_regular_session(bars)
        if bars.empty:
            return {}
    return {
        symbol: group.drop(columns="symbol").reset_index(drop=True)
        for symbol, group in bars.groupby("symbol")
    }


def get_latest_price(symbol: str) -> float | None:
    """
    The most recent traded price, for pricing a bracket right before
    submitting it. Returns None on any failure so callers fall back to
    the last bar's close -- a slightly stale price is far better than
    skipping a trade over a quote lookup.
    """
    try:
        latest = data_client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=symbol)
        )
        trade = latest.get(symbol) if isinstance(latest, dict) else None
        price = float(trade.price) if trade is not None else None
        return price if price and price > 0 else None
    except Exception as e:
        log.info(f"[{symbol}] Could not fetch a fresh quote ({e}) -- using the last bar's close.")
        return None


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


_sp500_cache: dict = {"symbols": [], "fetched_at": None}


def fetch_sp500_symbols() -> list[str]:
    """
    The current S&P 500 constituent tickers, cached in-process for
    SP500_REFRESH_HOURS. Returns [] on any failure (network, malformed
    CSV, etc.) rather than raising -- callers should treat that as "S&P
    500 breadth unavailable this cycle," not as an error worth stopping
    the scan over. Falls back to the last successfully cached list if a
    refresh fails, since a slightly stale S&P 500 list is far better
    than no liquidity backstop at all.
    """
    now = datetime.now(timezone.utc)
    if (_sp500_cache["fetched_at"] is not None
            and (now - _sp500_cache["fetched_at"]).total_seconds() < SP500_REFRESH_HOURS * 3600):
        return _sp500_cache["symbols"]

    try:
        with urllib.request.urlopen(SP500_LIST_URL, timeout=10) as response:
            text = response.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(text))
        symbols = [s.strip().upper() for s in df["Symbol"] if isinstance(s, str) and s.strip()]
        if not symbols:
            raise ValueError("parsed list was empty")
        _sp500_cache["symbols"] = symbols
        _sp500_cache["fetched_at"] = now
        log.info(f"S&P 500: refreshed constituent list ({len(symbols)} symbols).")
        return symbols
    except Exception as e:
        if _sp500_cache["symbols"]:
            log.warning(f"S&P 500: could not refresh constituent list ({e}) -- "
                         f"using the last cached list ({len(_sp500_cache['symbols'])} symbols).")
            return _sp500_cache["symbols"]
        log.warning(f"S&P 500: could not fetch constituent list ({e}) and no cache exists -- "
                     f"skipping the liquidity backstop this cycle.")
        return []


def fetch_sp500_candidates(already_picked: set[str], needed: int) -> list[str]:
    """
    Fills the S&P 500 liquidity backstop (see USE_SP500_UNIVERSE): the
    top `needed` S&P 500 names NOT already in `already_picked`, ranked by
    trailing average dollar volume -- the most liquid names first, since
    that's the property the backtest tied to this system's actual
    performance, not size of today's move (that's what the movers scan
    already optimizes for).
    Deliberately skips the news-catalyst filter that applies to movers:
    "no recent news" suggests thin/noise volume on a stock that's
    already moving a lot, but says nothing meaningful about an S&P 500
    megacap on an ordinary quiet day.
    """
    if needed <= 0:
        return []
    sp500 = fetch_sp500_symbols()
    candidates = [s for s in sp500 if s not in already_picked]
    if not candidates:
        return []

    try:
        # regular_session_only left at its False default deliberately --
        # this is a dollar-volume liquidity read, not indicator input, and
        # pre/post-market volume is real tradable-liquidity signal here.
        # See filter_to_regular_session's docstring.
        bars = get_recent_bars_batch(candidates, lookback_days=5)
    except Exception as e:
        log.warning(f"S&P 500: could not fetch bars for the liquidity backstop ({e}) -- skipping it this cycle.")
        return []

    scored = []
    for symbol in candidates:
        df = bars.get(symbol)
        if df is None or df.empty:
            continue
        last_price = float(df["close"].iat[-1])
        if last_price < SCANNER_MIN_PRICE:
            continue
        avg_dollar_volume = float((df["close"] * df["volume"]).mean())
        scored.append((symbol, avg_dollar_volume))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    picked = [symbol for symbol, _ in scored[:needed]]
    if picked:
        log.info(f"S&P 500 liquidity backstop -> {', '.join(picked)}")
    return picked


_daily_trend_cache: dict[str, dict] = {}  # symbol -> {"map": {date: bool}, "fetched_at": datetime}


def refresh_daily_trend_maps_if_needed(symbols: list[str]) -> None:
    """
    Keeps _daily_trend_cache fresh for whatever's currently on the
    watchlist, in ONE batched request per refresh rather than one call
    per symbol -- same batching principle as get_recent_bars_batch, so a
    wider watchlist doesn't cost proportionally more API calls.

    Best-effort throughout: a failed refresh leaves existing cache
    entries in place (a slightly stale trend map is far better than
    losing the filter's protection entirely over one bad network call),
    and a symbol with no cache entry at all is treated as "trend unknown"
    by the caller, which fails toward blocking the entry -- see
    compute_daily_trend_map's docstring for why that's the safe default.
    """
    if not USE_MULTI_TIMEFRAME_FILTER or not symbols:
        return
    now = datetime.now(timezone.utc)
    stale = [
        s for s in symbols
        if s not in _daily_trend_cache
        or (now - _daily_trend_cache[s]["fetched_at"]).total_seconds() >= DAILY_TREND_REFRESH_HOURS * 3600
    ]
    if not stale:
        return
    try:
        request = StockBarsRequest(
            symbol_or_symbols=stale,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            # Comfortably covers SLOW_MA(21) trading days of warmup even
            # across weekends/holidays.
            start=now - timedelta(days=60),
        )
        bars = data_client.get_stock_bars(request).df
    except Exception as e:
        log.warning(f"Multi-timeframe filter: could not refresh daily bars for "
                     f"{len(stale)} symbol(s) ({e}) -- keeping existing cache entries.")
        return
    if bars.empty:
        return
    bars = bars.reset_index()
    for symbol, group in bars.groupby("symbol"):
        trend_map = compute_daily_trend_map(group.drop(columns="symbol").reset_index(drop=True))
        _daily_trend_cache[symbol] = {"map": trend_map, "fetched_at": now}
    log.info(f"Multi-timeframe filter: refreshed daily trend for {len(stale)} symbol(s).")


def daily_trend_confirms_entry(symbol: str, today) -> bool:
    """True if `symbol` is clear to take new entries under the
    multi-timeframe filter -- either the filter doesn't apply to it (an
    S&P 500 name; see USE_MULTI_TIMEFRAME_FILTER's comment for why), or
    it applies and the prior day's daily trend was up."""
    if not USE_MULTI_TIMEFRAME_FILTER:
        return True
    if symbol in fetch_sp500_symbols():
        return True
    cached = _daily_trend_cache.get(symbol)
    if cached is None:
        return False  # unknown trend -- fail toward blocking, not allowing
    return bool(cached["map"].get(today, False))


def spy_regime_confirms_entry() -> bool:
    """
    True if new long entries are clear to take this cycle under the SPY
    regime gate -- either the gate is off, or SPY_REGIME_GATE_SYMBOL's OWN
    latest bar does NOT show a confirmed downtrend on this same
    BAR_MINUTES timeframe (ADX >= ADX_TREND_THRESHOLD AND fast EMA < slow
    EMA -- see USE_SPY_REGIME_GATE's comment above for the full
    reasoning). Same true-means-ok framing as daily_trend_confirms_entry
    above, so both gates plug into check_symbol's chain the same way:
    `blocks_entry = not confirms_entry(...)`.

    Computed ONCE per cycle by the caller (SPY isn't itself a tradeable
    watchlist symbol, just a market-regime read), not once per symbol
    checked.

    Fails OPEN (returns True, i.e. doesn't block) on any fetch/data
    failure or while SPY's own indicators are still warming up -- SPY
    itself failing to fetch says something's off with market data
    broadly, which isn't a specific enough reason to pause every other
    symbol's entries on a filter that was never actually evaluated.
    """
    if not USE_SPY_REGIME_GATE:
        return True
    try:
        # regular_session_only=True: this feeds add_indicators (below) same
        # as check_symbol's own bars do -- see filter_to_regular_session's
        # docstring.
        bars_by_symbol = get_recent_bars_batch([SPY_REGIME_GATE_SYMBOL], regular_session_only=True)
    except Exception as e:
        log.warning(f"SPY regime gate: could not fetch {SPY_REGIME_GATE_SYMBOL} bars ({e}) -- not blocking this cycle.")
        return True
    bars = bars_by_symbol.get(SPY_REGIME_GATE_SYMBOL)
    if bars is None or bars.empty:
        log.warning(f"SPY regime gate: no {SPY_REGIME_GATE_SYMBOL} data this cycle -- not blocking.")
        return True

    enriched = add_indicators(bars)
    i = len(enriched) - 1
    adx = enriched["adx"].iat[i]
    ema_fast = enriched["ema_fast"].iat[i]
    ema_slow = enriched["ema_slow"].iat[i]
    if pd.isna(adx) or pd.isna(ema_fast) or pd.isna(ema_slow):
        return True  # SPY's own indicators still warming up -- nothing to gate on yet

    spy_bearish = bool(adx >= ADX_TREND_THRESHOLD and ema_fast < ema_slow)
    if spy_bearish:
        log.info(f"SPY regime gate: {SPY_REGIME_GATE_SYMBOL} in a confirmed downtrend "
                  f"(ADX {adx:.1f}, ema_fast < ema_slow) -- new long entries paused this cycle.")
    return not spy_bearish


def meets_min_listing_age(symbols: list[str], min_days: int) -> set[str]:
    """
    Returns the subset of `symbols` with at least `min_days` TRADING days
    of daily-bar history available from Alpaca as of now. Alpaca simply
    has no bars before a symbol's IPO/listing date, so a short history is
    a free, reliable proxy for "recently listed, no real track record" --
    see USE_SCANNER_MIN_LISTING_AGE's comment for the concrete real cases
    (ALMR ~80 days, EROC ~43) this is meant to catch.

    Reuses the exact same daily-bar StockBarsRequest pattern
    refresh_daily_trend_maps_if_needed already uses above -- one batched
    call, no new API dependency.

    A symbol simply missing from the response (zero bars returned) is
    treated the same as "not enough history" and excluded -- same
    fail-toward-blocking-the-specific-candidate shape as
    daily_trend_confirms_entry's "unknown trend" case and
    is_leveraged_etf's verdict-None case above. A total request failure
    (the endpoint itself down, not a per-symbol data gap) instead fails
    OPEN -- same degrade-rather-than-return-nothing shape as the
    dollar-volume liquidity check below, since a systemic outage isn't a
    reason to block every candidate on a filter that was never actually
    evaluated.
    """
    if not symbols:
        return set()
    # Comfortably covers min_days TRADING days even across weekends and
    # market holidays -- same margin-of-safety style as the 60-calendar-
    # days-for-21-trading-days figure in refresh_daily_trend_maps_if_needed
    # above (roughly a 2x multiplier over the naive 5/7 weekday fraction).
    calendar_days = min_days * 2 + 20
    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=datetime.now(timezone.utc) - timedelta(days=calendar_days),
        )
        bars = data_client.get_stock_bars(request).df
    except Exception as e:
        log.warning(f"SCANNER: listing-age check failed ({e}) -- not filtering on listing age this cycle.")
        return set(symbols)
    if bars.empty:
        return set()
    bars = bars.reset_index()
    counts = bars.groupby("symbol").size()
    return {s for s in symbols if counts.get(s, 0) >= min_days}


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
      4. If USE_SCANNER_MIN_LISTING_AGE is on: requires at least
         SCANNER_MIN_LISTING_AGE_DAYS trading days of daily-bar history
         (see meets_min_listing_age) -- excludes genuinely recent, no-
         track-record listings (real cases: ALMR ~80 days, EROC ~43).
      5. Liquidity check: requires average DOLLAR volume (price x shares)
         of at least SCANNER_MIN_DOLLAR_VOLUME, measured from real recent
         bars EXCLUDING today (see SCANNER_MIN_DOLLAR_VOLUME's comment --
         today's bars are the very spike that got the candidate picked in
         the first place, so including them would let a symbol qualify as
         "liquid" purely off the abnormal volume of the day being
         evaluated). Alpaca's "most actives" list is used only as a free
         fast path (anything in it is already known liquid) and as a
         fallback if the bar fetch fails -- deliberately NOT as a hard
         gate; see SCANNER_MIN_DOLLAR_VOLUME's comment for the outage
         that caused.
      6. If USE_NEWS_FILTER is on: drops candidates with fewer than
         MIN_NEWS_ITEMS recent articles (a mover with no news behind it
         is more likely noise/thin-volume than a real catalyst) -- unless
         that would eliminate every candidate this cycle, in which case
         the filter is skipped for now rather than returning nothing.
      7. Ranks the survivors by size of move and returns the top N.

    Returns an empty list if the scan fails or nothing qualifies --
    callers should fall back to the manual SYMBOLS list in that case.
    """
    try:
        movers = screener_client.get_market_movers(
            MarketMoversRequest(top=SCANNER_CANDIDATE_POOL, market_type=MarketType.STOCKS)
        )
    except Exception as e:
        # A genuine screener-endpoint outage, not "nothing qualified
        # today" -- out of scope for the S&P 500 backstop below, which
        # assumes the rest of the pipeline (bars, Alpaca connectivity in
        # general) is working. refresh_watchlist_if_needed's own
        # SCANNER_MIN_WATCHLIST_SIZE fallback to SYMBOLS covers this case.
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

    # Minimum listing-age filter -- see USE_SCANNER_MIN_LISTING_AGE's
    # comment. Cheap (one batched daily-bar call) and runs before the
    # liquidity check so a too-new symbol doesn't cost an extra intraday
    # bar fetch it was always going to be excluded from anyway.
    if USE_SCANNER_MIN_LISTING_AGE and prefiltered:
        old_enough = meets_min_listing_age(
            [m.symbol for m in prefiltered], SCANNER_MIN_LISTING_AGE_DAYS)
        kept = [m for m in prefiltered if m.symbol in old_enough]
        if len(kept) < len(prefiltered):
            log.info(f"SCANNER: excluded {len(prefiltered) - len(kept)} candidate(s) with fewer than "
                      f"{SCANNER_MIN_LISTING_AGE_DAYS} trading days of listing history "
                      f"from {len(prefiltered)} candidate(s).")
        prefiltered = kept
        if not prefiltered:
            log.warning(f"SCANNER: every candidate this cycle had fewer than "
                         f"{SCANNER_MIN_LISTING_AGE_DAYS} trading days of listing history.")

    # Dollar-volume liquidity check on whatever isn't already known liquid.
    # Bars are fetched with `end` capped at the start of TODAY (ET) so the
    # average deliberately EXCLUDES today's session -- see this function's
    # docstring, step 5. Without this, a stock only has to be loud today to
    # pass, not reliably liquid on an ordinary day; lookback_days is bumped
    # from 5 to 6 so the window still covers a comparable amount of real
    # trailing history once today itself is cut off the end of it.
    needs_check = [m.symbol for m in prefiltered if m.symbol not in known_liquid]
    liquid_enough = set(known_liquid)
    if needs_check:
        try:
            # regular_session_only left at its False default deliberately,
            # same reasoning as fetch_sp500_candidates above -- this is a
            # liquidity read, not indicator input.
            now_et = datetime.now(MARKET_TZ_FOR_LOGS)
            today_start_et = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
            bars = get_recent_bars_batch(needs_check, lookback_days=6, end=today_start_et)
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
    if not qualified and prefiltered:
        log.warning(f"SCANNER: {len(prefiltered)} candidate(s) passed price/extension filters but "
                     f"none met the ${SCANNER_MIN_DOLLAR_VOLUME:,.0f} average dollar-volume bar.")

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

    # S&P 500 liquidity backstop -- runs regardless of how the movers scan
    # did, including when it found nothing at all, since the whole point
    # is guaranteeing liquidity is represented independent of today's
    # biggest-movers list. Reserved slots come out of the movers picks'
    # own budget (not on top of SCANNER_WATCHLIST_SIZE) if the movers scan
    # already filled every slot, so the backstop can't blow the watchlist
    # size past what the rest of the system was tuned against.
    if USE_SP500_UNIVERSE:
        already_in_backstop_budget = len(set(picked) & set(fetch_sp500_symbols()))
        slots_open = max(SCANNER_WATCHLIST_SIZE - len(picked), 0)
        needed = max(SP500_MIN_WATCHLIST_SLOTS - already_in_backstop_budget, 0)
        needed = min(needed, slots_open)
        if needed > 0:
            backstop_picks = fetch_sp500_candidates(set(picked), needed)
            if len(picked) + len(backstop_picks) > SCANNER_WATCHLIST_SIZE:
                overflow = len(picked) + len(backstop_picks) - SCANNER_WATCHLIST_SIZE
                picked = picked[:-overflow] if overflow < len(picked) else []
            picked.extend(backstop_picks)

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
                "symbol_cooldown_until": {s: t.isoformat() for s, t in symbol_cooldown_until.items()},
                "previously_held": sorted(previously_held),
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
    global day_start_equity, current_trading_day, daily_loss_breaker_tripped, \
        symbol_cooldown_until, previously_held
    try:
        with open(DAILY_RISK_STATE_FILE, "r") as f:
            data = json.load(f)
        saved_day = data.get("current_trading_day")
        if saved_day:
            current_trading_day = date.fromisoformat(saved_day)
            day_start_equity = data.get("day_start_equity")
            daily_loss_breaker_tripped = bool(data.get("daily_loss_breaker_tripped", False))
            # Cooldowns must survive across runs: each GitHub Actions job
            # is a separate process, so without this a re-entry ban simply
            # evaporates whenever one job hands over to the next.
            for symbol, raw in (data.get("symbol_cooldown_until") or {}).items():
                try:
                    symbol_cooldown_until[symbol] = datetime.fromisoformat(raw)
                except Exception:
                    continue
            previously_held = set(data.get("previously_held") or [])
            log.info(f"Restored daily risk state from previous run: day {current_trading_day}, "
                      f"breaker tripped: {daily_loss_breaker_tripped}")
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"Could not load saved daily risk state ({e}), starting fresh.")


# ---------------------------------------------------------------------------
# OPEN POSITION CONTEXT -- persists, per currently-open symbol, which
# strategy opened it and (for breakout entries) the frozen invalidation
# level, so USE_BREAKOUT_INVALIDATION_EXIT can be checked in check_symbol
# without any new Alpaca API call: reason_key and the entry bar's own
# breakout_recent_high[_wick] are already known locally at the exact
# moment check_symbol places a breakout BUY, so they're captured right
# there (see check_symbol's BUY branch) rather than re-derived later.
# Persisted to disk (and tracked in git, same as watchlist_state.json/
# daily_risk_state.json -- see .gitignore's comment) so this survives a
# restart or a fresh GitHub Actions checkout exactly like those two.
# ---------------------------------------------------------------------------

def save_open_position_context() -> None:
    try:
        with open(OPEN_POSITION_CONTEXT_FILE, "w") as f:
            json.dump(open_position_context, f)
    except Exception as e:
        log.warning(f"Could not save open-position context: {e}")


def load_open_position_context() -> None:
    """
    Restores {symbol: {"strategy", "invalidation_level"}} after a
    restart. A symbol legitimately open right now but ABSENT from this
    file (e.g. it was bought before this feature existed, or the file
    was deleted/reset) is simply not in the dict afterwards -- callers
    must treat that as "unknown," never guess it was a breakout
    position, which is exactly what check_symbol's .get()-based lookup
    already does for free.
    """
    global open_position_context
    try:
        with open(OPEN_POSITION_CONTEXT_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            open_position_context = data
        log.info(f"Restored open-position context for {len(open_position_context)} "
                  f"symbol(s) from previous run.")
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"Could not load saved open-position context ({e}), starting fresh.")


def clear_open_position_context(symbol: str) -> None:
    """
    Removes a symbol's persisted entry-context once its position is
    fully closed -- called from every path that can close a position
    (check_symbol's own SELL and breakout-invalidation branches,
    record_auto_exit's bracket-leg-close detection, and
    flatten_all_positions), so a later re-entry (possibly opened by a
    completely different strategy) never inherits a stale invalidation
    level left over from the position that used to be there.
    """
    if open_position_context.pop(symbol, None) is not None:
        save_open_position_context()


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


def estimate_new_position_risk_usd(last_price: float, atr_value: float | None, equity: float | None,
                                    high_vol_tercile: bool = False, reason_key: str = "unknown") -> float:
    """
    Best-effort $-at-risk of a position that WOULD be opened right now --
    (entry - stop) * qty -- computed with the exact same
    compute_stop_and_target/compute_position_size helpers place_buy_order
    itself uses for sizing, so the heat-cap gate below is checking against
    a number that actually matches what would really be opened, under
    EITHER sizing mode (flat-$ or risk-based), including the
    USE_VOLATILITY_SCALED_SIZING reduction and the USE_CONVICTION_SIZING
    boost (see place_buy_order) when high_vol_tercile / reason_key say
    one of them applies.

    reason_key is only consulted for the CONVICTION_BOOST_USD substitution
    (mirroring place_buy_order's own flat-sizing branch exactly, including
    its precedence rule: if a trade is BOTH in a high-conviction strategy
    AND in its own high-vol tercile, the volatility-based size-DOWN wins,
    never the conviction-based size-up). Getting this wrong has a real
    direction of harm, unlike the vol-scaled case: overestimating risk
    only makes the heat-cap gate stricter than necessary (still safe), but
    UNDER-estimating a conviction-boosted trade's risk would make the gate
    laxer than its own docstring promises -- silently letting more heat
    onto the book than MAX_PORTFOLIO_HEAT_USD is supposed to allow. Passing
    reason_key here is what keeps that from happening.

    Uses the last completed bar's close rather than a fresh quote --
    place_buy_order re-checks that right before submitting -- since this
    only has to be close enough to gate on, not penny-perfect; a stale
    close moving the qty estimate by a share or two doesn't matter next
    to a $200 cap. Returns 0.0 if last_price is invalid rather than
    raising, so a bad quote fails toward "no estimated risk" (a laxer
    gate), matching how every other best-effort estimate in this file
    degrades.
    """
    if last_price is None or last_price <= 0:
        return 0.0
    stop_price, _ = compute_stop_and_target(last_price, atr_value)
    risk_per_share = max(last_price - stop_price, 0.0)
    if USE_RISK_BASED_SIZING and equity is not None and equity > 0:
        qty = compute_position_size(equity, last_price, stop_price)
    else:
        trade_amount = TRADE_AMOUNT_USD
        if USE_VOLATILITY_SCALED_SIZING and high_vol_tercile:
            trade_amount = VOLATILITY_SCALED_REDUCED_USD
        # elif, not a second independent if: volatility-based size-down
        # always takes precedence over conviction-based size-up when a
        # trade qualifies for both (see place_buy_order's identical
        # branch for the full precedence rationale).
        elif USE_CONVICTION_SIZING and reason_key in HIGH_CONVICTION_STRATEGIES:
            trade_amount = CONVICTION_BOOST_USD
        qty = int(trade_amount // last_price)
    return risk_per_share * qty


def would_exceed_portfolio_heat_cap(portfolio_heat_estimate: float, projected_new_risk_usd: float) -> bool:
    """
    Whether opening one more position would push aggregate open $-risk
    over MAX_PORTFOLIO_HEAT_USD -- a FIXED dollar ceiling, unlike
    would_exceed_portfolio_risk_cap()'s %-of-equity one. See
    USE_PORTFOLIO_HEAT_CAP's comment for why fixed dollars specifically:
    this must keep working (and mean the same thing) regardless of
    account size or which position-sizing mode is active, which a
    percentage-of-equity cap does not.
    """
    if not USE_PORTFOLIO_HEAT_CAP:
        return False
    return (portfolio_heat_estimate + projected_new_risk_usd) > MAX_PORTFOLIO_HEAT_USD


# Sector never changes intraday, so cache by symbol like _asset_name_cache
# above -- including negative ("unknown") results, so a symbol this bot
# keeps re-checking every cycle (it's on the watchlist but never mapped)
# doesn't cost a lookup every single time.
_sector_cache: dict[str, str | None] = {}


def get_symbol_sector(symbol: str) -> str | None:
    """
    Best-effort sector classification for `symbol`, or None if it can't
    be determined. SECTOR_MAP (see its own comment) is checked FIRST,
    not Alpaca's API -- confirmed against alpaca-py 0.43.5 (the version
    actually installed here) that trading.models.Asset has no
    sector/industry field at all, only class/exchange/tradable/
    marginable/shortable/fractionable/etc., so calling get_asset() up
    front would mean a real network round-trip per new symbol for data
    the SDK structurally cannot return today. Falls back to get_asset()
    -- checked defensively via getattr, not assumed -- only for symbols
    the map doesn't cover, so this picks up a sector/industry field for
    free with no code change here if a future alpaca-py version adds one.

    Returns None for anything neither source covers. Callers MUST treat
    None as "unknown, fail open" (skip the sector check for this
    symbol), not as a reason to block a trade -- this bot's watchlist is
    wide and changes daily (scanner picks, S&P 500 backstop), and most
    of the scanner's own picks are small/micro-caps that were never
    going to be in a hardcoded map. The cost of skipping the
    concentration check for one unmapped symbol is near zero; the cost
    of blocking real trades over incomplete sector metadata is not --
    same fail-open reasoning as get_current_portfolio_risk_usd skipping
    positions with no matching stop order, just applied to a lookup
    instead of an order.
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


def sector_concentration_blocks_entry(candidate_symbol: str, open_position_symbols: set[str]) -> bool:
    """
    True if opening `candidate_symbol` would push the same-sector open
    position count to MAX_POSITIONS_PER_SECTOR or beyond. This is a
    portfolio-CONSTRUCTION control -- it needs visibility into every
    other currently open position at once, which strategy.py's pure,
    single-symbol decision functions deliberately don't have (see that
    module's docstring) -- so it lives here and is checked externally,
    the same split already used for MAX_CONCURRENT_POSITIONS and the
    portfolio risk cap above.

    Only meaningful for NEW entries: callers gate this on the same
    `signal == "BUY" and current_qty == 0` branch as the other entry
    checks in check_symbol, so an existing position's own SELL/exit
    logic never passes through here at all.

    Fails OPEN (returns False, never blocks) if the candidate's own
    sector is unknown, or if the feature is off -- see
    get_symbol_sector's docstring for why. An unknown OPEN position's
    sector is simply not counted toward the total (rather than blocking
    the whole check), so one unmapped small-cap sitting in the portfolio
    can't silently disable the cap for every other, mappable symbol.
    """
    if not USE_SECTOR_CONCENTRATION_CAP:
        return False
    candidate_sector = get_symbol_sector(candidate_symbol)
    if candidate_sector is None:
        return False
    same_sector_open = sum(
        1 for symbol in open_position_symbols
        if get_symbol_sector(symbol) == candidate_sector
    )
    return same_sector_open >= MAX_POSITIONS_PER_SECTOR


def get_sector_etf(symbol: str) -> str | None:
    """
    The SPDR sector ETF for `symbol`'s own sector (SECTOR_MAP /
    get_symbol_sector, then SECTOR_ETF_MAP), or None if either lookup
    comes up empty. Callers MUST treat None as "unknown, fail open" --
    same reasoning as get_symbol_sector's own docstring, just one lookup
    further: a symbol with no known sector, or a real sector that simply
    isn't one of the 11 SECTOR_ETF_MAP covers (there is no eligible GICS
    sector left out, but SECTOR_MAP's Alpaca-asset-metadata fallback can
    in principle return a string that doesn't exactly match one of
    SECTOR_ETF_MAP's keys), should skip the sector-relative check for
    that symbol, not block a real trade over it.
    """
    sector = get_symbol_sector(symbol)
    if sector is None:
        return None
    return SECTOR_ETF_MAP.get(sector)


def sector_relative_mean_reversion_blocks_entry(symbol: str, enriched: pd.DataFrame, i: int,
                                                  sector_etf_bars: dict[str, pd.DataFrame] | None) -> bool:
    """
    True if a mean_reversion BUY on `symbol` should be refused because its
    own SECTOR_RELATIVE_LOOKBACK_BARS-bar return isn't meaningfully weaker
    than its sector ETF's return over that same window -- i.e. the RSI-
    oversold-and-turning-up read looks like an ordinary soft day for the
    whole sector, not this stock genuinely lagging its peers. See
    USE_SECTOR_RELATIVE_MEAN_REVERSION's comment above for the research
    motivation. Callers gate this on `signal == "BUY" and reason_key ==
    "mean_reversion"`, same as USE_VWAP_VOLUME_CONFIRMATION's
    reason_key-specific check -- it has nothing to say about any other
    strategy's entries.

    Fails OPEN (never blocks) at every step where the comparison can't be
    made honestly: feature off, sector/ETF unknown (get_sector_etf),
    this cycle's ETF bars didn't fetch, or either series doesn't have
    SECTOR_RELATIVE_LOOKBACK_BARS of history yet. Same fail-open
    philosophy as sector_concentration_blocks_entry -- an incomplete
    sector map or a missed ETF fetch should never be the reason a real
    trade gets blocked.
    """
    if not USE_SECTOR_RELATIVE_MEAN_REVERSION:
        return False
    etf_symbol = get_sector_etf(symbol)
    if etf_symbol is None:
        return False
    if not sector_etf_bars:
        return False
    etf_bars = sector_etf_bars.get(etf_symbol)
    if etf_bars is None or etf_bars.empty:
        return False

    n = SECTOR_RELATIVE_LOOKBACK_BARS
    if i < n or len(etf_bars) <= n:
        return False

    candidate_then = enriched["close"].iat[i - n]
    candidate_now = enriched["close"].iat[i]
    if pd.isna(candidate_then) or candidate_then == 0 or pd.isna(candidate_now):
        return False
    candidate_return_pct = (candidate_now / candidate_then - 1) * 100

    # sector_etf_bars is fetched fresh once per cycle (see run_one_cycle),
    # same cycle as `enriched` -- so the ETF's OWN last row is "now" here,
    # same simplifying assumption spy_regime_confirms_entry makes about
    # SPY's last row. No timestamp alignment needed the way backtest.py's
    # bar-by-bar historical replay requires (see
    # compute_sector_relative_return_at_bars there).
    etf_then = etf_bars["close"].iat[-1 - n]
    etf_now = etf_bars["close"].iat[-1]
    if pd.isna(etf_then) or etf_then == 0 or pd.isna(etf_now):
        return False
    etf_return_pct = (etf_now / etf_then - 1) * 100

    underperformance_pct = etf_return_pct - candidate_return_pct
    return underperformance_pct < SECTOR_RELATIVE_MIN_UNDERPERFORMANCE_PCT


# How far the real fill can drift from the quote used to price the
# bracket before the stop/target legs get corrected. See
# reconcile_bracket_with_real_fill for why this exists.
BRACKET_REPRICE_THRESHOLD_PCT = float(os.getenv("BRACKET_REPRICE_THRESHOLD_PCT", 0.3))


def reconcile_bracket_with_real_fill(order, atr_value: float | None, reference_price: float):
    """
    A market entry fills at whatever the market gives, which can diverge
    from the quote used to PRICE the bracket -- the stop/target legs are
    absolute price levels fixed at submission, so a stale-quote gap
    silently changes the real % risk of the trade, not just its cost.

    Found live on 2026-07-29: HURN's entry quote was $149.055, submitted
    2 seconds later, real fill $154.30 (+3.5%). The bracket's stop stayed
    at $141.60 -- 5% below the QUOTE, but 8.2% below the REAL entry.
    Sizing wasn't wrong (flat-dollar here), but a trader who thought they
    were risking 5% were actually risking 1.65x that if it had reversed.

    Polls briefly for the fill (paper fills are normally near-instant,
    but this must never hang a cycle if one doesn't arrive), and if the
    real fill drifted more than BRACKET_REPRICE_THRESHOLD_PCT from the
    reference price, replaces both leg orders so the stop and target are
    correctly positioned relative to what was ACTUALLY paid rather than
    the quote from a moment before.

    Returns (real_fill_price, corrected) -- real_fill_price falls back to
    reference_price if the fill couldn't be confirmed; corrected is True
    only if the legs were actually repriced.
    """
    filled = None
    for _ in range(6):  # ~3s total -- enough for a paper fill, never worth blocking longer
        try:
            fetched = trading_client.get_order_by_id(order.id, GetOrderByIdRequest(nested=True))
        except Exception as e:
            log.info(f"[{order.symbol}] Could not check fill status ({e}) -- "
                      f"using the pre-trade quote for records.")
            return reference_price, False
        if fetched.status == OrderStatus.FILLED and fetched.filled_avg_price is not None:
            filled = fetched
            break
        time.sleep(0.5)

    if filled is None or not filled.filled_avg_price:
        log.info(f"[{order.symbol}] Fill not confirmed yet -- using the pre-trade quote for records; "
                  f"the bracket legs stay as originally priced.")
        return reference_price, False

    real_fill_price = float(filled.filled_avg_price)
    drift_pct = abs(real_fill_price - reference_price) / reference_price * 100
    if drift_pct <= BRACKET_REPRICE_THRESHOLD_PCT:
        return real_fill_price, False

    new_stop, new_target = compute_stop_and_target(real_fill_price, atr_value)
    new_stop = round(new_stop, 2)
    new_target = round(new_target, 2)
    log.warning(f"[{order.symbol}] Filled at ${real_fill_price:.2f}, {drift_pct:.1f}% away from the "
                 f"${reference_price:.2f} quote the bracket was priced from -- repricing stop/target "
                 f"to stay at the intended % risk (stop -> ${new_stop:.2f}, target -> ${new_target:.2f}).")

    legs = filled.legs or []
    stop_leg = next((leg for leg in legs if leg.order_type == OrderType.STOP), None)
    target_leg = next((leg for leg in legs if leg.order_type == OrderType.LIMIT), None)
    corrected = True
    if stop_leg is not None:
        try:
            trading_client.replace_order_by_id(stop_leg.id, ReplaceOrderRequest(stop_price=new_stop))
        except Exception as e:
            log.warning(f"[{order.symbol}] Could not reprice the stop-loss leg ({e}) -- "
                         f"it stays at its original level.")
            corrected = False
    if target_leg is not None:
        try:
            trading_client.replace_order_by_id(target_leg.id, ReplaceOrderRequest(limit_price=new_target))
        except Exception as e:
            log.warning(f"[{order.symbol}] Could not reprice the take-profit leg ({e}) -- "
                         f"it stays at its original level.")
            corrected = False
    return real_fill_price, corrected


def place_buy_order(symbol: str, last_price: float, atr_value: float | None, equity: float | None,
                     reason_key: str = "unknown", high_vol_tercile: bool = False):
    """
    Buys a stop-loss/take-profit-protected position ("bracket" order).

    Position SIZE comes from strategy.compute_position_size when
    USE_RISK_BASED_SIZING is on and equity is known: sized off how much
    you're willing to LOSE if the stop is hit (RISK_PER_TRADE_PCT of
    equity), not a flat dollar amount -- a volatile stock with a wide
    stop gets fewer shares than a calm one with a tight stop for the
    same dollar risk. Falls back to flat TRADE_AMOUNT_USD sizing if
    risk-based sizing is off or equity couldn't be fetched this cycle.

    high_vol_tercile: this symbol's own trailing realized-vol reading is
    in its top third right now (strategy.high_vol_tercile, see that
    column's comment -- a second regime axis independent of ADX). Only
    consulted in the FLAT-sizing branch below, gated by
    USE_VOLATILITY_SCALED_SIZING (default off): when both are true, this
    trade spends VOLATILITY_SCALED_REDUCED_USD instead of
    TRADE_AMOUNT_USD -- still a flat dollar figure, never a fraction of
    equity, never able to exceed TRADE_AMOUNT_USD (see that constant's
    guard in strategy.py). Ignored entirely under risk-based sizing,
    which already scales share count down for a wide/volatile stop via
    its own mechanism -- stacking a second, independent size cut on top
    of that hasn't been backtested and isn't this candidate's claim.

    reason_key also feeds the FLAT-sizing branch's conviction-boost check,
    gated by USE_CONVICTION_SIZING (default off): if reason_key is in
    HIGH_CONVICTION_STRATEGIES (default EMPTY -- see that set's comment in
    strategy.py for why), this trade spends CONVICTION_BOOST_USD instead
    of TRADE_AMOUNT_USD -- again a flat dollar figure, capped at 2x
    TRADE_AMOUNT_USD by strategy.py's own import-time guard, never a
    fraction of equity. EXPLICIT PRECEDENCE RULE: if a trade is BOTH in a
    high-conviction strategy AND in its own high-vol tercile, the
    volatility-based SIZE-DOWN above always wins over this size-up --
    safety takes precedence over a return-chasing boost, every time, no
    exceptions. Also ignored entirely under risk-based sizing, same
    reasoning as high_vol_tercile above.

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
    # `last_price` is the last completed BAR's close, which can be minutes
    # stale. In a fast-falling stock the market has already moved below it
    # by the time we submit, and Alpaca rejects the whole bracket:
    #   "stop_loss.stop_price must be <= base_price - 0.01"
    # (seen live on VEEE, 2026-07-27 15:28). Pricing the stop off a fresh
    # quote fixes the rejection and, more importantly, makes the stop
    # correct rather than anchored to a price that no longer exists.
    reference_price = get_latest_price(symbol) or last_price

    stop_price, take_profit_price = compute_stop_and_target(reference_price, atr_value)
    stop_price = round(stop_price, 2)
    take_profit_price = round(take_profit_price, 2)

    # Belt and braces: even with a fresh quote the price can tick down
    # between fetch and submit, so never let the stop sit at or above it.
    max_allowed_stop = round(reference_price - 0.01, 2)
    if stop_price > max_allowed_stop:
        log.info(f"[{symbol}] Price moved to ${reference_price:.2f} while sizing -- "
                  f"lowering stop from ${stop_price:.2f} to ${max_allowed_stop:.2f} to stay valid.")
        stop_price = max_allowed_stop
    if stop_price <= 0 or take_profit_price <= reference_price:
        log.warning(f"[{symbol}] Could not build a sane bracket around ${reference_price:.2f} "
                     f"(stop ${stop_price:.2f}, target ${take_profit_price:.2f}) -- skipping.")
        return None, 0.0

    if not stop_is_wider_than_noise(reference_price, atr_value, stop_price):
        atr_pct = (atr_value / reference_price * 100) if reference_price else 0
        stop_pct = (reference_price - stop_price) / reference_price * 100
        log.info(f"[{symbol}] ACTION: No trade -- too volatile to protect. ATR is "
                  f"${atr_value:.2f} ({atr_pct:.1f}% per bar) but the stop is only "
                  f"{stop_pct:.1f}% away, so it sits inside the noise and would be hit "
                  f"at random. Needs {MIN_STOP_TO_ATR_RATIO:.1f}x ATR of room.")
        return None, 0.0

    # Never boosted under risk-based sizing -- see high_vol_tercile's
    # identical reasoning in the docstring above; set here so it's always
    # defined for last_details below regardless of which branch runs.
    conviction_boosted = False
    if USE_RISK_BASED_SIZING and equity is not None and equity > 0:
        qty = compute_position_size(equity, reference_price, stop_price)
        sizing_style = f"risk-based, {RISK_PER_TRADE_PCT:.1f}% of ${equity:,.0f}"
    else:
        trade_amount = TRADE_AMOUNT_USD
        sizing_style = "flat $"
        if USE_VOLATILITY_SCALED_SIZING and high_vol_tercile:
            trade_amount = VOLATILITY_SCALED_REDUCED_USD
            sizing_style = f"flat $ (vol-scaled, high tercile -> ${trade_amount:.0f})"
        # elif, not a second independent if: this makes the two branches
        # structurally mutually exclusive, not just documented as such.
        # EXPLICIT PRECEDENCE RULE -- see this function's docstring: a
        # trade that is BOTH in a high-conviction strategy AND in its own
        # high-vol tercile gets the REDUCED (volatility) amount above,
        # never the boosted one below. Safety (size down on real,
        # measured noise) always wins over a return-chasing boost (size
        # up on an unconfirmed strategy-level edge), every time, no
        # exceptions.
        elif USE_CONVICTION_SIZING and reason_key in HIGH_CONVICTION_STRATEGIES:
            trade_amount = CONVICTION_BOOST_USD
            conviction_boosted = True
            sizing_style = f"flat $ (conviction-boosted, {reason_key} -> ${trade_amount:.0f})"
        qty = int(trade_amount // reference_price)

    if qty < 1:
        log.warning(f"[{symbol}] Computed position size is 0 shares at ${reference_price:.2f} "
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
    notional = qty * reference_price
    log.info(f"[{symbol}] Buying {qty} share(s) (~${notional:.0f}, {sizing_style} sizing) | "
              f"stop-loss ${stop_price:.2f} | take-profit ${take_profit_price:.2f} ({stop_style})")
    order = trading_client.submit_order(order_request)

    real_fill_price, corrected = reconcile_bracket_with_real_fill(order, atr_value, reference_price)
    recorded_stop, recorded_target = stop_price, take_profit_price
    if corrected:
        recorded_stop, recorded_target = compute_stop_and_target(real_fill_price, atr_value)
        recorded_stop, recorded_target = round(recorded_stop, 2), round(recorded_target, 2)

    place_buy_order.last_details = {
        "qty": qty,
        "price": real_fill_price,
        "notional": qty * real_fill_price,
        "stop_loss": recorded_stop,
        "take_profit": recorded_target,
        "sizing_style": sizing_style,
        "equity": equity,
        "conviction_boosted": conviction_boosted,
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
                  current_qty: float, equity: float | None, portfolio_risk_estimate: float,
                  in_cooldown: bool = False, daily_trend_blocks_entry: bool = False,
                  spy_regime_blocks_entry: bool = False, sector_cap_blocks_entry: bool = False,
                  sector_etf_bars: dict[str, pd.DataFrame] | None = None) -> float:
    """
    Checks one symbol and acts on its signal. Returns the notional $
    amount of a newly opened position (0.0 if none), so the caller can
    track MAX_CONCURRENT_POSITIONS, the sector concentration cap, the
    portfolio risk caps (both the %-of-equity one and the fixed-dollar
    heat cap), and the running equity estimate live across a single
    cycle without an extra API call per symbol.
    """
    if df is None or df.empty:
        log.warning(f"[{symbol}] No price data returned, skipping this check.")
        return 0.0

    enriched = add_indicators(df)
    i = len(enriched) - 1
    signal, reason_key, reason = decide_signal_at(enriched, i)
    last_price = enriched["close"].iat[i]
    atr_value = enriched["atr"].iat[i]
    high_vol_tercile = bool(enriched["high_vol_tercile"].iat[i])
    minutes_since_open_now = enriched["minutes_since_open"].iat[i]
    in_lunch_blackout = ENTRY_BLACKOUT_START_MINUTES <= minutes_since_open_now < ENTRY_BLACKOUT_END_MINUTES

    # Scanner-picks-only volume confirmation for vwap_reversion -- see
    # USE_VWAP_VOLUME_CONFIRMATION's comment. Checked here (not inside
    # strategy.py) because it needs S&P 500 membership, which is an
    # operational concept the pure decision file has no access to.
    vwap_volume_blocks_entry = (
        USE_VWAP_VOLUME_CONFIRMATION and signal == "BUY" and reason_key == "vwap_reversion"
        and symbol not in fetch_sp500_symbols()
        and not vwap_reversion_volume_confirms(enriched, i)
    )

    # Sector-relative mean-reversion filter -- see USE_SECTOR_RELATIVE_MEAN_REVERSION's
    # comment. Only ever consulted for a mean_reversion BUY, same
    # reason_key-scoping as vwap_volume_blocks_entry above.
    sector_relative_blocks_entry = (
        USE_SECTOR_RELATIVE_MEAN_REVERSION and signal == "BUY" and reason_key == "mean_reversion"
        and sector_relative_mean_reversion_blocks_entry(symbol, enriched, i, sector_etf_bars)
    )

    # Scanner opening-range entry blackout -- scanner picks (non-S&P-500
    # names) ONLY, see USE_SCANNER_OPENING_BLACKOUT's comment. DIFFERENT
    # FROM and ADDITIVE TO in_lunch_blackout below (that one targets the
    # low-volume midday chop window; this one targets the dangerous first
    # few minutes after the open), and unlike vwap_volume_blocks_entry
    # above (reason_key-scoped) this applies to EVERY strategy's BUY
    # signal, not just one -- the real losses this is modeled on (RGEN,
    # ALMR, NN, ATRO, CHYM) weren't all the same strategy, just all
    # entries into a still-running opening spike. Minutes checked before
    # the S&P-500 membership lookup so the common case (well past the
    # opening window) never pays for that call.
    scanner_opening_blackout_blocks_entry = (
        USE_SCANNER_OPENING_BLACKOUT and signal == "BUY"
        and minutes_since_open_now < SCANNER_OPENING_BLACKOUT_MINUTES
        and symbol not in fetch_sp500_symbols()
    )

    # Breakout invalidation exit -- see USE_BREAKOUT_INVALIDATION_EXIT and
    # breakout_invalidated_at() in strategy.py for the reasoning. Reads the
    # frozen entry-time level from open_position_context (persisted by
    # THIS function's own BUY branch below, on whatever earlier cycle
    # opened the position) instead of any new Alpaca API call. A symbol
    # missing from that state (never opened by breakout, or opened before
    # this feature existed, or the state file was reset) simply never
    # matches here -- .get() returning None fails every check below
    # closed, this never guesses at a position's origin.
    breakout_invalidation_triggered = False
    if USE_BREAKOUT_INVALIDATION_EXIT and current_qty > 0:
        entry_context = open_position_context.get(symbol)
        if (entry_context and entry_context.get("strategy") == "breakout"
                and entry_context.get("invalidation_level") is not None):
            breakout_invalidation_triggered = breakout_invalidated_at(
                enriched, i, entry_context["invalidation_level"])

    log.info(f"[{symbol}] {reason} | Signal: {signal} | Shares held: {current_qty} | Last price: ${last_price:.2f}")

    notional_opened = 0.0
    try:
        if signal == "BUY" and current_qty == 0:
            if entries_paused_reason:
                log.info(f"[{symbol}] ACTION: No trade ({entries_paused_reason}).")
            elif in_cooldown:
                log.info(f"[{symbol}] ACTION: No trade (cooling off for "
                          f"{SYMBOL_COOLDOWN_MINUTES:.0f} min after this symbol's last position closed).")
            elif daily_trend_blocks_entry:
                log.info(f"[{symbol}] ACTION: No trade (multi-timeframe filter -- "
                          f"the prior day's daily trend isn't confirmed up).")
            elif spy_regime_blocks_entry:
                log.info(f"[{symbol}] ACTION: No trade (SPY regime gate -- "
                          f"the broad market is in a confirmed downtrend).")
            elif vwap_volume_blocks_entry:
                log.info(f"[{symbol}] ACTION: No trade (vwap_reversion volume filter -- "
                          f"entry bar's volume didn't confirm).")
            elif sector_relative_blocks_entry:
                log.info(f"[{symbol}] ACTION: No trade (sector-relative mean reversion filter -- "
                          f"not meaningfully weaker than its own sector ETF over the same window).")
            elif scanner_opening_blackout_blocks_entry:
                log.info(f"[{symbol}] ACTION: No trade (scanner opening-range blackout -- within "
                          f"{SCANNER_OPENING_BLACKOUT_MINUTES:.0f} min of the open on a non-S&P-500 "
                          f"scanner pick, real-money-confirmed as the most dangerous entry window).")
            elif in_lunch_blackout:
                log.info(f"[{symbol}] ACTION: No trade (within the historically weak "
                          f"{ENTRY_BLACKOUT_START_MINUTES}-{ENTRY_BLACKOUT_END_MINUTES} min-since-open entry window).")
            elif at_position_cap:
                log.info(f"[{symbol}] ACTION: No trade (at MAX_CONCURRENT_POSITIONS={MAX_CONCURRENT_POSITIONS} cap).")
            elif sector_cap_blocks_entry:
                log.info(f"[{symbol}] ACTION: No trade (at MAX_POSITIONS_PER_SECTOR="
                          f"{MAX_POSITIONS_PER_SECTOR} cap for its sector).")
            elif would_exceed_portfolio_risk_cap(equity, portfolio_risk_estimate):
                log.info(f"[{symbol}] ACTION: No trade (would exceed MAX_PORTFOLIO_RISK_PCT="
                          f"{MAX_PORTFOLIO_RISK_PCT:.1f}% aggregate risk cap).")
            elif would_exceed_portfolio_heat_cap(
                    portfolio_risk_estimate,
                    estimate_new_position_risk_usd(last_price, atr_value, equity, high_vol_tercile, reason_key)):
                log.info(f"[{symbol}] ACTION: No trade (would exceed MAX_PORTFOLIO_HEAT_USD="
                          f"${MAX_PORTFOLIO_HEAT_USD:,.0f} fixed-dollar aggregate risk cap).")
            else:
                order, notional = place_buy_order(symbol, last_price, atr_value, equity, reason_key,
                                                   high_vol_tercile)
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
                    # Persist what opened this position (and, for breakout
                    # specifically, the exact level breakout_at() itself
                    # used to confirm the entry) so a LATER cycle -- quite
                    # possibly a different process entirely, after a
                    # restart -- can check USE_BREAKOUT_INVALIDATION_EXIT
                    # without any new Alpaca API call. See the OPEN
                    # POSITION CONTEXT section above.
                    invalidation_level = None
                    if reason_key == "breakout":
                        level_col = ("breakout_recent_high_wick" if USE_CLOSE_BEYOND_LEVEL_CONFIRMATION
                                     else "breakout_recent_high")
                        level_value = enriched[level_col].iat[i]
                        if not pd.isna(level_value):
                            invalidation_level = float(level_value)
                    open_position_context[symbol] = {
                        "strategy": reason_key,
                        "invalidation_level": invalidation_level,
                    }
                    save_open_position_context()
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
            clear_open_position_context(symbol)
        elif breakout_invalidation_triggered:
            # Same sell mechanics as the strategy-SELL-signal branch above
            # (cancel the resting bracket legs, close the position) -- the
            # only difference is WHY: the level that justified this
            # specific breakout entry gave back, independent of whatever
            # the currently-active regime strategy's own signal says.
            order = place_sell_order(symbol)
            log.info(f"[{symbol}] ACTION: SELL - breakout invalidation exit "
                      f"(close ${last_price:.2f} back below entry level "
                      f"${open_position_context[symbol]['invalidation_level']:.2f}, order id {order.id})")
            record_trade(
                "SELL", symbol,
                trading_day_et=_trading_day_et(),
                strategy="breakout",
                reason="breakout invalidated: price closed back below the entry breakout level",
                qty=current_qty, price=last_price, notional=current_qty * last_price,
                equity=equity, order_id=str(order.id),
                context=extract_context(enriched, i),
            )
            clear_open_position_context(symbol)
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


def record_auto_exit(symbol: str) -> None:
    """
    Records a position's exit in trades.csv when it closes WITHOUT
    check_symbol ever seeing a SELL signal -- i.e. a bracket's stop-loss
    or take-profit leg filling on its own, which is how most exits
    actually happen (check_symbol only sees the strategy's own exit
    signal, never a leg fill).

    Found live on 2026-07-29: trades.csv had BUY rows for GRMN and MANH
    but no exit at all -- both closed via their bracket legs, and
    nothing was watching for that. Without this, most of a day's trades
    are missing their outcome entirely, which defeats the point of
    keeping trades.csv for research.

    Looks up the most recent closed sell for the symbol (the leg fill
    itself) and the most recent closed buy before it (to attribute the
    strategy via client_order_id) via Alpaca's own order history --
    best-effort throughout, since a lookup failure here must never be
    allowed to disrupt the trading cycle that called it.

    Also clears this symbol's open_position_context entry -- the caller
    (run_one_cycle) only reaches this function once it's already detected
    the position is gone, so that part doesn't depend on the Alpaca order
    history lookup below succeeding, unlike the best-effort trades.csv
    recording it does after.
    """
    clear_open_position_context(symbol)
    try:
        sells = trading_client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.CLOSED, symbols=[symbol],
            side=OrderSide.SELL, direction=Sort.DESC, limit=3))
        exit_order = next((o for o in sells if o.filled_at is not None), None)
        if exit_order is None:
            return

        buys = trading_client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.CLOSED, symbols=[symbol],
            side=OrderSide.BUY, direction=Sort.DESC, limit=3))
        entry_order = next(
            (o for o in buys if o.filled_at is not None and o.filled_at < exit_order.filled_at), None)
        strategy = extract_strategy(entry_order.client_order_id) if entry_order else "unknown"

        exit_kind = {OrderType.STOP: "STOP_HIT", OrderType.LIMIT: "TARGET_HIT"}.get(
            exit_order.order_type, "AUTO_EXIT")
        qty = float(exit_order.filled_qty)
        price = float(exit_order.filled_avg_price)
        record_trade(exit_kind, symbol, trading_day_et=_trading_day_et(),
                     strategy=strategy, reason=f"{exit_kind} (order type {exit_order.order_type})",
                     qty=qty, price=price, notional=qty * price, order_id=str(exit_order.id))
    except Exception as e:
        log.info(f"[{symbol}] Could not record how its position closed ({e}) -- "
                  f"not critical, continuing.")


def flatten_all_positions() -> bool:
    """
    Closes every open position individually and cancels pending orders,
    verifying each close instead of trusting a bulk call not to raise.

    Found live on 2026-08-03: trading_client.close_all_positions(cancel_orders=True)
    completed with no exception, and this function logged "all positions
    closed" and wrote FLATTEN rows to trades.csv for MSFT and NVDA -- but
    the account's real order history showed no closing order was ever
    placed for either symbol, and both were still open the next day.
    close_all_positions()'s own docstring explains why this can happen
    silently: it returns "a list of responses from each closed position
    containing the status code and order id" -- a per-symbol failure can
    sit inside that list's status codes without the overall call raising
    at all, and this function never looked at the list's contents, only
    whether the call itself threw. Same silent-failure shape as this
    project's other postmortems (2026-07-24: scanner degrading into an
    empty-but-successful-looking result).

    Fixed by calling close_position() per symbol instead: per its own
    docstring it "will throw an error if the position does not exist"
    -- i.e. a failed close raises for THAT symbol specifically, so it
    can't be silently absorbed into an unchecked bulk response.

    Returns True only if every open position was confirmed closed. The
    caller uses this to decide whether it's safe to mark today's flatten
    as done -- on a partial failure it deliberately is NOT marked done,
    so the next (seconds away) cycle retries instead of giving up for
    the rest of the day with open positions and no further attempt.
    """
    # Read positions BEFORE closing them -- afterwards there's nothing
    # left to describe, and an end-of-day exit is exactly the kind of
    # trade worth being able to analyze separately from a signal-driven
    # one (it's an exit the strategy didn't choose).
    try:
        closing = get_all_open_positions()
    except Exception as e:
        log.error(f"EOD FLATTEN: could not read open positions ({e}) -- aborting, will retry.")
        return False

    if not closing:
        log.info("EOD FLATTEN: no open positions to close.")
        return True

    try:
        trading_client.cancel_orders()
    except Exception as e:
        log.warning(f"EOD FLATTEN: could not cancel open orders ({e}) -- "
                     f"continuing to close positions anyway.")

    day = _trading_day_et()
    all_closed = True
    for symbol, details in closing.items():
        try:
            trading_client.close_position(symbol)
            record_trade("FLATTEN", symbol, trading_day_et=day,
                          strategy="end_of_day", reason="Flattened before market close",
                          qty=details.get("qty"))
            log.info(f"[{symbol}] EOD FLATTEN: close order submitted for {details.get('qty')} share(s).")
            clear_open_position_context(symbol)
        except Exception as e:
            all_closed = False
            log.error(f"[{symbol}] EOD FLATTEN FAILED: {e} -- position likely still open, will retry.")

    if all_closed:
        log.info("EOD FLATTEN: all positions closed, all pending orders cancelled.")
    else:
        log.error("EOD FLATTEN: one or more positions failed to close -- "
                   "NOT marking today's flatten as done, will retry next cycle.")
    return all_closed


def compute_next_cycle_sleep(seconds_left_today: float) -> float:
    """
    How long to sleep before the next check cycle. Normally just
    CHECK_INTERVAL_MINUTES, capped at whatever's left in the session --
    but never allowed to sleep PAST the point where the end-of-day
    flatten window is supposed to start.

    Found 2026-07-31 while rechecking the CHECK_INTERVAL_MINUTES change
    (5 -> 15): with a wider interval, a normal-cadence sleep can overshoot
    FLATTEN_MINUTES_BEFORE_CLOSE's buffer entirely -- e.g. a cycle with
    16 minutes left in the session sleeps the full 15 and wakes with only
    ~1 minute left, instead of the intended 10, right before the bot
    flattens everything for the day. Capping here works for ANY
    CHECK_INTERVAL_MINUTES, rather than relying on
    FLATTEN_MINUTES_BEFORE_CLOSE happening to stay bigger than whatever
    the check interval is set to.

    Extracted out of run_one_cycle specifically so this timing logic is
    unit-testable on its own -- the rest of run_one_cycle needs a live
    Alpaca clock and account state to even reach this point.
    """
    sleep_seconds = min(CHECK_INTERVAL_MINUTES * 60, seconds_left_today)

    if FLATTEN_BEFORE_CLOSE:
        seconds_until_flatten_trigger = seconds_left_today - FLATTEN_MINUTES_BEFORE_CLOSE * 60
        if 0 < seconds_until_flatten_trigger < sleep_seconds:
            sleep_seconds = seconds_until_flatten_trigger + 2

    if sleep_seconds <= 1:
        log.info("Market closing very soon -- pausing checks until the next session.")
        return 30
    return sleep_seconds


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
        daily_loss_breaker_tripped, last_flatten_date, previously_held

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
            if flatten_all_positions():
                last_flatten_date = today_key
            # else: leave last_flatten_date unset. The next cycle is only
            # seconds away and still inside this same flatten window most
            # of the time, so it retries instead of silently leaving
            # positions open for the rest of the day (see flatten_all_positions).
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

    # Anything we held last cycle but don't now has closed -- by stop-loss,
    # by take-profit, or by our own sell. Start its cooldown. Derived from
    # the positions we already fetched, so this costs no extra API call
    # and catches bracket-leg fills the bot never sees as orders.
    just_closed = previously_held - open_position_symbols
    for symbol in just_closed:
        symbol_cooldown_until[symbol] = clock.timestamp + timedelta(minutes=SYMBOL_COOLDOWN_MINUTES)
        log.info(f"[{symbol}] Position closed -- no re-entry for {SYMBOL_COOLDOWN_MINUTES:.0f} min "
                  f"(guards against repeatedly buying the same falling stock).")
        record_auto_exit(symbol)
    if just_closed:
        previously_held = set(open_position_symbols)
        save_daily_risk_state()
    else:
        previously_held = set(open_position_symbols)
    # Computed whenever EITHER portfolio-level cap is active: the %-of-
    # equity one (USE_RISK_BASED_SIZING) or the fixed-dollar heat cap
    # (USE_PORTFOLIO_HEAT_CAP). Both gates check the exact same underlying
    # quantity -- real $-at-risk summed from every open position's actual
    # bracket stop order -- just against two different shapes of ceiling,
    # so there's one shared computation here rather than fetching it twice.
    portfolio_risk_estimate = (
        get_current_portfolio_risk_usd(open_positions)
        if (USE_RISK_BASED_SIZING or USE_PORTFOLIO_HEAT_CAP) else 0.0
    )
    equity_estimate = equity_now

    symbols_to_check = sorted(set(active_watchlist) | open_position_symbols)

    log.info(f"Market is open (closes {fmt(clock.next_close)} your local time). "
              f"Checking {len(symbols_to_check)} symbols: {', '.join(symbols_to_check)}"
              + (f" ({entries_paused_reason})" if entries_paused_reason else ""))

    bars_by_symbol = {}
    try:
        # regular_session_only=True: this is what check_symbol runs
        # add_indicators on for every symbol this cycle -- see
        # filter_to_regular_session's docstring for why extended-hours
        # bars need to be dropped here specifically.
        bars_by_symbol = get_recent_bars_batch(symbols_to_check, regular_session_only=True)
    except Exception as e:
        log.error(f"Could not fetch price data for this cycle's watchlist: {e}")

    refresh_daily_trend_maps_if_needed(symbols_to_check)
    today_et = clock.timestamp.astimezone(MARKET_TZ_FOR_LOGS).date()

    # Computed ONCE per cycle (a market-wide read, not per-symbol) --
    # see spy_regime_confirms_entry's docstring for why this can't just
    # live inside check_symbol like the per-symbol gates do.
    spy_regime_blocks_entry_this_cycle = not spy_regime_confirms_entry()

    # Sector ETF bars for USE_SECTOR_RELATIVE_MEAN_REVERSION, fetched ONCE
    # per cycle in a single batched call for whatever distinct sector
    # ETFs this cycle's watchlist actually needs -- not one fetch per
    # symbol, same batching principle as get_recent_bars_batch itself.
    # Left empty (not fetched at all) when the toggle is off, so this
    # adds zero API calls to the default configuration.
    sector_etf_bars: dict[str, pd.DataFrame] = {}
    if USE_SECTOR_RELATIVE_MEAN_REVERSION:
        needed_etfs = sorted({
            etf for etf in (get_sector_etf(s) for s in symbols_to_check) if etf
        })
        if needed_etfs:
            try:
                # NOTE: regular_session_only left at its False default here.
                # sector_relative_mean_reversion_blocks_entry (below) reads
                # these bars directly (raw close-price returns), never
                # through add_indicators, so it's out of scope for THIS fix
                # (see filter_to_regular_session's docstring) -- flagged as
                # a separate, smaller version of the same class of bug:
                # backtest.py's fetch_sector_etf_bars DOES go through
                # fetch_historical_bars (regular-session filtered), so this
                # is still a live/backtest mismatch, just not the one this
                # commit addresses.
                sector_etf_bars = get_recent_bars_batch(needed_etfs)
            except Exception as e:
                log.warning(f"Sector-relative mean reversion: could not fetch sector ETF bars "
                             f"({e}) -- filter disabled this cycle (fails open, doesn't block).")

    # Tracked live and updated as positions open below (same pattern as
    # open_count/portfolio_risk_estimate) so two same-sector BUYs landing
    # in the SAME cycle still trip the cap on the second one, not just
    # ones that were already open at the top of the cycle.
    open_symbols_this_cycle = set(open_position_symbols)

    for symbol in symbols_to_check:
        at_position_cap = open_count >= MAX_CONCURRENT_POSITIONS
        current_qty = open_positions.get(symbol, {}).get("qty", 0.0)
        cooldown_until = symbol_cooldown_until.get(symbol)
        in_cooldown = cooldown_until is not None and clock.timestamp < cooldown_until
        daily_trend_blocks_entry = not daily_trend_confirms_entry(symbol, today_et)
        sector_cap_blocks_entry = sector_concentration_blocks_entry(symbol, open_symbols_this_cycle)
        notional = check_symbol(symbol, bars_by_symbol.get(symbol), entries_paused_reason,
                                 at_position_cap, current_qty, equity_estimate, portfolio_risk_estimate,
                                 in_cooldown=in_cooldown, daily_trend_blocks_entry=daily_trend_blocks_entry,
                                 spy_regime_blocks_entry=spy_regime_blocks_entry_this_cycle,
                                 sector_cap_blocks_entry=sector_cap_blocks_entry,
                                 sector_etf_bars=sector_etf_bars)
        if notional > 0:
            open_count += 1
            open_symbols_this_cycle.add(symbol)
            # Uses the REAL stop/qty place_buy_order just chose (available
            # on its last_details attribute, no extra API call) rather than
            # re-deriving an estimate -- this has to be correct under BOTH
            # sizing modes now that the fixed-dollar heat cap also reads
            # portfolio_risk_estimate, and equity*RISK_PER_TRADE_PCT/100
            # (the old formula here) is only a valid risk estimate under
            # risk-based sizing, not flat-$ sizing.
            details = place_buy_order.last_details
            actual_new_risk = max(details.get("price", 0.0) - details.get("stop_loss", 0.0), 0.0) * details.get("qty", 0)
            portfolio_risk_estimate += actual_new_risk
            if equity_estimate is not None:
                equity_estimate -= notional

    try:
        clock = trading_client.get_clock()
    except Exception:
        pass

    seconds_left_today = seconds_until(clock.next_close, clock.timestamp) if clock.is_open else 0
    return compute_next_cycle_sleep(seconds_left_today)


# In --duration-minutes mode, a cycle asking to idle at least this long
# is worth a quick "is the session actually over?" check before we commit
# to waiting it out. Derived from CHECK_INTERVAL_MINUTES (with a margin)
# rather than a fixed constant -- found 2026-07-31 while rechecking the
# CHECK_INTERVAL_MINUTES change (5 -> 15) that a hardcoded 900s here
# used to safely exceed every NORMAL cycle's sleep (300s), so this only
# ever fired for genuinely long waits (market closed, up to 3600s). Once
# CHECK_INTERVAL_MINUTES itself became 900s, a normal open-market cycle
# started tripping this on every single tick -- not dangerous (the extra
# clock check just returns False immediately while the market's open),
# but pure waste, and it defeated the intent of the check. Tying it to
# CHECK_INTERVAL_MINUTES means it can't silently fall out of sync again
# if that value changes in the future.
LONG_IDLE_SECONDS = CHECK_INTERVAL_MINUTES * 60 + 60
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
    if USE_SCANNER_MIN_LISTING_AGE:
        log.info(f"Scanner minimum listing age: ON -- requires >= {SCANNER_MIN_LISTING_AGE_DAYS} trading "
                  f"days of daily bar history (real cases this catches: ALMR ~80d, EROC ~43d)")
    log.info(f"Bar size: {BAR_MINUTES}min | Check interval while market open: {CHECK_INTERVAL_MINUTES}min")
    log.info(f"Active strategies: {', '.join(active_strategies)}")
    log.info(f"Trend strategy: EMA {FAST_MA}/{SLOW_MA} crossover | Range strategy: RSI {RSI_PERIOD} "
              f"({RSI_OVERSOLD}/{RSI_OVERBOUGHT}) | Switched by ADX {ADX_PERIOD} (threshold {ADX_TREND_THRESHOLD})")
    log.info(f"No new entries between {ENTRY_BLACKOUT_START_MINUTES}-{ENTRY_BLACKOUT_END_MINUTES} min after "
              f"the open (historically weak window in backtesting)")
    if USE_SCANNER_OPENING_BLACKOUT:
        log.info(f"Scanner opening-range blackout: ON -- no new scanner-pick (non-S&P-500) entries in the "
                  f"first {SCANNER_OPENING_BLACKOUT_MINUTES} min after the open (real-money-confirmed, "
                  f"additive to the lunch-window blackout above, not a replacement for it)")
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
        if USE_VOLATILITY_SCALED_SIZING:
            log.info(f"Volatility-scaled sizing: ON -- trades in a symbol's own top vol-percentile "
                      f"tercile sized at ${VOLATILITY_SCALED_REDUCED_USD:.0f} instead of "
                      f"${TRADE_AMOUNT_USD:.0f} (see strategy.VOL_PERCENTILE_LOOKBACK)")
        if USE_CONVICTION_SIZING:
            if HIGH_CONVICTION_STRATEGIES:
                log.info(f"Conviction-boosted sizing: ON -- {', '.join(sorted(HIGH_CONVICTION_STRATEGIES))} "
                          f"sized at ${CONVICTION_BOOST_USD:.0f} instead of ${TRADE_AMOUNT_USD:.0f} "
                          f"(volatility-based size-down always wins when a trade qualifies for both)")
            else:
                log.info("Conviction-boosted sizing: ON but HIGH_CONVICTION_STRATEGIES is empty -- "
                          "structurally a no-op right now (no strategy currently has real evidence "
                          "supporting a boost, see strategy.HIGH_CONVICTION_STRATEGIES)")
    log.info(f"Position limits: max {MAX_CONCURRENT_POSITIONS} concurrent positions | "
              f"max {MAX_PORTFOLIO_RISK_PCT:.1f}% aggregate portfolio risk | "
              f"daily loss circuit breaker at -{MAX_DAILY_LOSS_PCT:.0f}% (pauses new entries only)")
    if USE_PORTFOLIO_HEAT_CAP:
        log.info(f"Portfolio heat cap: max ${MAX_PORTFOLIO_HEAT_USD:,.0f} aggregate $-at-risk across all open positions")
    if USE_SECTOR_CONCENTRATION_CAP:
        log.info(f"Sector concentration cap: max {MAX_POSITIONS_PER_SECTOR} open positions per sector "
                  f"(new entries only; symbols with no known sector are exempt)")
    if USE_SECTOR_RELATIVE_MEAN_REVERSION:
        log.info(f"Sector-relative mean reversion filter: mean_reversion entries require the candidate "
                  f"to trail its own sector ETF by >= {SECTOR_RELATIVE_MIN_UNDERPERFORMANCE_PCT:.1f}pp over "
                  f"{SECTOR_RELATIVE_LOOKBACK_BARS} bars (symbols with no known sector ETF are exempt)")
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

    load_open_position_context()

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
