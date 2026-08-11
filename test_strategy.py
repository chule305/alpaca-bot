"""
Synthetic regression tests for strategy.py
--------------------------------------------
No network access, no pytest dependency (kept dependency-free, matching
requirements.txt) -- just hand-built pandas DataFrames with known
expected outcomes, run directly with `py test_strategy.py`.

This is the "every change gets tested with synthetic/engineered data
before being called done" standard from CLAUDE.md, made runnable and
kept in the repo so future changes can be checked the same way instead
of re-deriving verification from scratch each time.

NOTE: trading_bot.py's live-only logic (batched Alpaca calls, the news
filter, MAX_CONCURRENT_POSITIONS, the daily loss circuit breaker) needs
a live/mocked Alpaca connection to exercise and is NOT covered here --
this file only covers the pure calculation logic in strategy.py, which
is everything that can be tested without network access.
"""

import numpy as np
import pandas as pd

import strategy as strat

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def ts_range(day: str, start_time: str, n: int, bar_minutes: int = 5) -> list:
    start = pd.Timestamp(f"{day} {start_time}", tz="America/New_York")
    return [(start + pd.Timedelta(minutes=bar_minutes * i)).tz_convert("UTC") for i in range(n)]


def build_df(timestamps: list, opens, highs, lows, closes, volumes) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def flat_bars(timestamps: list, closes, volumes=None) -> pd.DataFrame:
    """Doji-style bars where open == high == low == close, for clean pivot/level tests."""
    closes = list(closes)
    n = len(closes)
    if volumes is None:
        volumes = [1000] * n
    return build_df(timestamps, closes, closes, closes, closes, volumes)


# ---------------------------------------------------------------------------
# 1. Indicator precompute regression: computing once over the full history
#    must match recomputing the same indicator on a growing window slice
#    (the OLD approach) -- this is the core claim behind the speed fix.
# ---------------------------------------------------------------------------

def test_indicator_precompute_matches_growing_window():
    rng = np.random.default_rng(42)
    n = 300
    ts = ts_range("2024-01-02", "09:30", n)
    # Random walk with no missing bars, single continuous session block
    # (session-day boundaries don't matter for EMA/RSI/ADX, only for
    # VWAP/ORB/gap, which aren't under test here).
    steps = rng.normal(0, 0.3, n)
    closes = 100 + np.cumsum(steps)
    highs = closes + np.abs(rng.normal(0.2, 0.1, n))
    lows = closes - np.abs(rng.normal(0.2, 0.1, n))
    opens = closes + rng.normal(0, 0.1, n)
    volumes = rng.integers(800, 1200, n)

    df = build_df(ts, opens, highs, lows, closes, volumes)
    enriched = strat.add_indicators(df)

    for i in [50, 120, 200, 299]:
        window = df.iloc[:i + 1]
        old_ema_fast = strat.compute_ema(window["close"], strat.FAST_MA).iloc[-1]
        old_rsi = strat.compute_rsi(window["close"], strat.RSI_PERIOD).iloc[-1]
        old_adx = strat.compute_adx(window, strat.ADX_PERIOD).iloc[-1]

        new_ema_fast = enriched["ema_fast"].iat[i]
        new_rsi = enriched["rsi"].iat[i]
        new_adx = enriched["adx"].iat[i]

        check(f"ema_fast matches growing-window recompute @ i={i}",
              np.isclose(old_ema_fast, new_ema_fast))
        check(f"rsi matches growing-window recompute @ i={i}",
              (pd.isna(old_rsi) and pd.isna(new_rsi)) or np.isclose(old_rsi, new_rsi))
        check(f"adx matches growing-window recompute @ i={i}",
              (pd.isna(old_adx) and pd.isna(new_adx)) or np.isclose(old_adx, new_adx))


# ---------------------------------------------------------------------------
# 2. Breakout (existing strategy, should still work post-refactor)
# ---------------------------------------------------------------------------

def test_breakout():
    # 21 flat bars, then an initial breakout bar (should NOT fire yet --
    # no confirmation), then a confirmation bar that still pushes higher
    # on continued volume (should fire).
    #
    # USE_TIME_OF_DAY_VOLUME_NORM defaults true (see strategy.py), which
    # compares each bar's volume against OTHER sessions' bars in the same
    # time-of-day bucket -- a single-session fixture has no prior session
    # to build that average from, so it reads NaN and fails closed (see
    # compute_time_of_day_avg_volume's docstring). One prior "normal
    # volume" seed day at the same flat 1000/bar level establishes a real
    # baseline for the buckets this test's bars actually fall in, same
    # seeding pattern test_gap_quality_filter uses for the same reason.
    n = 23
    seed_ts = ts_range("2024-01-01", "09:30", n)
    seed_closes = [100.0] * n
    seed_volumes = [1000] * n
    ts = ts_range("2024-01-02", "09:30", n)
    closes = [100.0] * 21 + [110.0, 115.0]
    volumes = [1000] * 21 + [3000, 3500]
    df = flat_bars(seed_ts + ts, seed_closes + closes, seed_volumes + volumes)
    enriched = strat.add_indicators(df)
    seed_i = n  # bars [0, n) are the seed day; the test day starts at index n
    check("breakout does NOT fire on the initial (unconfirmed) breakout bar",
          not strat.breakout_at(enriched, seed_i + 21))
    check("breakout fires on the confirmation bar (still pushing higher, volume held up)",
          strat.breakout_at(enriched, seed_i + 22))

    volumes_no_spike = [1000] * 23
    df2 = flat_bars(seed_ts + ts, seed_closes + closes, seed_volumes + volumes_no_spike)
    enriched2 = strat.add_indicators(df2)
    check("breakout does NOT fire without a volume spike", not strat.breakout_at(enriched2, seed_i + 22))


# ---------------------------------------------------------------------------
# 3. Smash Day
# ---------------------------------------------------------------------------

def test_smash_day():
    ts = ts_range("2024-01-02", "09:30", 3)
    df = build_df(
        ts,
        opens=[95.5, 94.5, 94.0],
        highs=[96.0, 94.2, 94.5],
        lows=[95.0, 93.0, 93.5],
        closes=[95.5, 94.0, 94.3],
        volumes=[1000, 1000, 1000],
    )
    enriched = strat.add_indicators(df)
    check("smash day fires (setup + break above smash bar's high)", strat.smash_day_at(enriched, 2))

    df2 = df.copy()
    df2.loc[2, "high"] = 94.0  # no longer breaks above bar 1's high (94.2)
    df2.loc[2, "close"] = 93.9
    enriched2 = strat.add_indicators(df2)
    check("smash day does NOT fire without breaking the smash bar's high", not strat.smash_day_at(enriched2, 2))


# ---------------------------------------------------------------------------
# 4. Gap Pattern (Type A)
# ---------------------------------------------------------------------------

def test_gap_continuation():
    day1_ts = ts_range("2024-01-02", "09:30", 5)
    day2_ts = ts_range("2024-01-03", "09:30", 2)

    def make(day2_open, day2_bar1_high, day2_bar1_close, day2_bar2_close):
        ts = day1_ts + day2_ts
        closes = [99.0, 99.5, 100.0, 100.2, 100.0] + [day2_bar1_close, day2_bar2_close]
        opens = [98.5, 99.0, 99.5, 100.0, 100.1] + [day2_open, day2_bar1_close]
        highs = [99.5, 100.0, 100.5, 100.5, 100.3] + [day2_bar1_high, max(day2_bar2_close, day2_bar1_high + 0.1)]
        lows = [98.0, 99.0, 99.3, 100.0, 99.8] + [min(day2_open, day2_bar1_close) - 0.2, day2_bar1_close - 0.1]
        volumes = [1000] * 7
        return build_df(ts, opens, highs, lows, closes, volumes)

    # Day 1 closes at 100.0. Day 2 gaps up 4% to 104, first bar high 105,
    # second bar closes at 106 -- breaks above the first bar's high.
    df = make(day2_open=104.0, day2_bar1_high=105.0, day2_bar1_close=104.5, day2_bar2_close=106.0)
    enriched = strat.add_indicators(df)
    check("gap continuation fires on a qualifying gap + follow-through", strat.gap_continuation_at(enriched, 6))

    # Same shape but only a 0.5% gap -- below GAP_MIN_PCT (2%).
    df2 = make(day2_open=100.5, day2_bar1_high=101.5, day2_bar1_close=101.0, day2_bar2_close=103.0)
    enriched2 = strat.add_indicators(df2)
    check("gap continuation does NOT fire on a sub-threshold gap", not strat.gap_continuation_at(enriched2, 6))


# ---------------------------------------------------------------------------
# 4b. Gap quality gate (USE_GAP_QUALITY_FILTER) -- opening-bar volume vs.
#     this symbol's own historical same-time-of-day normal.
# ---------------------------------------------------------------------------

def test_gap_quality_filter():
    # 3 "normal" prior sessions establish this symbol's historical
    # opening-bar volume for the 9:30 bucket (bucket 0 at the default
    # 30-min TIME_OF_DAY_VOLUME_BUCKET_MINUTES): flat 1000 shares/bar,
    # 5 bars/session, all inside that first 30-min bucket. Each session
    # closes flat at 100.0 so the immediately-prior session's close
    # (what gap_continuation_at's gap %% is measured against) matches
    # the 100.0 used in test_gap_continuation above.
    seed_days = ["2024-01-02", "2024-01-03", "2024-01-04"]
    seed_ts, seed_v = [], []
    for day in seed_days:
        seed_ts += ts_range(day, "09:30", 5)
        seed_v += [1000] * 5
    seed_prices = [100.0] * len(seed_ts)

    gap_ts = ts_range("2024-01-05", "09:30", 2)
    # Same qualifying price shape as test_gap_continuation: 4%% gap up
    # from prior close 100 -> 104, first bar high 105, second bar closes
    # at 106 (breaks the opening bar's high) -- price checks alone
    # already pass here regardless of volume.
    gap_o = [104.0, 104.5]
    gap_h = [105.0, 106.1]
    gap_l = [103.8, 104.4]
    gap_c = [104.5, 106.0]

    def make(gap_open_volume: int) -> pd.DataFrame:
        ts = seed_ts + gap_ts
        opens = seed_prices + gap_o
        highs = seed_prices + gap_h
        lows = seed_prices + gap_l
        closes = seed_prices + gap_c
        volumes = seed_v + [gap_open_volume, gap_open_volume]
        return build_df(ts, opens, highs, lows, closes, volumes)

    gap_i = len(seed_ts) + 1  # the confirmation bar (2nd bar of the gap day)

    # Opening volume 3x the historical 1000/bar normal -> clears
    # GAP_QUALITY_VOLUME_MULT (default 1.5x).
    df_strong = make(gap_open_volume=3000)
    enriched_strong = strat.add_indicators(df_strong)
    check("gap_quality_confirmed_at fires when opening volume is well above normal",
          strat.gap_quality_confirmed_at(enriched_strong, gap_i))
    check("gap_continuation_at (price-only, filter off) still fires on the strong-volume gap",
          strat.gap_continuation_at(enriched_strong, gap_i))

    # Opening volume exactly matches the historical normal (1x, not
    # meaningfully elevated) -- the price shape alone still qualifies as
    # a gap_continuation, but the quality gate should refuse it.
    df_thin = make(gap_open_volume=1000)
    enriched_thin = strat.add_indicators(df_thin)
    check("gap_quality_confirmed_at does NOT fire on ordinary (non-elevated) opening volume",
          not strat.gap_quality_confirmed_at(enriched_thin, gap_i))
    check("gap_continuation_at (price-only, filter off) still fires on the same thin-volume gap",
          strat.gap_continuation_at(enriched_thin, gap_i))

    # Flip the toggle on and confirm gap_continuation_at actually gates
    # end to end, not just the standalone helper.
    original = strat.USE_GAP_QUALITY_FILTER
    try:
        strat.USE_GAP_QUALITY_FILTER = True
        check("USE_GAP_QUALITY_FILTER=true: strong-volume gap still fires",
              strat.gap_continuation_at(enriched_strong, gap_i))
        check("USE_GAP_QUALITY_FILTER=true: thin-volume gap is now blocked",
              not strat.gap_continuation_at(enriched_thin, gap_i))
    finally:
        strat.USE_GAP_QUALITY_FILTER = original


# ---------------------------------------------------------------------------
# 5. Ross Hook
# ---------------------------------------------------------------------------

def _ross_hook_series(final_close: float):
    seg1 = list(np.linspace(100, 90, 11))       # idx0-10: descent to point 1 (low=90 @ idx10)
    seg2 = list(np.linspace(90, 100, 7))[1:]     # idx11-16: rally to point 2 (high=100 @ idx16)
    seg3 = list(np.linspace(100, 93, 7))[1:]     # idx17-22: pullback to point 3 (low=93 @ idx22)
    seg4 = [93.0, 93.0, 93.0]                    # idx23-25: hold (confirms point 3, no new extreme)
    seg5 = [final_close]                         # idx26: hook trigger bar
    closes = seg1 + seg2 + seg3 + seg4 + seg5
    n = len(closes)
    ts = ts_range("2024-01-02", "09:30", n)
    return flat_bars(ts, closes)


def test_ross_hook():
    df = _ross_hook_series(final_close=96.0)  # breaks above point 3's high (93)
    enriched = strat.add_indicators(df)
    check("Ross Hook fires on a valid 1-2-3 + breakout above point 3's high",
          strat.ross_hook_at(enriched, len(df) - 1))

    df2 = _ross_hook_series(final_close=92.5)  # stays below 93 -- no breakout yet
    enriched2 = strat.add_indicators(df2)
    check("Ross Hook does NOT fire before price breaks point 3's high",
          not strat.ross_hook_at(enriched2, len(df2) - 1))


# ---------------------------------------------------------------------------
# 6. Opening Range Breakout
# ---------------------------------------------------------------------------

def test_orb():
    ts = ts_range("2024-01-02", "09:30", 6)  # 09:30 .. 09:55
    df = build_df(
        ts,
        opens=[100.0, 100.5, 101.5, 101.2, 100.0, 103.0],
        highs=[101.0, 102.0, 101.8, 101.5, 103.2, 105.5],
        lows=[99.0, 100.0, 101.0, 100.8, 100.0, 102.8],
        closes=[100.5, 101.5, 101.2, 100.0, 103.0, 105.0],
        volumes=[1000] * 6,
    )
    enriched = strat.add_indicators(df)
    # Opening range (first 15 min = bars 0,1,2) high = 102. Bar 3 (09:45,
    # exactly 15 min in) hasn't broken out yet. Bar 4 (09:50) is the
    # FIRST bar to break above 102 -- should NOT fire yet (unconfirmed).
    # Bar 5 (09:55) confirms by still pushing higher -- should fire.
    check("ORB does NOT fire while still below the opening range high", not strat.orb_at(enriched, 3))
    check("ORB does NOT fire on the initial (unconfirmed) breakout bar", not strat.orb_at(enriched, 4))
    check("ORB fires on the confirmation bar (still pushing higher)", strat.orb_at(enriched, 5))


# ---------------------------------------------------------------------------
# 7. VWAP mean-reversion
# ---------------------------------------------------------------------------

def test_compute_daily_trend_map():
    """
    Regression cover for the 2026-07-31 multi-timeframe filter. Builds
    synthetic DAILY bars (not the intraday BAR_MINUTES bars every other
    test here uses) spanning a clear downtrend into a clear uptrend, and
    checks the trend map is (a) shifted by exactly one day -- a date's
    entries are gated on the PRIOR day's already-closed trend, never
    today's own still-forming one -- and (b) fails toward False (blocked)
    for anything unknown, matching stop_is_wider_than_noise's philosophy.
    """
    # +20h so the UTC timestamp lands in the afternoon ET on the SAME
    # calendar date once _session_date converts it -- midnight UTC would
    # convert back to the PRIOR ET date and silently shift every key.
    days = pd.date_range("2024-01-02", periods=40, freq="B", tz="UTC") + pd.Timedelta(hours=20)
    # Falling for the first 25 sessions, then a sustained rally -- long
    # enough for EMA(9)/EMA(21) to unambiguously flip.
    closes = [100.0 - i * 0.8 for i in range(25)] + [80.0 + i * 1.5 for i in range(15)]
    daily = pd.DataFrame({"timestamp": days, "close": closes})

    trend_map = strat.compute_daily_trend_map(daily)

    check("returns one entry per date in the input", len(trend_map) == len(days))

    first_date = days[0].date()
    check("the very first date (no prior day at all) is False, not missing",
          trend_map.get(first_date) is False)

    # Directly prove the one-day shift: manually compute the RAW
    # (unshifted) trend for every day, then confirm the map's value for
    # day N equals the raw trend at day N-1 -- not day N's own, which
    # would be lookahead (today's daily bar is still forming intraday).
    ema_fast = daily["close"].ewm(span=strat.FAST_MA, adjust=False).mean()
    ema_slow = daily["close"].ewm(span=strat.SLOW_MA, adjust=False).mean()
    raw_trend_up = (ema_fast > ema_slow).tolist()
    mismatches = 0
    for idx in range(1, len(days)):
        expected = raw_trend_up[idx - 1]
        actual = trend_map[days[idx].date()]
        if actual != expected:
            mismatches += 1
    check("every date's value equals the PRIOR day's raw trend, not its own (no lookahead)",
          mismatches == 0)

    last_date = days[-1].date()
    check("the trend has clearly flipped up by the end of the rally",
          trend_map.get(last_date) is True)

    check("an empty daily-bars input returns an empty map, not an error",
          strat.compute_daily_trend_map(pd.DataFrame()) == {})
    check("None input returns an empty map, not an error",
          strat.compute_daily_trend_map(None) == {})


def test_stop_must_be_wider_than_noise():
    """
    Regression for 2026-07-27, using that day's real numbers. The bot
    bought VEEE three times with a 5% stop while VEEE moved 6-7% per
    5-minute bar -- the stop was inside one bar's normal range. Those
    three trades lost 8.78%, 8.32% and 8.26% and made up $127 of the
    day's $172 loss.
    """
    cases = [
        # symbol, price, atr/bar, should_be_tradeable
        ("VEEE", 17.25, 1.2602, False),   # ATR 7.31%/bar vs a 5% stop
        ("VEEE", 17.36, 1.1651, False),   # 6.71%
        ("VEEE", 16.84, 1.0166, False),   # 6.04%
        ("TRAX", 44.33, 0.9779, True),    # 2.21% -- tight but survivable
        ("QBTS", 19.24, 0.2633, True),    # 1.37%
        ("COIN", 163.56, 1.0310, True),   # 0.63%
        ("NVDA", 199.13, 0.9986, True),   # 0.50%
    ]
    for symbol, price, atr, tradeable in cases:
        stop, _ = strat.compute_stop_and_target(price, atr)
        got = strat.stop_is_wider_than_noise(price, atr, stop)
        label = "tradeable" if tradeable else "refused as too volatile"
        check(f"{symbol} at ATR {atr / price * 100:.2f}%/bar is {label}", got is tradeable)

    check("a missing ATR does not block the trade (no data != bad data)",
          strat.stop_is_wider_than_noise(100.0, None, 95.0))
    check("a NaN ATR does not block the trade",
          strat.stop_is_wider_than_noise(100.0, float("nan"), 95.0))
    check("a stop at or above the entry is always refused",
          not strat.stop_is_wider_than_noise(100.0, 0.1, 100.0))


def test_vwap_reversion():
    # A RANGING stock (low ADX): stretched below VWAP and turning up is a
    # genuine reversion setup, and should fire.
    n = 40
    ranging = [100.0 + (1.0 if i % 2 else -1.0) for i in range(n - 2)] + [96.0, 96.6]
    enriched = strat.add_indicators(flat_bars(ts_range("2024-01-02", "09:30", n), ranging))
    check("VWAP reversion fires in a RANGE when stretched below VWAP and turning up",
          strat.vwap_reversion_at(enriched, n - 1))

    ranging_no_turn = [100.0 + (1.0 if i % 2 else -1.0) for i in range(n - 2)] + [96.0, 95.5]
    enriched_no_turn = strat.add_indicators(
        flat_bars(ts_range("2024-01-02", "09:30", n), ranging_no_turn))
    check("VWAP reversion does NOT fire if price keeps falling",
          not strat.vwap_reversion_at(enriched_no_turn, n - 1))

    # A stock in a STRONG DOWNTREND (high ADX). This bar is deeply below
    # VWAP and ticking up -- textbook "stretched and turning", which the
    # old code bought without hesitation. It's the exact shape of the
    # 2026-07-27 losses (VEEE at ADX 44.6, TRAX at 25.0, NVDA at 53.6):
    # not a bounce, just a pause on the way down.
    trending = [100.0 - i * 0.8 for i in range(n - 1)]
    trending.append(trending[-1] + 0.3)
    enriched_trend = strat.add_indicators(
        flat_bars(ts_range("2024-01-02", "09:30", n), trending))
    stretched = enriched_trend["close"].iat[n - 1] < enriched_trend["vwap"].iat[n - 1] * 0.985
    turning_up = enriched_trend["close"].iat[n - 1] > enriched_trend["close"].iat[n - 2]
    check("(fixture sanity) the trending bar really is stretched below VWAP and turning up",
          stretched and turning_up)
    check("VWAP reversion does NOT fire in a strong downtrend -- the knife-catching case",
          not strat.vwap_reversion_at(enriched_trend, n - 1))

    # Unknown ADX must fail closed: with no trend reading there's no way to
    # tell reversion from knife-catching.
    short = strat.add_indicators(
        flat_bars(ts_range("2024-01-02", "09:30", 11), [100.0] * 9 + [96.0, 96.5]))
    check("VWAP reversion does NOT fire when ADX is unknown (fails closed)",
          not strat.vwap_reversion_at(short, 10))


def test_vwap_reversion_volume_confirms():
    """
    vwap_reversion_volume_confirms is a SEPARATE, standalone check (not
    folded into vwap_reversion_at itself) -- see its docstring for why:
    it's only applied to scanner picks by trading_bot.py/backtest.py, and
    "is this an S&P 500 symbol" is an operational concept strategy.py
    deliberately has no access to. This tests the check in isolation from
    whether vwap_reversion_at itself would fire.
    """
    n = 40
    ts = ts_range("2024-01-02", "09:30", n)
    ranging = [100.0 + (1.0 if i % 2 else -1.0) for i in range(n - 2)] + [96.0, 96.6]

    normal_volume = [1000] * (n - 1) + [1000]  # entry bar volume == average
    enriched_normal = strat.add_indicators(flat_bars(ts, ranging, volumes=normal_volume))
    check("does NOT confirm when entry-bar volume is merely average",
          not strat.vwap_reversion_volume_confirms(enriched_normal, n - 1))

    high_volume = [1000] * (n - 1) + [1500]  # entry bar 1.5x the average
    enriched_high = strat.add_indicators(flat_bars(ts, ranging, volumes=high_volume))
    check("confirms when entry-bar volume clears VWAP_REVERSION_MIN_VOLUME_MULT",
          strat.vwap_reversion_volume_confirms(enriched_high, n - 1))

    check("a missing average-volume reading does not confirm (fails closed)",
          not strat.vwap_reversion_volume_confirms(
              strat.add_indicators(flat_bars(ts_range("2024-01-02", "09:30", 3), [100.0, 101.0, 102.0])), 1))


# ---------------------------------------------------------------------------
# 8. Relative volume spike
# ---------------------------------------------------------------------------

def test_rvol_spike():
    ts = ts_range("2024-01-02", "09:30", 21)
    closes = [100.0] * 20 + [101.0]
    opens = [100.0] * 20 + [100.0]
    volumes = [1000] * 20 + [5000]
    df = build_df(ts, opens, closes, closes, closes, volumes)
    enriched = strat.add_indicators(df)
    check("RVOL spike fires on a green bar with unusual volume", strat.rvol_spike_at(enriched, 20))

    closes_red = [100.0] * 20 + [99.0]
    df2 = build_df(ts, [100.0] * 21, closes_red, closes_red, closes_red, volumes)
    enriched2 = strat.add_indicators(df2)
    check("RVOL spike does NOT fire on a red bar even with volume spike", not strat.rvol_spike_at(enriched2, 20))

    # Green, volume-spiking, but closes weakly (near the LOW of its own
    # range) -- should NOT fire, that's exactly the false-start pattern
    # RVOL_MIN_CLOSE_STRENGTH was added to filter out.
    opens_weak = [100.0] * 20 + [100.0]
    highs_weak = [100.0] * 20 + [103.0]
    lows_weak = [100.0] * 20 + [99.5]
    closes_weak = [100.0] * 20 + [100.2]  # green (100.2 > 100) but near the low of [99.5, 103]
    df3 = build_df(ts, opens_weak, highs_weak, lows_weak, closes_weak, volumes)
    enriched3 = strat.add_indicators(df3)
    check("RVOL spike does NOT fire on a weak close even when green with volume",
          not strat.rvol_spike_at(enriched3, 20))


# ---------------------------------------------------------------------------
# 9. decide_signal / decide_signal_at agree, and the live-bot entry point
#    (which computes indicators internally) matches the backtester's
#    precompute-then-decide path.
# ---------------------------------------------------------------------------

def test_decide_signal_entry_points_agree():
    # Needs >= min_bars_needed (max(SLOW_MA, RSI_PERIOD, ADX_PERIOD,
    # BREAKOUT_LOOKBACK) + 5 = 26) for decide_signal_at's overall warmup
    # gate, even though breakout_at itself only needs BREAKOUT_LOOKBACK.
    # First 3 bars (the opening range) are set to 115 so ORB -- which
    # outranks breakout in priority -- doesn't ALSO fire once the final
    # bars push up near that level. breakout now requires a confirmation
    # bar (see breakout_at), so the last TWO bars are the breakout
    # attempt (110, unconfirmed) then the confirmation (114, still below
    # the 115 ORB level so ORB stays out of it).
    #
    # A one-day prior seed session (flat, normal volume) is prepended so
    # USE_TIME_OF_DAY_VOLUME_NORM's per-bucket historical average has
    # something to compare the confirmation bar's volume spike against --
    # see test_breakout's comment for why a single-session fixture alone
    # reads NaN and fails closed.
    n = 31
    seed_ts = ts_range("2024-01-01", "09:30", n)
    seed_closes = [100.0] * n
    seed_volumes = [1000] * n
    ts = ts_range("2024-01-02", "09:30", n)
    closes = [115.0] * 3 + [100.0] * (n - 5) + [110.0, 114.0]
    volumes = [1000] * (n - 2) + [3000, 3500]
    df = flat_bars(seed_ts + ts, seed_closes + closes, seed_volumes + volumes)

    live_signal, live_reason = strat.decide_signal(df)
    enriched = strat.add_indicators(df)
    bt_signal, bt_reason_key, bt_reason = strat.decide_signal_at(enriched, len(df) - 1)

    check("decide_signal (live) and decide_signal_at (backtest) agree on signal", live_signal == bt_signal)
    check("decide_signal (live) and decide_signal_at (backtest) agree on reason text", live_reason == bt_reason)
    check("breakout wins priority and reports the right reason_key", bt_reason_key == "breakout")


# ---------------------------------------------------------------------------
# 10. Stop/target sizing: fixed % (default) vs. ATR-based (opt-in)
# ---------------------------------------------------------------------------

def test_compute_stop_and_target():
    # Explicitly forces USE_ATR_STOPS both ways rather than relying on
    # whatever the module-level default currently is, so this stays
    # correct even if that default changes again later.
    entry = 100.0
    original = strat.USE_ATR_STOPS

    try:
        strat.USE_ATR_STOPS = False
        stop, target = strat.compute_stop_and_target(entry, atr_value=2.5)
        expected_stop = entry * (1 - strat.STOP_LOSS_PCT / 100)
        expected_target = entry * (1 + strat.TAKE_PROFIT_PCT / 100)
        check("fixed-%% stops used when USE_ATR_STOPS=false even with an ATR value available",
              np.isclose(stop, expected_stop) and np.isclose(target, expected_target))

        strat.USE_ATR_STOPS = True
        stop_atr, target_atr = strat.compute_stop_and_target(entry, atr_value=2.5)
        expected_stop_atr = entry - strat.ATR_STOP_MULTIPLIER * 2.5
        expected_target_atr = entry + strat.ATR_TARGET_MULTIPLIER * 2.5
        check("ATR-based stops used when USE_ATR_STOPS=true and ATR is available",
              np.isclose(stop_atr, expected_stop_atr) and np.isclose(target_atr, expected_target_atr))

        stop_fallback, target_fallback = strat.compute_stop_and_target(entry, atr_value=None)
        check("falls back to fixed %% when USE_ATR_STOPS=true but no ATR value was passed",
              np.isclose(stop_fallback, expected_stop) and np.isclose(target_fallback, expected_target))

        stop_nan, target_nan = strat.compute_stop_and_target(entry, atr_value=float("nan"))
        check("falls back to fixed %% when the ATR value is NaN (not warmed up yet)",
              np.isclose(stop_nan, expected_stop) and np.isclose(target_nan, expected_target))
    finally:
        strat.USE_ATR_STOPS = original


# ---------------------------------------------------------------------------
# 11. Risk-based position sizing
# ---------------------------------------------------------------------------

def test_compute_position_size():
    # $10,000 equity, 1% risk per trade (default) = $100 at risk.
    # Entry $50, stop $48 -> $2/share risk -> 50 shares by risk, but
    # MAX_POSITION_PCT_OF_EQUITY (5% since the 2026-08-05 sizing-incident
    # revert, see CLAUDE.md) caps notional at $500 / $50 = 10 shares --
    # the cap binds here rather than risk alone, unlike the old 25% cap
    # this test originally assumed.
    equity_pct_cap_qty = int(10000 * strat.MAX_POSITION_PCT_OF_EQUITY / 100 / 50)
    qty = strat.compute_position_size(equity=10000, entry_price=50, stop_price=48)
    check("position size matches min(risk_amount / risk_per_share, equity pct cap)",
          qty == min(50, equity_pct_cap_qty), f"got {qty}")

    # A wider stop (volatile stock) reduces the risk-based share count
    # further below the equity cap -- $100 risk / $10 per share = 10
    # shares by risk, vs. the same 10-share equity cap above, so the two
    # mechanisms agree here rather than one dominating. Use a stop wide
    # enough that risk-based sizing alone would clearly undercut the cap,
    # confirming the reduction is real and not just re-hitting the same cap.
    qty_wide_stop = strat.compute_position_size(equity=10000, entry_price=50, stop_price=30)
    expected_wide_stop_qty = min(int(100 / 20), equity_pct_cap_qty)
    check("wider stop (more volatile) produces a smaller-or-equal position",
          qty_wide_stop <= qty and qty_wide_stop == expected_wide_stop_qty,
          f"got {qty_wide_stop} vs {qty}, expected {expected_wide_stop_qty}")

    # A very tight stop should hit the MAX_POSITION_PCT_OF_EQUITY cap
    # instead of implying an absurdly large position (risk alone would
    # suggest 1000 shares here: $100 risk / $0.10 per share).
    qty_tight_stop = strat.compute_position_size(equity=10000, entry_price=50, stop_price=49.9)
    expected_capped_qty = int(10000 * strat.MAX_POSITION_PCT_OF_EQUITY / 100 / 50)
    check("very tight stop is capped by MAX_POSITION_PCT_OF_EQUITY, not risk alone",
          qty_tight_stop == expected_capped_qty, f"got {qty_tight_stop}, expected {expected_capped_qty}")

    check("invalid stop (>= entry) returns 0, not a negative/nonsense size",
          strat.compute_position_size(equity=10000, entry_price=50, stop_price=50) == 0)
    check("zero/negative equity returns 0",
          strat.compute_position_size(equity=0, entry_price=50, stop_price=48) == 0)


# ---------------------------------------------------------------------------
# 12. mean_reversion's new "must be turning up" entry confirmation
# ---------------------------------------------------------------------------

def test_mean_reversion_turning_confirmation():
    ts = ts_range("2024-01-02", "09:30", 20)
    # A long, steady decline keeps RSI oversold but price is STILL falling
    # on the final bar -- should NOT buy (no confirmation of a turn).
    closes_still_falling = list(np.linspace(110, 90, 19)) + [89.5]
    df_falling = flat_bars(ts, closes_still_falling)
    enriched_falling = strat.add_indicators(df_falling)
    check("mean_reversion does NOT buy while still falling, even if oversold",
          strat.mean_reversion_at(enriched_falling, len(df_falling) - 1) != "BUY")

    # Same decline, but the final bar ticks up -- should buy.
    closes_turning_up = list(np.linspace(110, 90, 19)) + [90.5]
    df_turning = flat_bars(ts, closes_turning_up)
    enriched_turning = strat.add_indicators(df_turning)
    check("mean_reversion buys once oversold AND turning back up",
          strat.mean_reversion_at(enriched_turning, len(df_turning) - 1) == "BUY")


# ---------------------------------------------------------------------------
# 13. Volatility-scaled sizing: second regime axis (Moreira & Muir 2017),
#    independent of ADX -- see VOL_PERCENTILE_LOOKBACK's comment in
#    strategy.py. Tests the PURE indicator (compute_vol_percentile) and
#    its wiring into add_indicators() separately; the actual sizing
#    decision (which $ amount a HIGH-tercile trade spends) lives outside
#    strategy.py in trading_bot.py/backtest.py, per this file's own
#    "computes, doesn't decide sizing" split, and is covered by
#    test_trading_bot.py instead.
# ---------------------------------------------------------------------------

def test_vol_percentile_ranks_against_own_trailing_history_only():
    # Constant price so ATR-as-%-of-price equals the raw ATR value,
    # keeping the arithmetic legible. 30 quiet bars (ATR=1.0, tied) then
    # one spike bar (ATR=5.0), lookback=10.
    atr = pd.Series([1.0] * 30 + [5.0])
    close = pd.Series([100.0] * 31)
    result = strat.compute_vol_percentile(atr, close, lookback=10)

    check("NaN for the first (lookback - 1) bars -- not enough trailing history to rank yet",
          result.iloc[:9].isna().all())
    check("a tied plateau (nothing stands out) does NOT rank at the top",
          not pd.isna(result.iloc[29]) and result.iloc[29] < strat.HIGH_VOL_TERCILE_CUTOFF,
          f"got {result.iloc[29]}")
    check("a bar that spikes well above its own trailing window ranks at the very top (pct=1.0)",
          np.isclose(result.iloc[30], 1.0), f"got {result.iloc[30]}")

    # No-lookahead: values already computed at earlier bars must not
    # change once MORE data (here, an even bigger future spike) is
    # appended after them -- same guarantee test_indicator_precompute_
    # matches_growing_window checks for every other indicator in this file.
    atr_extended = pd.concat([atr, pd.Series([50.0])], ignore_index=True)
    result_extended = strat.compute_vol_percentile(atr_extended, pd.Series([100.0] * 32), lookback=10)
    check("appending a future bar doesn't change any already-computed percentile (no lookahead)",
          np.allclose(result.iloc[9:31].values, result_extended.iloc[9:31].values, equal_nan=True))

    # A bar that's unusually QUIET relative to its own trailing window
    # ranks near the bottom, not the top -- confirms this isn't just
    # "any deviation reads as high vol."
    atr_low_spike = pd.Series([1.0 + 0.1 * i for i in range(10)] + [0.05])
    result_low = strat.compute_vol_percentile(atr_low_spike, pd.Series([100.0] * 11), lookback=10)
    check("a bar that's unusually QUIET (not loud) relative to its own window ranks near the bottom",
          result_low.iloc[10] < 1.0 / 3.0, f"got {result_low.iloc[10]}")


def test_high_vol_tercile_wiring_in_add_indicators():
    # Monkeypatched to a small lookback purely for a fast, deterministic
    # test -- compute_vol_percentile itself (tested above) takes lookback
    # as a plain argument, so only the add_indicators() WIRING (reading
    # the module constant, applying HIGH_VOL_TERCILE_CUTOFF) needs this.
    original_lookback = strat.VOL_PERCENTILE_LOOKBACK
    try:
        strat.VOL_PERCENTILE_LOOKBACK = 20

        n_quiet, n_loud = 50, 15
        ts = ts_range("2024-01-02", "09:30", n_quiet + n_loud, bar_minutes=15)
        widths = [1.0] * n_quiet + [20.0] * n_loud
        # Symmetric high/low around a CONSTANT close so the bar's true
        # range is controlled exactly by `width`, independent of any
        # close-to-close jump term compute_true_range also considers.
        closes = [100.0] * (n_quiet + n_loud)
        highs = [100.0 + w / 2 for w in widths]
        lows = [100.0 - w / 2 for w in widths]
        df = build_df(ts, closes, highs, lows, closes, [1000] * (n_quiet + n_loud))

        enriched = strat.add_indicators(df)

        check("high_vol_tercile column exists and is boolean",
              "high_vol_tercile" in enriched.columns and enriched["high_vol_tercile"].dtype == bool)
        check("during ATR/percentile warmup (NaN vol_percentile), high_vol_tercile defaults to False, "
              "never True -- NaN can never accidentally trigger the reduced-size path",
              not enriched.loc[enriched["vol_percentile"].isna(), "high_vol_tercile"].any())
        check("a long, uniformly quiet stretch (well past warmup) is NOT flagged high-vol",
              not enriched["high_vol_tercile"].iloc[40:50].any())
        check("the bar where the loud regime BEGINS is immediately flagged high-vol",
              bool(enriched["high_vol_tercile"].iloc[n_quiet]))
        check("the loud regime stays flagged high-vol throughout",
              enriched["high_vol_tercile"].iloc[n_quiet:].all())
    finally:
        strat.VOL_PERCENTILE_LOOKBACK = original_lookback


def test_volatility_scaled_reduced_usd_never_exceeds_trade_amount_usd():
    """
    Pins the CRITICAL safety constraint directly (see the 2026-08-05
    CLAUDE.md incident this candidate exists to never repeat): whatever
    values .env/the environment currently resolve these to, the reduced
    high-vol-tercile amount must never be larger than the existing flat
    baseline. strategy.py additionally raises ValueError AT IMPORT TIME
    if this is ever violated (see the guard next to
    VOLATILITY_SCALED_REDUCED_USD) -- that's exercised separately, in a
    fresh subprocess, by test_bad_volatility_scaled_env_var_fails_
    at_import below, since mutating env vars and reloading this module
    in-process would leak into every later test in this same run.
    """
    check("VOLATILITY_SCALED_REDUCED_USD does not exceed TRADE_AMOUNT_USD under the current config",
          strat.VOLATILITY_SCALED_REDUCED_USD <= strat.TRADE_AMOUNT_USD,
          f"reduced=${strat.VOLATILITY_SCALED_REDUCED_USD}, baseline=${strat.TRADE_AMOUNT_USD}")
    check("USE_VOLATILITY_SCALED_SIZING defaults off (every untested lever in this project does)",
          strat.USE_VOLATILITY_SCALED_SIZING is False)


def test_bad_volatility_scaled_env_var_fails_at_import():
    """
    Runs strategy.py's import-time guard in an isolated subprocess (env
    vars are read once at module load, so mutating os.environ and
    reloading in-process would silently change every OTHER test's
    module-level constants for the rest of this run). Directly proves
    the 2026-08-05 failure shape -- a sizing constant that's allowed to
    exceed the intended baseline -- cannot happen here even via a bad
    env var, not just "wasn't set that way in this repo's current .env."
    """
    import os
    import subprocess
    import sys

    here = os.path.dirname(os.path.abspath(__file__)) or "."

    env_over_baseline = dict(os.environ)
    env_over_baseline["TRADE_AMOUNT_USD"] = "500"
    env_over_baseline["VOLATILITY_SCALED_REDUCED_USD"] = "600"  # ABOVE the $500 baseline
    result_bad = subprocess.run([sys.executable, "-c", "import strategy"], cwd=here,
                                 env=env_over_baseline, capture_output=True, text=True)
    check("importing with the reduced amount set ABOVE the baseline raises, not silently loads",
          result_bad.returncode != 0 and "must not exceed" in result_bad.stderr,
          f"returncode={result_bad.returncode}, stderr tail: {result_bad.stderr[-400:]}")

    env_ok = dict(os.environ)
    env_ok["TRADE_AMOUNT_USD"] = "500"
    env_ok["VOLATILITY_SCALED_REDUCED_USD"] = "350"
    result_ok = subprocess.run([sys.executable, "-c", "import strategy"], cwd=here,
                                env=env_ok, capture_output=True, text=True)
    check("importing with the reduced amount properly BELOW the baseline loads cleanly",
          result_ok.returncode == 0, f"stderr tail: {result_ok.stderr[-400:]}")


if __name__ == "__main__":
    tests = [
        test_compute_daily_trend_map,
        test_stop_must_be_wider_than_noise,
        test_indicator_precompute_matches_growing_window,
        test_breakout,
        test_smash_day,
        test_gap_continuation,
        test_gap_quality_filter,
        test_ross_hook,
        test_orb,
        test_vwap_reversion,
        test_vwap_reversion_volume_confirms,
        test_rvol_spike,
        test_decide_signal_entry_points_agree,
        test_compute_stop_and_target,
        test_compute_position_size,
        test_vol_percentile_ranks_against_own_trailing_history_only,
        test_high_vol_tercile_wiring_in_add_indicators,
        test_volatility_scaled_reduced_usd_never_exceeds_trade_amount_usd,
        test_bad_volatility_scaled_env_var_fails_at_import,
        test_mean_reversion_turning_confirmation,
    ]
    for t in tests:
        print(f"\n{t.__name__}")
        t()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    else:
        print("All tests passed.")
