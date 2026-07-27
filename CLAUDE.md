# Alpaca Trading Bot — Project Context

Paper-trading bot on Alpaca. Built iteratively with Claude (claude.ai chat)
over many sessions — this file captures the reasoning and history that
isn't already obvious from reading the code, so a fresh session has full
context instead of just inferring from what's on disk.

## Files
- `trading_bot.py` — the live bot (paper trading only)
- `strategy.py` — all decision logic (indicators, signals). Shared by
  the live bot AND the backtester so they can never drift apart.
- `backtest.py` — replays strategy.py against historical data
- `test_strategy.py` — synthetic/engineered-data regression tests for
  strategy.py, no network access needed. Run after any strategy change.
- `test_trading_bot.py` — mocked tests (unittest.mock, no real API calls)
  for trading_bot.py's own control flow: position-cap tracking, the
  portfolio risk cap, daily-risk-state persistence, check_symbol's
  gating order. Covers what test_strategy.py can't (needs a client to mock).
- `export_trades.py` — pulls order history / equity curve to CSV
- `.env` — API keys and settings (never commit or share this)
- `.gitignore` — added once it was noticed `.env` (with real paper keys)
  had nothing stopping it from being committed if this ever becomes a git repo.

## Key architectural decisions, and why
- **Bracket orders (stop-loss/take-profit) always use whole-share qty,
  never fractional/notional.** Alpaca doesn't reliably support fractional
  quantities on bracket/OCO/OTO order classes, regardless of whether the
  underlying stock itself is normally fractionable. This was a real bug
  found and fixed — the original version tried fractional qty on
  brackets and every buy on a fractionable stock would have failed.
- **Backtester filters out pre-market/after-hours bars.** Alpaca's
  historical bar data includes extended hours by default, with no
  request-level flag to exclude it. The live bot only ever trades while
  `clock.is_open` (regular 9:30am-4pm ET session), so an unfiltered
  backtest was testing trades that could never happen live. This was a
  real bug that initially made the strategy look unprofitable when it
  wasn't — 69% of "trades" in one early backtest run were outside real
  market hours.
- **STOP_NEW_ENTRIES_MINUTES_BEFORE_CLOSE (default 90)** exists because
  backtesting showed the last ~90 minutes of the session performing much
  worse than the rest of the day (25% win rate vs 56%) — likely not
  enough runway before the forced end-of-day flatten.
- **Every trade's `entry_reason` is tracked** (breakout / smash_day /
  trend_following / mean_reversion) in backtest CSV exports, specifically
  so individual strategies can be evaluated rather than judging the bot
  as one black box.
- **UPDATE, contradicts the note below**: in the first backtest run
  against the current (much larger) strategy set (TSLA/NVDA/COIN, ~3
  months, 239 trades), COIN was the best-performing symbol (+$67.13),
  TSLA close behind (+$66.45), NVDA the only net loser (-$23.10). The
  earlier "COIN underperforms" finding was from a smaller, earlier
  strategy set and should not be treated as still true — this is exactly
  the kind of thing that can flip when the strategy mix changes.
  ~~COIN has consistently underperformed NVDA/TSLA across multiple
  backtests (e.g. one run: COIN -$176 to -$221 vs NVDA/TSLA roughly flat
  to positive).~~ Kept struck through rather than deleted so it's clear
  this was checked again, not just assumed stale.
- **Leveraged ETFs (SOXL, etc.) are excluded from the scanner by
  default** — they move 2-3x their underlying index structurally, so
  they show up in "biggest movers" scans constantly without anything
  unusual actually happening.
- **strategy.py splits indicator computation from decision logic**
  (`add_indicators()` runs once over a whole price history; `decide_signal_at()`
  just reads already-computed columns at a row). Every indicator here
  (EMA/RSI/ADX/ATR/rolling levels/VWAP/pivots) only ever looks backward,
  so this is mathematically identical to recomputing on every growing
  window — just without redoing the work. This was a real fix, not a
  cosmetic one: the old backtester called the decision function on a
  growing window every bar, silently recomputing every indicator from
  bar zero each time (O(n^2) per symbol). Confirmed equivalent via
  `test_strategy.py`'s regression check (recompute-on-slice vs.
  precompute-once, compared at several indices).
- **trading_bot.py fetches bars for the whole watchlist in ONE batched
  API call** (`StockBarsRequest(symbol_or_symbols=<list>)`) instead of
  one call per symbol. This is what makes checking a much larger/faster-
  refreshing scanner watchlist cheap on API rate limits — total calls
  per cycle stay roughly constant regardless of watchlist size.
- **ATR-based stop-loss/take-profit is opt-in (`USE_ATR_STOPS=false` by
  default), fixed-% stays the default.** Volatility-scaled stops are
  more theoretically sound (tight stops on calm stocks, room to breathe
  on wild ones) but change realized behavior in a way that hadn't been
  backtested yet when added — flip on and backtest first, per this
  project's existing standard of not trusting a change until measured.
- **MAX_CONCURRENT_POSITIONS and MAX_DAILY_LOSS_PCT are live-only, not
  backtestable.** `backtest.py` tests one symbol in isolation and has no
  notion of "how many other positions are open right now" or "today's
  total account P&L" — both are inherently portfolio-level, cross-symbol
  concepts. Don't expect backtest results to reflect these limits.
- **The news filter (`USE_NEWS_FILTER`) counts recent article presence,
  not sentiment.** Deliberately NOT an AI/NLP sentiment score — that
  would add per-symbol latency and cost inside a scan loop that already
  needs to stay fast. It's a simple, fail-open filter (falls back to
  unfiltered candidates if news data is unavailable or would eliminate
  every candidate that cycle) rather than a hard dependency.
- **Found and fixed during testing: the first Ross Hook implementation
  scanned for the 1-2-3 pivot sequence in the wrong direction** (forward
  from an arbitrary "point 1"), which meant by the time price reached a
  breakout bar, the nearest-pivot-low lookup would always land on point
  3 itself, never finding a real point 1/point 2 behind it — so it could
  never fire. Fixed by scanning backward from the current bar (nearest
  pivot low = point 3, then walk back to point 2, then point 1), which
  is also the only version that matches what "the most recent completed
  1-2-3 relevant to right now" actually means. `test_strategy.py` has a
  hand-built pivot sequence that pins this down.

## Strategy sourcing: Oxford Capital Strategies (oxfordstrat.com)
A public-domain strategy review site was used as a source for additional
entry strategies beyond the original three (breakout, trend-following,
mean-reversion). Important context: **almost none of their reviewed
strategies are rated A by their own testing — only ~5 are B-rated, the
rest are C/D.** Only the 5 B-rated ones have been considered so far.
Implemented: **Smash Day Pattern (Type B)**, a Larry Williams reversal
pattern (long side only — this bot doesn't short).

Also now implemented, previously deferred:
- **Gap Pattern (Type A)** — exact Oxford rule text was never fully
  sourced (still true), so `gap_continuation_signal()` is a standard,
  well-documented "gap and go" continuation implementation rather than a
  verified reproduction of their exact rule: gap up >= `GAP_MIN_PCT` vs.
  the prior session's close, entry on the first break above the opening
  bar's high. Needed day-boundary detection first (gaps are a daily
  concept) — `strategy.py` now tracks session date / prior-session-close
  / opening-bar-high as precomputed columns.
- **Ross Hook** — needed swing-point/pivot detection to identify a 1-2-3
  reversal formation, which `compute_pivots()` now provides (centered
  rolling min/max, with an explicit "confirmed `lookback` bars later"
  rule so nothing uses bars that hadn't happened yet at decision time).
  See the "found and fixed during testing" note above — the pivot-chain
  scan direction mattered a lot here.

Beyond the Oxford list, three more **standard, well-documented intraday
patterns** were added (not sourced from Oxford — chosen because they're
each a widely-used, independent signal rather than a variant of what was
already there): **Opening Range Breakout (ORB)**, **VWAP mean-reversion**,
and **relative volume (RVOL) spike**. All three, plus Gap Pattern and
Ross Hook, are individually toggleable via `.env` the same way
`USE_SMASH_DAY_PATTERN` already was.

## First real backtest: findings and what changed
The expanded strategy set's first actual backtest (TSLA/NVDA/COIN, ~3
months ending 2026-07-22, 239 trades with risk management). Full
numbers live in the conversation that produced this; the takeaways:

- **The real split wasn't "with vs. without stop-loss" — it was which
  strategies resolve on a real signal vs. drift to the end-of-day
  flatten.** Per-strategy EOD-flatten share: `vwap_reversion` 100%,
  `smash_day` 82%, `mean_reversion` 79%, `ross_hook` 70%, `breakout`
  58%, `rvol_spike` 44%. Critically, this ISN'T just "drifting to EOD is
  bad" — `vwap_reversion` drifts 100% of the time and had the best win
  rate (80%) in the sample. It's strategy-specific, which is why
  `STOP_LOSS_PCT`/`TAKE_PROFIT_PCT` were deliberately left unchanged
  here rather than uniformly tightened (tightening risked cutting
  `vwap_reversion`'s winners short to fix a problem that's really
  concentrated in `smash_day`/`ross_hook`).
- **`smash_day` (78 trades, -$35.23, 82% EOD-flatten) and `ross_hook`
  (10 trades, -$28.83, worst avg return of any strategy at -0.58%/trade)
  were both net losers with no exit discipline**, so both got flipped to
  off by default (`USE_SMASH_DAY_PATTERN=false`, `USE_ROSS_HOOK=false`
  in `strategy.py`). Code is untouched/still fully toggleable — 78 and
  especially 10 trades on 3 correlated symbols over one window is still
  thin evidence, not proof, so this is reversible, not a deletion.
- **`breakout` fired most often (77/239 trades) but netted almost
  nothing per trade (+0.07% avg)** — it's a loose trigger (new 20-bar
  high + 1.5x volume) that was very likely grabbing bars that a
  stronger, later-checked strategy would otherwise have taken.
  `BREAKOUT_VOLUME_MULTIPLIER` raised 1.5→2.0 (fire less often, more
  selectively), AND priority order in `_decide_from_indicators` was
  reshuffled to check `rvol_spike` → `vwap_reversion` → `orb` →
  `gap_continuation` → `breakout` (was: breakout → gap → smash_day →
  ross_hook → orb → rvol_spike → vwap_reversion) — proven-stronger
  signals now get first claim on a bar.
- **`gap_continuation` fired zero times** in this window — TSLA/NVDA/COIN
  just didn't gap >`GAP_MIN_PCT` (2%) in this particular 3 months. Left
  enabled since "never fired" is "unproven," not "proven bad," unlike
  smash_day/ross_hook above.
- **`backtest.py` no longer runs a "without risk management" comparison**
  — dropped per explicit request; every simulated trade now carries real
  stop-loss/take-profit protection, output is a single
  `backtest_trades.csv` instead of two files.

## Second backtest: live-validated (not just reasoned)
The tuning above was reasoned from a single run, then actually
re-verified live the same session (see "Working style" below for the
"no network access" correction that made this possible) -- 90 days,
6 symbols. This is worth keeping separate from the first section above
because it's a genuinely different kind of evidence: confirmed vs. wrong.

- **Confirmed**: on the original TSLA/NVDA/COIN alone, the retune (smash_day
  off, breakout tightened, priority reordered) improved combined P&L
  from +$110.47 (239 trades, old tuning) to +$126.16 (209 trades, new
  tuning) on the same $1500 notional -- a real, if modest, improvement,
  not just a story fit to the first dataset.
- **Confirmed**: `vwap_reversion` held up as the standout -- 48 trades
  across 6 symbols, 71% win rate, +$213.75, by far the best strategy.
  AMD was the standout SYMBOL: +24.9% ($124.65), 60% win rate.
- **Wrong, and worth being honest about**: `rvol_spike` was promoted to
  first priority based on a 25-trade sample (60% win rate, +$87.50).
  With more bars actually reaching it (155 trades across 6 symbols) its
  edge diluted to a coin flip: 50% win rate, **-$27.82 net**. Small
  samples looking strong and regressing toward mediocre with more data
  is exactly the overfitting risk this project's docs have always
  flagged -- this is a real instance of it, not just a caveat. Priority
  was deliberately NOT reshuffled again over this single additional data
  point (that would just be repeating the same mistake one level up --
  chasing whichever result the last run happened to produce). Needs a
  third, independent data point before touching it again.
- **New, decisive**: symbol universe widened to TSLA/NVDA/COIN/AMD/PLTR/MSTR,
  backtested, then **MSTR was dropped** -- worst symbol in the run
  (-18.6%, -$92.86) with by far the worst max drawdown of anything
  tested (42.1%, vs. 5-23% for everything else). AMD and PLTR stayed
  (both net positive). Current default: `TSLA,NVDA,COIN,AMD,PLTR`.
- **`.env` had explicit overrides for exactly 3 of the settings this
  session retuned** (`SYMBOLS`, `BREAKOUT_VOLUME_MULTIPLIER`,
  `USE_SMASH_DAY_PATTERN`) that silently overrode the new code defaults
  -- the first live run under the "new" tuning was actually still running
  old settings until this was caught and `.env` was corrected. Lesson:
  changing a code default doesn't change behavior for anyone whose `.env`
  already pins that setting explicitly -- always check `.env` before
  trusting that a default change took effect.

## Third pass: risk management overhaul + finding actual loss causes
Prompted by a code review that flagged flat position sizing, no
portfolio-level risk cap, and other structural gaps, plus a direct ask
to find the EXACT reasons the bot was losing money on trades (not just
"which strategies," but why). Grounded in `backtest_trades.csv` from the
second backtest above (362 trades) before any of this was built.

**What the loss analysis actually found** (see conversation history for
the full breakdown, this is the summary):
- The single biggest dollar-loss bucket wasn't stop-losses -- it was
  trades drifting to the end-of-day flatten with no clean exit (98
  trades, -$706.98, more than stop-losses and bad sell signals combined).
- `mean_reversion` had a 44% win rate with symmetric win/loss size --
  buying RSI-oversold with zero confirmation was catching falling knives.
- `rvol_spike` won 52% of the time but lost more per loss (-2.15%) than
  it won per win (+1.89%), and accounted for 13 of 23 stop-loss hits
  (57% of all of them) -- volume alone wasn't enough confirmation.
- Entries clustered overwhelmingly at the 9:30 open (67% of all trades)
  despite 11am-12pm ET having the best win rate (64-65%) and 1-2pm ET
  being a net drag.

**Fixes applied for each**:
- **Risk-based position sizing** (`strategy.compute_position_size`,
  `USE_RISK_BASED_SIZING=true` by default): position size is now `equity
  * RISK_PER_TRADE_PCT / (entry - stop)` instead of a flat dollar amount,
  capped at `MAX_POSITION_PCT_OF_EQUITY`. This is the highest-leverage
  change in this pass -- **combined max drawdown dropped from 6.8% (old
  flat-$500 sizing, 6-symbol backtest) to 1.0%** (new risk-based sizing,
  same symbols) in backtesting. Directly answers the original complaint
  that started this: MSTR alone had hit a 42% single-symbol drawdown
  under flat sizing.
- **`mean_reversion` tightened**: `RSI_OVERSOLD`/`RSI_OVERBOUGHT` 30/70 ->
  25/75, AND the BUY side now requires price to already be turning up
  (`mean_reversion_at`), not just an oversold reading. SELL side (exits)
  deliberately left untouched -- this same function governs exits for
  ANY position held during a choppy regime, not just its own entries,
  and the evidence was specifically about entry quality.
- **`rvol_spike` given a confirmation requirement, then disabled anyway**:
  added `RVOL_MIN_CLOSE_STRENGTH` (requires the spike bar to close in the
  upper third of its own range, not just barely green) specifically to
  fix the stop-loss-heavy pattern above. **Honest result: it didn't
  work.** Re-backtested after the fix: still net negative (-$3,047.66
  across 101 trades, 50% win rate) -- fewer trades, same bad win rate.
  `USE_RVOL_SPIKE` flipped to off by default as a result. This is worth
  remembering as a case where the diagnosis (false starts on volume
  alone) may have been right but the specific fix (require a stronger
  close) wasn't sufficient -- a higher `RVOL_MULTIPLIER` might be a
  better next attempt, not another close-strength tweak.
- **Entry blackout window** (`ENTRY_BLACKOUT_START_MINUTES`/`END_MINUTES`,
  default 210-270 = 1-2pm ET): blocks new entries during the historically
  weak window found above. Enforced by the caller (trading_bot.py /
  backtest.py), same architecture as `STOP_NEW_ENTRIES_MINUTES_BEFORE_CLOSE`
  -- not inside `decide_signal`, which stays pure pattern-recognition.
- **Scanner extension filter** (`SCANNER_MAX_EXTENSION_PCT`, default 50%):
  doesn't fix the scanner's structural momentum-chasing bias (ranking by
  size of today's move necessarily favors stocks that already moved --
  there's no cheap way to catch a move before it starts on Alpaca's free
  data), but filters out the most likely-exhausted candidates rather
  than ranking them #1.

**Other structural fixes from the same review**:
- **`MAX_PORTFOLIO_RISK_PCT`** (default 5%): sums $-at-risk across every
  open position (from their live stop-loss orders) and blocks new
  entries that would push the total over the cap. This is how
  correlated-exposure risk actually gets bounded -- not by trying to
  compute correlation between positions, but because each position's
  risk contribution is already capped individually by risk-based sizing,
  so the aggregate can't blow past the cap regardless of how correlated
  the underlying stocks are.
- **Daily-risk state now persists to disk** (`daily_risk_state.json`):
  found while reviewing the "protections only work while the bot runs"
  concern -- the sharper, more fixable version of that problem wasn't
  "no protection while down" (nothing places trades while down anyway),
  it was that `day_start_equity`/`daily_loss_breaker_tripped` were
  in-memory only, so a crash-and-restart mid-bad-day would silently
  un-trip an already-tripped circuit breaker. Now persisted at the
  moments they change and reloaded at startup.
- **Top-level try/except around the main loop body**: a last-resort
  safety net so one unanticipated exception doesn't kill the whole
  process (and with it, every portfolio-level protection). Existing
  narrower try/excepts (clock, batch fetch, per-symbol checks) are
  unchanged -- this only catches what slips past those.
- **Consolidated position lookups**: `get_all_open_positions()` (one
  `get_all_positions()` call) replaced a separate `get_current_position_qty()`
  call per symbol plus a separate `get_all_open_position_symbols()` call
  -- same data, fewer API calls per cycle regardless of watchlist size.
- **`USE_BREAKOUT` added**: breakout was the only entry strategy without
  an on/off toggle. Consistency fix, not a behavior change (defaults on).
- **Log rotation**: `trading_log.txt` now rotates at ~5MB / keeps 5
  backups instead of growing forever.

**Found and fixed while validating all of the above**: the FIRST
attempt to re-run `backtest.py` after risk-based sizing produced **zero
trades on every symbol**. Root cause: `simulate()` was using
`TRADE_AMOUNT_USD` ($500) as the starting "equity" for the compounding
per-symbol curve -- fine when that number just meant "flat dollars per
trade," nonsensical once reinterpreted as account equity for 1%-risk
sizing (1% of $500 is $5, which doesn't buy 1 share of most stocks at a
normal stop distance, so every trade rounded down to 0 shares and got
silently skipped). Fixed by having `backtest.py` fetch the REAL Alpaca
paper account equity (`get_starting_equity()`) and use that as each
symbol's starting simulated capital instead -- also makes the backtest
more representative of what would actually happen on the real account
it'll run on (confirmed the real paper account is ~$99,982, the default
Alpaca gives new paper accounts, not some number a session guessed at).

**Final validated numbers** (90 days, TSLA/NVDA/COIN/AMD/PLTR, real
$99,982 starting equity per symbol, risk-based sizing, all fixes above
applied): 299 trades, 53% win rate, +$15,995.81 combined (+3.2%),
profit factor 1.36, **combined max drawdown 1.0%**. `vwap_reversion`
remains the standout across all three backtests now (63% win rate this
run). `mean_reversion` flipped from a net loser to net positive
(+$1,510.77) after the tightening. `breakout` also improved further
now that `rvol_spike` isn't absorbing bars ahead of it.

## Known open questions / natural next steps
- Does changing BAR_MINUTES (5min vs 15min vs 30min) actually improve
  win rate, or just change trade frequency? `backtest.py
  --compare-timeframes 5,15,30` was built to test this but hasn't been
  run/analyzed yet as of this file's writing.
- `rvol_spike` is now off by default after two straight net-negative
  backtests, one of them with a targeted (and unsuccessful) fix
  attempt. A higher `RVOL_MULTIPLIER` (stricter volume threshold,
  untried) is a more promising next attempt than another confirmation
  filter, if it's worth revisiting at all.
- `MAX_PORTFOLIO_RISK_PCT`, `MAX_CONCURRENT_POSITIONS`, and the daily
  loss circuit breaker are ALL portfolio-level controls `backtest.py`
  structurally cannot simulate (single-symbol isolation, no shared
  capital/position tracking across symbols) — they're only provable by
  watching the live bot actually run, not by backtesting harder.
- `orb` is regressing toward breakeven as its sample size grows (was the
  best profit-factor performer on ~8-14 trades, now roughly breakeven on
  85) — watch this the same way `rvol_spike`'s early promise didn't hold up.
- `gap_continuation` has fired only 1-3 times total across every backtest
  so far (TSLA/NVDA/COIN/AMD/PLTR rarely gap >`GAP_MIN_PCT` intraday) —
  still genuinely unproven either way, not enough data to judge.
- Does changing BAR_MINUTES (5min vs 15min vs 30min) actually improve
  win rate, or just change trade frequency? `backtest.py
  --compare-timeframes 5,15,30` was built to test this but hasn't been
  run/analyzed yet as of this file's writing.
- Whether more strategies is even the right lever, now that there are
  8 entry strategies plus 2 regime-switch strategies — the per-strategy
  backtest breakdown exists specifically to answer "which of these is
  actually pulling weight" rather than assuming more is better.
- Whether to build more (realistically C-grade) Oxford strategies beyond
  the 5 B-rated ones, vs. refining what's already there.
- Crypto trading (Alpaca supports it, 24/7, same account) was discussed
  as a future direction but deliberately deferred.
- The news filter currently only counts article presence/frequency.
  Sentiment scoring (positive/negative, not just "is anything happening")
  was deliberately deferred — would need an LLM/NLP call in the scan
  loop, adding latency and cost that hasn't been justified yet.
- Eventually: real money. Do NOT flip `paper=True` to `False` in
  trading_bot.py without a much longer paper track record and explicit
  discussion — this hasn't been raised as ready yet. The account this
  bot trades against is a fake-money paper account; `RISK_PER_TRADE_PCT`/
  `MAX_PORTFOLIO_RISK_PCT` sizing math would need fresh scrutiny (not
  just a config toggle) before ever pointing this at a real account.

## Working style established in this project
- Every change gets tested with synthetic/engineered data before being
  called done. **Correction, 2026-07-22**: earlier sessions assumed "this
  sandbox has no network access to Alpaca" -- that was never actually
  verified and turned out to be wrong. The sandbox has normal outbound
  internet access; `.env` already has real paper-trading keys in it, so
  `py backtest.py` (read-only, local-only output) can just be run
  directly. What's still true and still the standard: strategy LOGIC
  changes get proven with synthetic/engineered data first (`test_strategy.py`,
  codified as of this session), since that's how you isolate one specific
  rule and know exactly why it passed or failed -- a live backtest run is
  the next step after that, to see whether the logic actually performs,
  not a replacement for it. Don't assume a live/network limitation exists
  without checking -- it wasted a full round of "I can't verify this"
  hedging that turned out to be unnecessary.
- Backtests and live behavior must use the exact same strategy.py logic
  — never duplicate decision logic between the two.
- Be honest about limitations in output/docs (slippage not modeled,
  overfitting risk, performance characteristics) rather than overstating
  confidence in results.

## 2026-07-24: first full unattended trading day — two silent failures

The bot traded exactly one symbol (TSLA) on its first real day running on
GitHub Actions. Neither cause was visible from the outside: every workflow
run reported **success**, and the daily summary email sent fine. Both bugs
failed silently into a plausible-looking fallback, which is the specific
thing to watch for in this system.

### 1. The scanner had been returning an empty list for two days

`watchlist_state.json` still read `{"active_watchlist": ["SMCI"],
"last_scan_time": "2026-07-22..."}` — a fossil. The fallback path in
`refresh_watchlist_if_needed` deliberately does *not* update
`last_scan_time` (so it retries next cycle), which means a stale
timestamp there is the tell that scans have been failing, not succeeding.
Every run since 07-22 fell back to `SYMBOLS`, and TSLA is simply the
first name in that list.

Root cause, reproduced live rather than guessed: the scan required a
candidate to appear in **both** the top-50 movers and the top-50
most-actives-by-volume. Most-actives is share-count based, so it's
dominated by cheap stocks; the intersection of "biggest % movers" and
"most shares traded" is almost entirely sub-$10 penny stocks — exactly
what `SCANNER_MIN_PRICE=10` then rejects. Measured: 100 candidates → 5 in
both lists → **0** survived the price filter. The two filters were
mutually exclusive by construction, so the scanner could never return
anything. Fixed by making liquidity a **dollar-volume** measurement from
real bars (`SCANNER_MIN_DOLLAR_VOLUME`); most-actives is now only a free
fast path and a degradation fallback, never a gate.

A second bug was hiding behind the first: 14 of the 21 candidates that
then survived were 2x **single-stock** leveraged ETFs ("Tradr 2X Short
NBIS Daily ETF", "GraniteShares 2x Long NBIS Daily ETF") that the
hardcoded `LEVERAGED_ETF_DENYLIST` had never heard of. A ticker denylist
is unmaintainable here — these launch constantly under new symbols — so
detection is now by **asset name pattern** against Alpaca's own metadata
(`is_leveraged_etf`), which is what actually catches them. This matters
beyond tidiness: the fixed 5%/10% stop and target assume ordinary
single-stock volatility, and a 2x product moves through them twice as
fast. That check fails **closed** (an unverifiable symbol is skipped) —
skipping a candidate costs nothing, trading a leveraged ETF by accident
does not.

### 2. GitHub ran the bot 5 times, not 96

`cron: "*/5"` does not mean "every 5 minutes." Actual scheduled runs on
07-24: 14:58, 16:49, 18:10, 19:47, 20:52 UTC — gaps of 65–111 minutes,
and the first landed **88 minutes after the open**, so ORB and gap
continuation (both enabled) never had a chance to fire at all. GitHub
treats scheduled workflows as best-effort and drops them under load;
hammering it with a high-frequency cron makes this worse, not better.

Fixed by inverting the model: the cron no longer *is* the heartbeat, it
just starts a **window**. `trading_bot.py --duration-minutes 150` keeps
one job alive for 2.5h running its own cycle every
`CHECK_INTERVAL_MINUTES`. Since GitHub permits one running + one queued
run per concurrency group, the queued job starts the moment the current
one ends — coverage stays continuous even when most ticks are dropped,
and it degrades gracefully (only one tick needs to land per window). Jobs
start before the open so one is alive at the bell, and exit early once
the session is over rather than holding a runner all night.

### The lesson worth carrying forward
Both failures were invisible because each fell back to something
reasonable-looking and still exited zero. When adding a fallback path,
also add the signal that says the fallback is being used — a stale
`last_scan_time` was the only evidence either bug existed, and only
because the code happens not to update it on failure. "All runs green"
proved nothing about whether the bot was actually working.

### Follow-up: making the next failure loud instead of silent
Both 07-24 bugs shared one property: they degraded into something
plausible and still exited zero. Confidence in the fixes isn't the real
protection — the protection is that a recurrence announces itself.

- `daily_summary.py` now reads `watchlist_state.json` and reports scanner
  health in the daily email. If `last_scan_time` isn't today's date, the
  email says so explicitly and the subject is prefixed `[CHECK ME]`. Fed
  the actual state file from the outage, this produces the warning — so
  the two-day blind spot would have surfaced on day one.
- The email also warns whenever a position is left open overnight, since
  `FLATTEN_BEFORE_CLOSE` should make that impossible; it happening means
  the bot wasn't running in the final 10 minutes.
- `run_for_duration()` was extracted out of `__main__` specifically so the
  market-OPEN path could be tested. Outside trading hours the only live-
  exercisable path is the market-closed early exit, which is not the path
  that matters. It now has coverage for cycling, deadline clamping, early
  exit, and error-survival-but-still-report.

Still unproven until a live session: the scanner against *intraday*
movers (composition differs from after-hours), and GitHub's actual
queuing behavior under the new 150-min/20-min arrangement. Both are
observable in Monday's email rather than requiring a code dive.

## 2026-07-26: logs and trade history now survive the runner
Added `logs/<date>.log` (terminal-identical output, one file per trading
day, pruned after `LOG_RETENTION_DAYS`) and `trades.csv` (one row per
trade with indicator context), both committed back by `trade.yml`.

The gap this closes: `trading_log.txt` was in `.gitignore`, and a GitHub
Actions runner is destroyed when its job ends. So every log the live bot
had ever written was unrecoverable, which is why the 07-24 post-mortem
had to reproduce failures locally rather than read them. The workflow's
commit step uses `if: always()` specifically so a CRASHED run still saves
its log — that's the run whose log you actually need.

`trades.csv` is deliberately NOT a re-derivation of Alpaca's order
history. Alpaca knows what and at what price; it cannot know why. RSI,
ADX, VWAP distance, ATR and relative volume at the decision instant exist
nowhere else once the cycle ends, and they're the only way to answer
"which strategy works in which regime" — the question that actually
drives tuning. `client_order_id` already smuggles the strategy name into
Alpaca, but there's nowhere to put twenty indicator values.

Recording is best-effort by design (`trade_recorder.record_trade` never
raises): a bot that won't trade because it can't write a CSV row is a
worse failure than a missing row.

## 2026-07-27: first losing day (-$171.61), and a hypothesis that failed
10 trades, 30% win rate. vwap_reversion took 8 of them, lost 7, and
accounted for essentially the whole loss while breakout and
gap_continuation were both slightly green. This is the first analysis
done from `trades.csv` rather than by reproducing things locally -- the
indicator context recorded at each entry is what made it solvable.

### The wrong answer (recorded because it was convincing)
Every losing entry had ADX >= 25: VEEE at 44.6/40.7/30.8, TRAX at 25.0
twice, NVDA at 53.6. "Mean reversion in a strong trend is knife-catching"
is textbook, the correlation was perfect, and it was wrong.

A 90-day backtest killed it. Gating vwap_reversion at ADX < 25 takes the
strategy from 56 trades / 66% wins / +$76.61 to 14 trades / 50% /
-$12.54, and halves total return (+11.1% -> +5.9%). High ADX is just as
common in this strategy's winners; one day's losers aren't a sample.
Kept at 50 as a backtest-neutral rail (+10.9%, and vwap_reversion is
fractionally better at +$81.60), explicitly NOT as the fix.

### The actual cause: the stop was inside the noise
VEEE's ATR was 6.0-7.3% PER FIVE-MINUTE BAR against a 5% stop. The stop
sat inside a single bar's normal range, so it was certain to be hit at
random and certain to slip when hit. The day sorts almost perfectly by
this one number:

    VEEE  ATR 6.0-7.3%/bar -> -8.78%, -8.32%, -8.26%
    TRAX  ATR 2.21%/bar    -> -4.21%, -4.16%, -3.28%
    NVDA  ATR 0.50%/bar    -> -1.19%
    QBTS  ATR 1.37%/bar    -> +0.47%
    COIN  ATR 0.63%/bar    -> +1.37%

`MIN_STOP_TO_ATR_RATIO` (2.0) now refuses any entry whose stop isn't at
least 2 ATRs away. It blocks exactly the three VEEE trades ($127 of the
$172) and nothing else from that day, and is backtest-neutral on
megacaps (+10.9% vs +11.1%) because their ATR is 0.5-0.6%/bar. Widening
the stop instead would have been the wrong move -- it converts a 5% risk
into a 15% one. These stocks are simply untradeable with this system.

Why this one generalizes where ADX didn't: it's a property of the
INSTRUMENT rather than of one day's tape, it protects every strategy
rather than one, and it's enforced in `strategy.py` so the backtester
can't flatter a setup the live bot would refuse.

### Also fixed
- 7 of the 10 trades were rapid re-entries into two falling stocks (VEEE
  bought back five minutes after being stopped out, four times total).
  `SYMBOL_COOLDOWN_MINUTES` (60) bans re-entry after any position closes,
  detected from the positions already fetched each cycle so bracket-leg
  fills the bot never sees as orders still count.
- Brackets are now priced off a fresh quote, not the last bar's close.
  A stale close in a fast-falling stock produced
  "stop_loss.stop_price must be <= base_price - 0.01" and Alpaca rejected
  the whole order (VEEE, 15:28).

### Lesson
`trades.csv` paid for itself on day one -- none of this was reachable
from order history alone. But a perfect correlation across one day's
seven losers still pointed at the wrong cause. Backtest the hypothesis
before shipping it, especially when it's textbook enough to feel obvious.

## 2026-07-28: backtesting the universe the bot ACTUALLY trades
Every backtest before this ran on TSLA/NVDA/COIN/AMD/PLTR -- megacaps the
bot barely trades. Its real universe is scanner picks: FBRX, VEEE, PN,
TRAX, QBTS, SMCI, SAFT, RNG. Testing those changed several conclusions.

### The backtester was flattering volatile stocks
It filled stops at exactly the stop price. Real stops are market orders
once triggered and fill at whatever comes next. Measured from the three
live VEEE stop-outs on 07-27, slippage was 0.51-0.54 x ATR every time
(stop 16.56 -> filled 15.90, 16.90 -> 16.31, 15.87 -> 15.33), so
`STOP_SLIPPAGE_ATR_FRACTION` now defaults to 0.5, capped at the bar's low.

This was not a small correction. On scanner picks it moved the whole
system from "+7.7%, profit factor 1.05" to "profit factor 0.90, 27.7%
drawdown". Every backtest number recorded before today is optimistic, and
by more on volatile names than on megacaps.

### It also reversed yesterday's conclusion about MIN_STOP_TO_ATR_RATIO
With perfect fills, the volatility guard looked like it HURT the scanner
universe (+7.7% -> -0.1%). With realistic fills it clearly helps:

    guard off: profit factor 0.90, max drawdown 27.7%, vwap_reversion -$239.21
    guard on:  profit factor 0.95, max drawdown 12.4%, vwap_reversion  +$74.52

The guard was right; the measuring instrument was wrong. Worth
remembering that a backtest disagreeing with live evidence is not
automatically the more trustworthy of the two -- check what the
simulation is assuming first.

### ORB is off by default now
Highest-volume strategy in the system, and it doesn't pay for itself:

    scanner picks: 183 trades, 37% wins, -$474.56
    megacaps:       78 trades, 44% wins,   +$0.73  (noise)

Disabling it improves BOTH universes, which is why it's a default change
rather than a per-universe tweak:
    scanner: profit factor 0.95 -> 1.08, drawdown 12.4% -> 8.8%
    megacap: profit factor 1.37 -> 1.52, drawdown  3.0% -> 2.1%
It also frees capital for breakout ($166 -> $233). ORB looked merely flat
for months because its losers cluster in exactly the fast-moving names
where the old perfect-fill assumption was most wrong.

### Where the system stands, honestly
    scanner picks (real universe): +5.5% / 90d, 47% wins, PF 1.08, DD 8.8%
    megacaps:                     +11.2% / 90d, 55% wins, PF 1.52, DD 2.1%

The bot performs far better on liquid megacaps than on the volatile
stocks its own scanner selects. Profit factor 1.08 is barely above
breakeven -- thin enough that ordinary commissions or a bad week would
erase it. That is a strategy-level question (should the scanner prefer
liquidity over volatility?) rather than a bug, and it is the most
important open question in the project.

Still losing on scanner picks: trend_following (-$274.86, 30% wins). It
can't simply be toggled off -- it's the fallback that also governs exits
-- so gating its ENTRIES separately from its exits is the next thing
worth trying.

### Selection-bias caveat
The scanner list was assembled from names the scanner picked in the last
few days, i.e. stocks known to be volatile NOW. Backtesting them over 90
days has look-ahead bias in symbol selection. Directionally useful,
absolutely not a forecast.
