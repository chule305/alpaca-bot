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

## 2026-07-29: bracket mispricing on entry slippage, missing auto-exits, and a git data-loss incident

+$39.34 on 7 completed trades (86% win rate) -- a good day, but two real
gaps surfaced while checking it, plus one incident that was already fixed
earlier today.

### Real bug: bracket stop/target priced off a stale quote
HURN's entry was priced from a $149.055 quote, submitted 2 seconds later,
and filled at $154.30 (+3.5%). The bracket's stop/target are ABSOLUTE
price levels fixed at submission -- they don't move with the entry, so
the stop stayed at $141.60: 5% below the QUOTE, but 8.2% below the REAL
entry. Sizing wasn't affected here (flat-dollar), but a trader who
thought they were risking 5% were actually exposed to 1.65x that if the
trade had reversed.

Fixed with `reconcile_bracket_with_real_fill`: polls briefly for the
confirmed fill, and if it drifted more than `BRACKET_REPRICE_THRESHOLD_PCT`
(0.3%) from the quote used to price the bracket, replaces both leg orders
(`replace_order_by_id`) so stop/target sit at the intended % from the REAL
fill. Also corrects what gets logged to trades.csv -- it recorded
$149.055, not the $154.30 actually paid.

### Real bug: automatic bracket exits were never recorded
GRMN and MANH both closed via a plain market SELL (not the bracket legs
directly -- those got cancelled, which happens when place_sell_order's own
close_position() call fires from a strategy SELL signal). But most exits
in this system are NOT that -- they're the stop or take-profit LEG filling
on its own, which check_symbol never sees, because it only reacts to
strategy signals, never to fills. That path had never called record_trade
at all. Added `record_auto_exit`, triggered from the same
previously_held-vs-now-held diff that already drives the cooldown: looks
up the most recent closed sell for the symbol (the leg fill) and the most
recent buy before it (to attribute the strategy), and logs it as
STOP_HIT/TARGET_HIT/AUTO_EXIT. Verified against today's real GRMN/MANH
orders.

`extract_strategy` moved from daily_summary.py to trade_recorder.py so
both trading_bot.py and daily_summary.py can share it -- trading_bot.py
must never import daily_summary.py directly, since that module raises
SystemExit at import time if email env vars aren't set.

### Not a new bug: today's missing rows were already-fixed git data loss
GRMN/MANH's exits, CBZ's entry, and HURN's second entry are ALL missing
from trades.csv -- but not because record_trade didn't run. Workflow run
#22 (16:12-18:44 UTC) covers every one of those timestamps; its trading
step succeeded (so all four WERE recorded to that job's local trades.csv)
but its commit step hit the exact git-merge-wedge bug fixed earlier today
(commit e84e6fe) and failed all 5 retries. The whole commit -- correct
data included -- was destroyed with the ephemeral runner. Confirmed via
the Actions API, not guessed. No code fix needed beyond what already
shipped; recorded here so a future "why is data missing" doesn't get
re-diagnosed from scratch.

### What worked today (n=7, one day -- directional only)
    trend_following: 1 trade,           +$29.43 (+6.36%)  -- the outlier
    vwap_reversion:  1 trade,           +$11.15 (+2.41%)
    breakout:        4 trades, 3 wins,   -$1.69  net (small wins, one -1.67% loss)
    gap_continuation: 1 trade,           +$0.45

Consistent with the 90-day backtest's ranking (trend_following and
vwap_reversion ahead of breakout), but one day proves nothing on its own
-- noted as a data point, not a conclusion.

## 2026-07-29: S&P 500 liquidity backstop

Consulted with the user before building: two "drastic improvement"
directions were on the table (broader S&P 500 breadth vs. adding crypto).
Decision -- both, sequenced: S&P 500 breadth now (low-risk extension of
what already exists, and the backtest evidence already points at it),
crypto later as its own larger effort (different asset class, trades
24/7 with no session boundaries, and the current strategies -- VWAP
session-reset, ORB, gap continuation -- are all built around a 9:30-4:00
session and would need real rework, not just a symbol-list change).

Implementation: the movers scan ranks candidates by SIZE OF MOVE, which
structurally excludes S&P 500 megacaps (they rarely move enough in a day
to place in a top-50 gainers/losers list) even though 2026-07-28's 90-day
backtest found this system performs markedly better on them (profit
factor 1.52 vs 1.08 on the scanner's own picks). Rather than replace the
movers scan, `USE_SP500_UNIVERSE` reserves `SP500_MIN_WATCHLIST_SLOTS`
(default 4) watchlist slots specifically for S&P 500 names, ranked by
trailing dollar volume (the property the backtest tied to performance,
not size of move -- that's what the movers scan already optimizes for).
Deliberately skips the news-catalyst filter for these slots: "no recent
news" flags likely noise on a stock that's already moving a lot, but
says nothing meaningful about a megacap on an ordinary quiet day.

Constituents come from a community-maintained CSV mirror
(`datasets/s-and-p-500-companies` on GitHub, served over plain HTTPS, no
auth/rate limit), fetched dynamically per the user's explicit preference
over a hardcoded list -- chosen over scraping Wikipedia's table (fragile
to markup changes) or a paid data API (unnecessary for a list that
changes a handful of times a year). Cached in-process for
`SP500_REFRESH_HOURS` (24h) rather than to disk: a single
`--duration-minutes` job runs many scan cycles, so an in-process cache
avoids refetching within one job, and a fresh job refetching once every
~2.5h is cheap enough that persisting it across jobs wasn't worth another
git-tracked state file. Falls back to the last successfully cached list
on a failed refresh, and to an empty list (skip the backstop that cycle,
not the whole scan) if no cache exists yet.

Found and fixed while wiring this in: three separate early `return []`
statements in the movers pipeline (empty prefiltered list, all-leveraged-
ETF candidates, nothing meeting the dollar-volume bar) would each have
skipped the S&P 500 backstop entirely on exactly the kind of quiet day
it exists to cover. Changed each to fall through with an empty
`qualified`/`prefiltered` instead of returning, so the backstop always
gets a chance regardless of how the movers scan fared. Verified live: a
simulated zero-mover scan still produces a full backstop-only watchlist.

Reserved slots come out of the existing `SCANNER_WATCHLIST_SIZE` budget,
not on top of it -- if the movers scan already filled every slot, the
lowest-ranked movers picks get trimmed to make room, since the whole
point is guaranteeing liquidity representation, not adding to an already
-full list.

## 2026-07-30: three improvement ideas tested, none shipped

User asked when to add crypto, and whether to add more real-time news
sources or tune strategies to catch smaller price moves. Answered the
crypto question with reasoning (see README); tested the other two rather
than taking them on faith, since "seems obviously right" has already
been wrong once this week (the ADX-gate reversal on 2026-07-27/28).

**More news sources: not pursued, not because of cost, because it's not
even testable right now.** backtest.py has no concept of news at all --
the news filter is a LIVE SCANNER concern (which symbols to watch), not
something the backtester simulates (it evaluates entries within a symbol
list already chosen). Validating whether MORE news sources would help
requires reconstructing what the scanner would have picked on past days,
which is a real project (a scanner-replay harness), not a quick check.
Adding paid/rate-limited API dependencies on an untested hypothesis isn't
worth it. If this gets revisited, building that replay harness is the
actual prerequisite, not the news integration itself.

**Catching "tinier" price moves via faster bars: tested, clearly worse.**
Ran BAR_MINUTES=1 against the 90-day baseline on both universes:

    megacaps:       PF 1.52 -> 1.26, drawdown 2.1% -> 5.7%, trades 189->400
    scanner picks:  PF 1.08 -> 1.09 (flat), drawdown 8.8% -> 18.7%, trades 347->674

Scanner-universe total return looks higher at 1-minute bars, but that's
explained by trading almost twice as much capital through twice as many
trades at flat quality, while drawdown more than doubled. Same lesson as
the MIN_STOP_TO_ATR_RATIO finding: finer granularity trades quality for
quantity, because the rules can't reliably separate signal from noise at
that resolution. Not shipped.

**trend_following entry confirmation: tested, mixed, not shipped.** Added
a one-bar confirmation requirement to trend_following's BUY side (same
pattern already used for breakout_at/orb_at), since it was flagged
2026-07-28 as the scanner universe's biggest loser (-$274.86, 30% wins).
Direct before/after, same 90 days:

    scanner picks:  PF 1.10 -> 1.11, drawdown 8.9% -> 7.9%, trend_following -$280.69 -> -$263.17 (52->40 trades)
    megacaps:       PF 1.52 -> 1.45, drawdown 2.1% -> 2.2%, trend_following +$46.77 -> +$27.91 (57->49 trades)

Helps the scanner universe modestly (fewer whipsaws), but on megacaps
trend_following is already a WINNER, and delaying entry by one bar cuts
into real edge, not just noise -- total megacap return drops ~17%. Since
2026-07-29's S&P 500 backstop guarantees megacap exposure on every live
watchlist now, this isn't a hypothetical tradeoff between two separate
symbol lists -- it's a direct hit to real, current, simultaneous
exposure. A small win on one side doesn't justify a meaningfully worse
loss on the other. Reverted; strategy.py is unchanged from 2026-07-29.

If this gets revisited, the fix would need to be conditioned on
something OTHER than "is this a scanner pick" (strategy.py has no
visibility into which pipeline sourced a symbol, and building that
plumbing for one unproven idea isn't justified yet) -- e.g. requiring
confirmation only on weaker crosses (small ADX / small EMA separation)
and letting strong ones through immediately, which is a genuinely
separate, untested hypothesis.

## 2026-07-31: four improvement ideas backtested, one built (multi-timeframe, scanner picks only)

User asked what would make the bot more profitable, explicitly ruling out
scaling position size (the $500/trade cap is a real personal constraint,
not a tuning knob). Researched general trading practice + this project's
own history, proposed four ideas, backtested all four against the same
90-day window on both universes before building anything.

    idea                        megacap (PF/DD/return)          scanner (PF/DD/return)
    baseline                    1.52 / 2.1% / +11.5%             1.10 / 8.9% / +6.5%
    1. earnings blackout        1.56 / 1.9% / +11.8% (marginal)  1.04 / 9.7% / +2.7%  (WORSE)
    2. correlation limits       not P&L-backtestable (see below)
    3. multi-timeframe          1.60 / 1.9% / +6.3%  (mixed)     1.27 / 4.6% / +9.8%  (clear win)
    4. trailing stop            0.91 / 4.7% / -2.4%  (bad)       0.64 / 23.8% / -23.7% (very bad)

**Idea 1 (earnings blackout)**: approximated via Alpaca's own historical
news (no real earnings-calendar API key available) -- keyword-scanned
headlines for "earnings"/"EPS"/"beats estimates" etc. as a proxy for
likely report dates. Helped megacaps marginally, hurt scanner picks
(return cut more than half). Working theory: for the volatile small-caps
this bot trades, "earnings-shaped news" and "the rally the bot is trying
to catch" often overlap, so the blackout removes good trades along with
bad ones. Not built.

**Idea 2 (correlation-aware position limits)**: backtest.py tests every
symbol in complete isolation with its own equity curve -- it has no
concept of concurrent cross-symbol positions at all (this is already
documented in its own module docstring). A real P&L test needs a full
portfolio simulator, not a quick check. What IS real: computed actual
historical correlation on both universes. Megacaps average 0.32
correlation with pairs up to 0.58 (AMD-TSLA) -- genuinely material.
Scanner picks average a lower 0.17 but aren't uniformly safe (FBRX-TRAX
hit 0.74 purely by coincidence of what the scanner picked that week). Not
built -- real evidence, but no backtestable dollar impact without a
bigger rebuild than this warranted yet.

**Idea 4 (trailing stop)**: 2x-ATR trail, no fixed take-profit. Bad
everywhere, badly: scanner picks lost 23.7% instead of gaining 6.5%,
profit factor fell to 0.64. The implementation let winners round-trip
back into losses instead of locking in gains -- a different trail
distance or a hybrid (partial profit-take + trail the remainder) might
behave completely differently, but as tested this is a clear rejection,
not a "needs tuning" maybe.

**Idea 3 (multi-timeframe confirmation) -- built, scanner picks only.**
Requiring the prior day's daily EMA(9) > EMA(21) before taking an
intraday entry: unambiguous win on scanner picks (return, win rate,
profit factor, AND drawdown all improve together -- drawdown roughly
halves). On megacaps it's a real tradeoff, not a clean win: quality per
trade improves (win rate 55%->59%, PF 1.52->1.60, drawdown 2.1%->1.9%)
but total trades roughly halve, so total dollar return drops (+11.5%
-> +6.3%) purely from fewer bets being taken, not worse ones. Since the
2026-07-29 S&P 500 backstop guarantees megacap names are on the SAME live
watchlist as scanner picks every day, "apply everywhere" was never a
real option -- it would trade away proven megacap edge to fix a problem
megacaps don't have. Scoped instead by real S&P 500 membership (reusing
fetch_sp500_symbols(), the exact function built for the backstop): S&P
500 names are always exempt, everything else is gated.

Architecture: `strategy.compute_daily_trend_map(daily_bars) -> {date:
bool}` is the pure, synthetic-data-testable piece (shift-by-one-day logic
proven directly against a hand-computed EMA series in test_strategy.py --
gating today's entry on yesterday's already-closed trend, never today's
still-forming one). Everything OPERATIONAL (fetching daily bars, caching,
checking S&P 500 membership) stays in trading_bot.py/backtest.py, same
split as the rest of the project: strategy.py has zero network I/O by
design, and this doesn't change that. trading_bot.py caches per-symbol
daily trend maps in-process for `DAILY_TREND_REFRESH_HOURS` (4h -- daily
trend only changes once a day, so this doesn't need 5-minute-cycle
freshness); backtest.py does a simpler one-shot fetch per run, since it
doesn't need TTL caching at all.

An unknown/uncached symbol fails CLOSED (blocks the entry), matching
stop_is_wider_than_noise's established philosophy: the cost of skipping
one trade over missing data is near zero, the cost of skipping the
filter's protection over the same missing data is not.

Verified the SHIPPED code (not just the scratch backtest that produced
the comparison table) reproduces the improvement: re-running backtest.py
on the real scanner-pick symbols found the numbers moved slightly from
the original comparison (+7.6%/PF 1.18 vs the originally reported
+9.8%/PF 1.27) -- traced to SMCI actually being a real S&P 500 member
that the shipped code correctly exempts but the quick scratch experiment
didn't check for. Re-ran a same-day, same-code on/off comparison instead
of trusting the earlier number: +4.4%/PF 1.07/DD 8.7% (off) vs
+7.6%/PF 1.18/DD 5.7% (on) -- still a clear, real win, just smaller than
first estimated once the S&P 500 exemption was actually correct rather
than approximate. Megacap trade count came back at exactly 190 -- bit-for
-bit identical to the pre-change baseline -- confirming the filter is a
true no-op there, as designed.

## 2026-07-31: idea 5 built (scanner picks only) -- and a real methodology lesson

Built the vwap_reversion volume-confirmation idea from the 10-ideas round.
The build itself surfaced something worth remembering for every future
"test a strategy tweak" round: **testing a signal change in isolation can
give a different, misleading answer from testing it in the full priority
chain.**

Original test ran vwap_reversion as the ONLY active strategy and compared
"with volume filter" vs "without," both isolated. Looked like an
unambiguous win everywhere -- roughly doubled vwap_reversion's own profit
factor on liquid names (1.74 -> 3.53). Rebuilt it, verified the shipped
code, and the FULL-SYSTEM number (all strategies active, real priority
order) told a different story: a same-moment A/B showed megacap profit
factor going 1.60 -> 1.56 (worse) while scanner picks went 1.18 -> 1.30
(better).

The mechanism: vwap_reversion runs first in priority. Tightening its
entries means fewer of ITS signals fire on any given bar -- but that bar
doesn't just go untraded, it falls through to whichever strategy is next
in the chain (trend_following). On megacaps trend_following is weaker,
so bars that used to go to a now-stricter-but-still-decent vwap_reversion
were instead being picked up by a worse strategy, dragging the combined
result down even though vwap_reversion ITSELF got better in isolation.
On scanner picks this effect either doesn't apply as strongly or is
outweighed by vwap_reversion's own improvement -- net positive there.

Same split shape as the multi-timeframe filter (idea 3, 2026-07-31
earlier entry), and the same fix: scanner-picks-only, S&P 500 names
exempt. Architecture differs from idea 3 though -- the daily-trend filter
needed an external data fetch (daily bars) so it was always going to live
outside strategy.py; this one needs no new data (volume/rvol_avg_volume
are already in add_indicators' output), so the temptation was to just
bake it into vwap_reversion_at directly. Didn't, for the same reason as
idea 3: strategy.py has no way to know "is this symbol S&P 500" without
breaking its pure/network-free design. `vwap_reversion_volume_confirms`
is a separate, standalone pure function in strategy.py; trading_bot.py's
check_symbol and backtest.py's simulate() each call it externally, gated
on reason_key=="vwap_reversion" and non-S&P-500 membership, mirroring
exactly how the multi-timeframe filter's external gate works.

Verified: megacap trades came back at exactly 190 -- bit-for-bit
identical to the filter-off run -- confirming the S&P 500 exemption
makes this a true no-op there. Scanner picks: profit factor 1.18 -> 1.29,
drawdown 5.7% -> 5.0%.

**Takeaway for future strategy-tuning rounds**: an isolated single-
strategy backtest answers "is this strategy better on its own," not "is
the whole system better with this change." Given decide_signal_at is a
strict priority chain where an earlier strategy firing less often hands
bars to whatever's next, any change to one strategy's entry criteria
needs a FULL-CHAIN test before being trusted, not just an isolated one.
Isolated tests are still useful for a first pass (cheap, fast signal on
whether an idea has any merit at all) -- just don't ship off one.

## 2026-07-31: idea 1 built (15-min bars) -- verified safe, held for a clean deploy

Built the 15-minute bar timeframe from the 10-ideas round. Before shipping,
explicitly checked whether anything downstream assumes 5-minute
resolution, since this is a genuinely global change (touches every
indicator, not one function like ideas 3/5):

- Entry blackout window and EOD flatten/stop-new-entries timing: both
  driven by Alpaca's real wall clock (`clock.timestamp`/`clock.next_close`),
  verified via code read -- completely independent of BAR_MINUTES, no
  changes needed.
- Warmup bar count: verified live, get_recent_bars_batch's 10-day lookback
  returns 258-511 bars at 15-min resolution against a 26-bar minimum --
  comfortable margin.
- MIN_STOP_TO_ATR_RATIO's volatility guard: real effect, not a bug.
  15-min ATR runs ~1.8x the 5-min value (measured live: NVDA 0.27%->0.48%,
  TSLA 0.27%->0.50%, TRAX 1.11%->1.92%). The guard (needs >=2x stop/ATR)
  gets meaningfully stricter as a result -- comfortable margin on
  megacaps (18x/19x -> 10x), much tighter on volatile names like TRAX
  (4.5x -> 2.6x, still passing but close). This is likely PART of why the
  scanner-universe backtest improved, not a flaw: it's correctly filtering
  more marginal trades at a resolution where each bar naturally carries
  more noise.

Paired CHECK_INTERVAL_MINUTES (5 -> 15): checking every 5 minutes against
a bar that only closes every 15 achieves nothing but wasted API calls;
reaction speed is gated by the bar close either way.

Results (90-day backtest, verified against the actual shipped defaults,
not just the sweep that first found this):

    megacap: win rate 55% -> 61%, profit factor 1.60 -> 1.87, drawdown 2.0% -> 1.6%
    scanner: win rate 46% -> 53%, profit factor 1.18 -> 1.24, drawdown 5.7% -> 3.0%

Deployment note: user asked to hold the push until after 2026-07-31's
close specifically so today's session (already mid-flight when this was
built, one open VCYT position) stays on 5-minute logic uninterrupted, and
tomorrow starts clean on 15-minute logic -- avoids muddying the
before/after comparison for today's results. Confirmed this is safe
either way: GitHub Actions checks out the repo once per job and doesn't
pull mid-run, and open positions are tracked by Alpaca's own resting
orders, not by anything BAR_MINUTES-dependent -- but held the push per
the user's stated preference for a clean transition, not because it was
technically required.

## 2026-07-31: rechecking the CHECK_INTERVAL_MINUTES change found 2 real bugs

User asked to recheck the code before the held push goes out. Good call --
found two real issues, both stemming from the same root cause: raising
CHECK_INTERVAL_MINUTES (5 -> 15) changed what "a normal cycle's sleep
duration" looks like, and two OTHER constants had been silently assuming
it would stay small.

**Bug 1 (real, safety-relevant): EOD flatten could trigger with ~1 minute
of margin instead of 10.** The sleep calculation at the end of
run_one_cycle (`min(CHECK_INTERVAL_MINUTES * 60, seconds_left_today)`) had
no awareness of FLATTEN_MINUTES_BEFORE_CLOSE. Worst case: a cycle with 16
minutes left in the session sleeps the full 15-minute interval and wakes
with only ~1 minute left -- 9 minutes later than the intended 10-minute
buffer before flattening everything for the day. Verified precisely:
before the fix, that exact scenario left 60 seconds of margin; the
intended buffer is 600. Fixed by extracting the sleep computation into
`compute_next_cycle_sleep()` (also makes it independently unit-testable,
which it wasn't before) and capping it so the bot never sleeps past the
point where the flatten window is supposed to start -- works for ANY
CHECK_INTERVAL_MINUTES, not just today's specific numbers.

**Bug 2 (minor, wasteful not dangerous): LONG_IDLE_SECONDS (900s,
hardcoded) became numerically identical to the new CHECK_INTERVAL_MINUTES
(also 900s).** LONG_IDLE_SECONDS gates an extra "is the session actually
over?" clock check in run_for_duration, meant only for genuinely long
waits (market closed, sleeping up to an hour) -- not normal operating
cycles. With CHECK_INTERVAL_MINUTES now equal to it, a completely normal
open-market cycle started tripping this check every single tick. Not
dangerous (the check just returns False immediately while the market's
open) but pure waste, and it defeated the point of having the threshold
at all. Fixed by deriving it from CHECK_INTERVAL_MINUTES with a margin
(`CHECK_INTERVAL_MINUTES * 60 + 60`) instead of a hardcoded constant, so
it can't silently fall out of sync again if this gets tuned further.

Checked and NOT changed: STOP_NEW_ENTRIES_MINUTES_BEFORE_CLOSE (90 min) has
the same theoretical "could wake up a bit late" exposure as the flatten
bug, but it's a soft preference (a position opened ~14 minutes later than
the ideal 90-minute mark still has ~76 minutes of runway -- nowhere near
the safety-critical territory that made the flatten timing worth a hard
fix). The scanner's dollar-volume liquidity check (`bars_per_session =
(6.5*60)/BAR_MINUTES`) was already written as a ratio that scales with
BAR_MINUTES automatically, not a hardcoded constant -- no bug there.

Both fixes are live-operational only (trading_bot.py's own duration-mode
loop) -- backtest.py has no equivalent "sleep and re-check" concept, so
neither bug touches backtest results or the numbers already reported for
idea 1.

## 2026-08-02: real live win rate by strategy, a second git data-loss incident, and confirming a fix that was already shipped

Asked to list current strategies' real win rate from the last few trading
days. trades.csv alone undercounted: reconstructing round-trips from it
directly found only 5 closed trades in the 2026-07-27 to 07-31 window,
because FLATTEN log rows never carry a fill price (by design -- logged as
submitted-intent, not confirmed fill) and the later reconciliation that's
supposed to backfill the real price didn't fire for most of them. Pulling
the same window straight from Alpaca's own order history (unaffected by
any of this, since it doesn't depend on this repo's git state) found the
real number: **21 closed round-trips**, cross-referenced back to
trades.csv's strategy tags by symbol+day. Real picture: `trend_following`
5 trades/60% win, +$30.81; `breakout` 5/60%, -$17.00; `vwap_reversion`
7/**29%**, **-$81.05** (worst by far); `gap_continuation` 1/100%, +$6.75.
Two trades (AMKR, CBZ) had no strategy tag at all -- see below.

**Root cause of the AMKR/CBZ gap, confirmed (not guessed).** Cross-checking
the GitHub Actions API (`/actions/runs`, `/jobs`, `/check-runs/.../
annotations` -- all readable unauthenticated for a public repo) found both
missing trades fall exactly inside runs whose "Run one check cycle" step
**succeeded** but "Commit updated state, logs and trade history" step
**failed** ("Could not push bot state after 5 attempts"). The trade itself
was real (Alpaca confirms both fills) and record_trade() almost certainly
wrote it locally -- it just never survived to origin/master before the
ephemeral runner was destroyed. This is the same failure class the
2026-07-27 incident already fixed once (union merge driver for
logs/*.log and trades.csv) -- that fix handles the MERGE once reached, but
5 attempts / 75s of total backoff wasn't always enough budget to get
there in the first place. Fixed by widening the retry loop in trade.yml
to 10 attempts with a longer, jittered backoff (`attempt * 8 +
RANDOM % 10`, ~8-9 min total budget) -- affordable because this step only
runs after the 150-minute trading loop, inside the ~15-minute buffer
before the job's 165-minute timeout. Jitter specifically because the
thing being retried against is usually another run of this same
workflow a few minutes ahead or behind -- a fixed backoff formula lets
two runs keep re-colliding on the same schedule instead of spreading out.
Could not confirm the exact git-level error text (raw Actions logs need
an auth token this environment doesn't have) -- the fix is justified by
the step-level failure pattern and the mechanism the retry loop already
exists to handle, not by reading the literal error.

**vwap_reversion's real win rate: already addressed, not yet proven
live.** Ran the ALREADY-SHIPPED volume confirmation filter
(`vwap_reversion_volume_confirms`, built earlier this session as idea 5)
against the real bars for all 7 live vwap_reversion trades in this
window. It would have blocked 5 of 7 (VEEE 7/27, TRAX, PSN, ALNY, RTO) --
3 of those were real losses (-$41.40, -$20.13, -$2.64) and 2 were real
wins it would have missed (+$11.15, +$15.54). Net effect on this exact
sample: -$81.05 -> -$43.57 -- still negative, but roughly half the
damage. The 2 trades it would NOT have blocked (VEEE 7/28, VCYT) both had
strong entry-bar volume (5.2x and 2.8x average) and still lost --
volume confirmation genuinely doesn't explain those two. Did not add a
new filter on top (e.g. tightening VWAP_REVERSION_MAX_ADX) without new
evidence -- the existing ADX=50 threshold is already backtest-tuned with
an explicit "tighter costs money" finding on record, and 2 residual
losses in a 7-trade sample isn't enough to override that. Bottom line:
the fix for this specific problem shipped on 2026-07-31 (commit
b9b5a44) but never got a live trading day under it before this
window closed -- Monday 2026-08-03 is the first real test.

## 2026-08-02: widened the scanner's watchlist, and two ideas rejected with real numbers before writing any code

User asked what to do about "giving the bot a lot more trading options" --
more universes (NASDAQ, gold, etc.) to search through. Checked two
concrete options before touching code:

**A parallel NASDAQ-100 backstop (rejected).** Fetched both the S&P 500
and NASDAQ-100 constituent CSVs and computed the actual overlap: 88 of
101 NASDAQ-100 names (87%) are already S&P 500 members. Building a whole
second backstop (fetch/cache/exempt-from-filters, mirroring
SP500_MIN_WATCHLIST_SLOTS end to end) would buy just 13 genuinely new
symbols: ALNY, ARM, ASML, CCEP, FER, INSM, MELI, MSTR, PDD, SHOP, TEAM,
TRI, ZS. One of those (MSTR) was already deliberately dropped from this
bot's default SYMBOLS list once before (2026-07-xx decision, see task
history). Not enough new coverage to justify the machinery -- rejected.

**Raising SCANNER_CANDIDATE_POOL (impossible, not just rejected).**
Already documented in trading_bot.py: Alpaca's screener endpoints
hard-cap the `top` parameter at 50 server-side and the code already
clamps to it. There's no more raw candidate pool available to widen at
all, regardless of what the env var says.

**What actually shipped: SCANNER_WATCHLIST_SIZE 12 -> 18,
SP500_MIN_WATCHLIST_SLOTS 4 -> 6** (kept proportional so the liquidity
backstop stays ~1/3 of the list). Important honesty check on this one:
it is NOT backtest-validated the way strategy/filter changes normally
are here. backtest.py takes a fixed symbol list as input -- it has no
mechanism to replay what the scanner would historically have picked
day-by-day, so "does a bigger watchlist raise win rate" isn't a
backtestable question with the tooling that exists today. The change
rests on a structural argument only: batched bar-fetching already makes
a bigger watchlist ~free on API calls, more symbols is strictly more
chances for the SAME already-validated strategies and filters to find a
qualifying setup, and per-symbol/portfolio risk caps (MAX_CONCURRENT_
POSITIONS, MAX_PORTFOLIO_RISK_PCT, MAX_DAILY_LOSS_PCT) don't care how
big the watchlist is, only how many positions actually open. Kept the
increase to +50% rather than something larger specifically because that
argument, while reasonable, isn't proof. Tests: 67/67 pass (none of them
exercise scanner watchlist SIZE numerically, only the slot-budget math
with values they patch explicitly, so this was safe to change without
touching test code).

## 2026-08-04: EOD flatten bug found and fixed (two positions stuck open), plus a 7-candidate tuning investigation that found no profit/win-rate improvement

User asked why Monday 2026-08-03's positions were still open and why profit was low, and to fix it and aim for higher profit with a higher win rate. Two separate pieces of work came out of this.

**The flatten bug -- a real, confirmed silent failure.** The 2026-08-03 log reads "EOD FLATTEN: all positions closed, all pending orders cancelled" at 19:50:03 UTC, and trades.csv has matching FLATTEN rows for both MSFT and NVDA. Neither was true. Querying Alpaca's live API directly (not the log, not trades.csv) on 2026-08-04 found both positions still open, with **no closing order of any kind** in either symbol's order history after the original entry -- only the pre-existing stop-loss/take-profit bracket legs had been cancelled. Root cause: `flatten_all_positions()` called `trading_client.close_all_positions(cancel_orders=True)`, whose own docstring says it returns "a list of responses from each closed position containing the status code and order id" -- i.e. a per-symbol failure can sit inside that list without the overall call raising at all, and the function never inspected the list's contents, only whether the call itself threw. Same silent-failure shape as the 2026-07-24 scanner incident (degrading into something that looks successful).

Fixed by switching to `close_position(symbol)` per position -- its docstring says it "will throw an error if the position does not exist," i.e. a failed close now raises for that specific symbol instead of being absorbed into an unchecked bulk response. `flatten_all_positions()` now returns `bool` (true only if every position confirmed closed); the caller only sets `last_flatten_date` on a confirmed success, so a partial failure retries on the next (seconds-away) cycle instead of silently giving up for the rest of the day. Commit `c434883`. The two stuck positions were then closed manually via the same `close_position()` call -- market was closed at diagnosis time, so the orders queued (`status=NEW`) and filled automatically at the next open. Tests: 67/67 still pass, no test directly covered this function by name before.

**The tuning investigation -- honest result: nothing tested improved profit or win rate.** Ran a background workflow: measured a fresh baseline, then tested 7 candidate changes in parallel (each in its own isolated plain-directory copy of the repo -- git worktree isolation was tried first and failed outright, since it requires the *orchestrating* session's own working directory to be a git repo, which it wasn't, even though this repo is; worth remembering if this pattern gets reused from a non-git session directory), then a synthesis pass combined-tested the survivors before finalizing, per this project's own 2026-07-31 lesson that isolated-candidate results can mislead vs. full-priority-chain testing.

Fresh baseline (current .env, unchanged, 90 days, space-separated `py backtest.py` invocation -- comma-joined symbol lists silently fail, argparse's `nargs="*"` treats them as one literal unresolvable symbol, not a list):
- Megacap (TSLA/NVDA/COIN/AMD/PLTR): 105 trades, 61% win rate, PF 1.81, DD 1.6%, +8.4%. `trend_following` carries the result (31 trades, 68% win, +$120.44, >half of total P&L); `vwap_reversion` second (28, 64%, +$52.96); `breakout` weakest per-trade despite firing most (39, 54%, +$6.87, essentially flat); `mean_reversion` never fires (ADX routes trending megacaps to `trend_following` instead).
- Scanner (FBRX/VEEE/PN/TRAX/QBTS/SMCI/SAFT/RNG/INHD/FCUV/ATKR/SRAD): 93 trades, 49% win rate, PF 1.11, DD 1.9%, +1.0%. `breakout` top $ contributor but concentrated almost entirely in QBTS/SMCI; `trend_following` the clear drag (33% win, -$49.31); `vwap_reversion` best win rate (57%) but modest dollars; `mean_reversion` barely fires (2 trades total). SMCI is a standing risk flag: 39 trades (most of any symbol), 18.7% max drawdown (worst in either universe by a wide margin), roughly flat P&L -- a lot of activity for very little edge.

Candidates and verdicts, each against that same baseline:
- **`exempt_vwap_reversion_from_multiframe_filter` -- rejected, and the motivating live anecdote was WRONG.** 2026-08-03's log shows this filter blocking a vwap_reversion BUY on INHD twice, which looked like the filter wrongly gatekeeping the best strategy. 90 real days say otherwise: removing the exemption made the scanner universe worse (PF 1.11 -> 1.03, return +1.0% -> +0.3%) -- the filter was correctly screening bad vwap_reversion setups (concentrated in QBTS), not wrongly blocking good ones. A plausible-sounding, specific, evidence-backed hypothesis from a single day's log was still wrong once actually tested -- same category of lesson as the 2026-07-27 ADX-gate story.
- **`loosen_vwap_volume_confirmation` (1.2x -> 1.0x) -- inconclusive, not adopted.** Scanner return/PF ticked up, but on just 6 new trades, and `vwap_reversion`'s own win rate got WORSE (57% -> 53%) -- the dollar gain is win/loss-size luck in a tiny sample, not better signal quality. Also: FCUV, the specific symbol whose live-blocked vwap_reversion signals (3x on 2026-08-03) motivated this test, had **zero trades of any kind in the entire 90-day window** -- the backtest structurally cannot confirm or deny that specific anecdote either way.
- **`tighten_breakout_further` (2.0x -> 2.5x) -- rejected.** Megacap "improvement" (WR 61%->64%, PF 1.81->1.93) is illusory: same ~$7 total P&L from 38% fewer trades, not better trades. Scanner's dollar jump (+$98->+$145) is 68% attributable to 3 all-win trades on one symbol (QBTS) -- the same small-sample-looks-strong shape that burned `rvol_spike` and the original `orb` reading before.
- **`retest_orb` -- rejected, decisively.** Every metric worse on both universes simultaneously (megacap PF 1.81->1.47, DD 1.6%->2.7%; scanner flips from +1.0% profit to a net loss, PF drops to exactly 1.00). Reconfirms the 2026-07-28 finding that got it disabled, on a fresh, large sample (74 + 94 ORB trades) -- not a small-sample fluke this time.
- **`revive_rvol_spike_stricter` (`RVOL_MULTIPLIER` 3.0 -> 5.0) -- individually positive, rejected on interaction risk.** Scanner alone: return +1.0% -> +3.1%, win rate +3pp, PF 1.11 -> 1.32 -- a real, multi-metric improvement in isolation. But `rvol_spike` sits ahead of `vwap_reversion` in the priority chain, and combined-testing with `risk_based_sizing` showed it visibly cannibalizing `vwap_reversion` (scanner P&L roughly halved, +$475 -> +$243) -- eating into this project's most-trusted strategy for an unproven one's benefit. This exact strategy has now regressed from a promising small sample twice before (a 25-trade sample that diluted to a coin-flip; a targeted close-strength fix that didn't work). This run's sample (12 + 20 trades) is smaller than the one that already regressed once. Not shipped -- flagged as worth live/paper monitoring or a second independent backtest window, matching the "needs a third, independent data point" note already on record from 2026-07-31. Side note: `RVOL_MULTIPLIER=5.0` exactly makes `test_strategy.py`'s `test_rvol_spike` fail -- its fixture hardcodes a 5000/1000=5.0x ratio against the code's strict `>` comparison, a boundary-value coincidence, not a real bug -- would need the fixture adjusted before this value ever ships.
- **`bar_timeframe_5min` / `bar_timeframe_30min` -- both rejected**, reconfirming the existing 15-minute default from both directions. 5-min: win rate and drawdown worse on both universes (SMCI's DD nearly doubles to 39.2%), reversing the already-validated 5->15 decision. 30-min: better ratios but from trading 44-61% less, not from better decisions -- megacap's best strategy (`trend_following`) got worse on both count AND win rate, contradicting the "fewer, higher-conviction" rationale for testing it.
- **`risk_based_sizing` (`USE_RISK_BASED_SIZING` false -> true) -- the only one adopted.** Does NOT move win rate or profit factor -- scanner came back trade-for-trade IDENTICAL (same 93 trades, 49% WR, PF 1.11, down to the same per-symbol/per-strategy dollar figures) between flat and risk-based sizing, and megacap softened only mildly (WR 61%->58%, PF 1.81->1.61). What it does do: cut max drawdown ~75-80% on BOTH universes (1.6%->0.4% megacap, 1.9%->0.4% scanner) -- reconfirming the historical "42%->1.0%" single-symbol finding, this time verified fresh under the CURRENT full strategy mix and on both universes rather than the one the original claim was based on. The live `.env`'s own comment already called `true` "the recommended default"; it had been overridden to flat sizing with no offsetting benefit ever measured. Commit `f25a02d`.

**Important architectural discovery while shipping the sizing change: the live bot does not read the local `.env` at all.** `.github/workflows/trade.yml` hardcodes its own `env:` block (by design, per its own comment -- "so a future code change to a default can't silently change what the live bot does"), and that block had its own explicit `USE_RISK_BASED_SIZING: "false"` that needed updating separately from `.env`. A local `.env` edit alone would have silently done nothing to the deployed bot. Worth remembering for any future tuning change: **the file that actually matters for live behavior is `trade.yml`, not `.env`** -- `.env` only affects local `backtest.py`/manual runs.

**Bottom line, stated plainly to the user rather than oversold:** none of the 7 hypotheses tested produced a genuine, robust profit or win-rate improvement -- every one that looked promising on its own turned out to be a small-sample artifact, a filter that was already doing its job, or a strategy with a documented history of exactly this kind of false start. The one real, shippable result was a risk-management gap (flat sizing carrying uncapped single-symbol drawdown risk for no measured benefit), not a profit one. Consistent with this project's own standing culture: an honest null result on a specific ask is still a correct answer, not a failure to look hard enough.

## 2026-08-05: risk-based sizing reverted after one live day, confirmed position sizes far exceeded what was agreed; BAR_MINUTES re-checked and confirmed correct

User reported the bot "traded with all the money, not the $500 we agreed" and asked for a fix, a check of what else went wrong, and an independent re-check of whether 15-minute bars (vs the original 5-minute) caused the loss.

**The sizing complaint was accurate, confirmed against real Alpaca order history (not trades.csv, not assumptions).** 2026-08-04's fills: UFPT $19,883, MSFT $19,745, PLTR $15,772, AAPL $19,641, AMZN $19,745 -- each roughly $15,700-$19,900 against a ~$99,645 account, vs. the $500 flat sizing the bot ran on for most of this project's life. Root cause: `MAX_POSITION_PCT_OF_EQUITY=25` (set long ago, back when `USE_RISK_BASED_SIZING` was `false` and this constant was completely dormant -- `compute_position_size()` is never even called in that mode) became a real ~$25k-per-position cap the moment 2026-08-04's session flipped `USE_RISK_BASED_SIZING` to `true` on real, honest backtest evidence (see the entry above -- 75-80% max-drawdown reduction, no measured cost to win rate or profit factor). That backtest evidence was correct on its own terms (win rate, profit factor, drawdown-as-%-of-equity all genuinely support risk-based sizing), but none of those metrics surface the ABSOLUTE DOLLAR size of a single position on THIS account, which is what actually violated what the user understood the bot to be doing. Real cost that day: -$176.78 net, mostly one -$1,206.55 loss on UFPT alone -- a single-trade swing that's structurally impossible under $500 flat sizing (max plausible loss per trade there is roughly $25-50 at a normal stop distance).

**Fix: reverted `USE_RISK_BASED_SIZING` to `false`** in both `trade.yml` (the file that actually governs live behavior) and `.env` (for local runs -- was already consistent with `trade.yml` this time, unlike the earlier `SCANNER_WATCHLIST_SIZE` drift on 2026-08-02). Also dropped `MAX_POSITION_PCT_OF_EQUITY` 25 -> 5 defensively, so if risk-based sizing is ever deliberately re-enabled later, it can't silently reach $20k+ single positions again without someone explicitly widening this number with full knowledge of what it means in dollars. Confirmed via `py -c "import trading_bot as tb; print(...)"` that the resolved constants now match (`USE_RISK_BASED_SIZING=False`, `TRADE_AMOUNT_USD=500.0`, `MAX_POSITION_PCT_OF_EQUITY=5.0`). 67/67 tests still pass.

**Other things checked while investigating, in case this looked like it might be the same root cause -- it wasn't:**
- The EOD flatten bug from 2026-08-04 (silent `close_all_positions()` failure, see the entry above) is confirmed fixed and holding: 2026-08-04's log has zero `ERROR` lines, zero `FLATTEN FAILED` messages, and a live Alpaca pull confirms no open positions remain stuck anywhere. Not a new issue, already resolved before this session started.
- 2026-08-03's log independently confirms the flatten bug's exact failure mode as documented: `"EOD FLATTEN: all positions closed, all pending orders cancelled"` logged at 19:50:03 UTC while MSFT and NVDA were, in fact, not closed -- consistent with the prior write-up, no new information, just verified rather than trusted.
- `daily_risk_state.json` / `watchlist_state.json` both look normal -- loss breaker not tripped, watchlist within the current 18-symbol cap.

**BAR_MINUTES (5 vs 15 vs 30) -- independently re-verified, not just trusted from the prior session's writeup.** Ran fresh 90-day backtests at all three timeframes against the current full strategy/filter set, both universes:
- Megacap: 5-min 57% WR / PF 1.62 / DD 0.6%; 15-min (current) 59% WR / PF 1.81 / DD 0.4%; 30-min 53% WR / PF 1.88 / DD 0.3%. 15-min beats 5-min on all three metrics simultaneously -- not a mixed result.
- Scanner: 5-min 47% WR / PF 1.26 / DD 0.8% / +1.2% return (200 trades); 15-min (current) 52% WR / PF 1.23 / DD 0.4% / +0.4% return (95 trades); 30-min 54% WR / PF 1.29 / DD 0.2% / +0.2% return (37 trades). Genuinely mixed here: 5-min's higher RAW return comes from trading roughly 2x more often, not from better decisions -- win rate and drawdown are both worse, the same over-trading shape already rejected for `rvol_spike` and the original `orb` reading.
- Conclusion: kept 15-min. The 2026-08-04 loss was not a bar-timeframe problem -- it was a position-sizing problem. A -1.2% adverse move on UFPT produced a $1,206.55 loss only because the position was $19,883, not $500; the signal itself firing and reversing is normal, expected strategy variance that happens at any timeframe.

## 2026-08-06: research sweep + 6 candidates built, backtested, and merged

User asked to research broad improvements (not just strategies -- signals,
screening, precision, risk, exits, anything) and to actually build and test
what came out of it. Two-phase approach: a 6-way parallel research sweep
(56 real, sourced ideas across signals/precision/screening/risk/exits/meta
angles), then implementation + backtesting of a 6-candidate shortlist
picked from that research, each built independently in its own isolated
plain-directory copy of the repo (git worktree isolation was tried first
and failed for the same reason the 2026-08-04 session already documented:
it needs the *orchestrating* session's own working directory to be a git
repo, which it isn't here either) -- then manually merged back into one
codebase, since 6 independent copies aren't a deployable result on their
own. All numbers below are from real 90-day backtests, not estimates.

**Adopted, ON by default:**
- **Time-of-day-normalized breakout volume
  (`USE_TIME_OF_DAY_VOLUME_NORM`, default true,
  `TIME_OF_DAY_VOLUME_BUCKET_MINUTES=30`).** The strongest result of the
  6. Compares each bar's volume against the historical average for that
  SAME minutes-since-open bucket instead of a flat trailing average.
  Genuine improvement on profit factor, return, AND drawdown on BOTH
  universes: megacap PF 1.78->2.37, DD 1.7%->1.4%, return +7.5%->+8.6%;
  scanner PF 1.15->1.22, DD 1.8%->1.7%, return +1.3%->+1.8%. Re-verified
  a second time after merging alongside the other 5 candidates (not just
  in isolation, per this project's 2026-07-31 lesson) -- numbers held:
  megacap 62% WR/PF 2.32/DD 1.4%/+8.4%, scanner 48% WR/PF 1.21/DD
  1.8%/+1.8%. The mechanism found is the OPPOSITE of the original
  hypothesis: BREAKOUT_LOOKBACK=20 bars at BAR_MINUTES=15 is only a
  5-hour rolling window that doesn't reset per session, so early bars of
  a new session are still mostly averaging in the QUIET back half of the
  PRIOR session -- meaning the flat average was actually the LOOSER bar
  at the open (real volume runs ~2.5x it before any real surge) and the
  STRICTER one at midday, not the reverse. Initially shipped OFF pending
  a longer track record; turned ON the same day (2026-08-06) by explicit
  user decision after seeing both verification runs agree -- ahead of
  this project's usual convention, and worth remembering if this specific
  one doesn't hold up: it's one 90-day window on two small universes, not
  a long track record, and the early flip was a conscious trade-off, not
  a default.
- **Portfolio heat cap (`USE_PORTFOLIO_HEAT_CAP`, default true,
  `MAX_PORTFOLIO_HEAT_USD=200`).** Fixed-dollar aggregate open-risk
  ceiling -- deliberately NOT a %-of-equity cap, to avoid exactly the
  2026-08-05 failure mode (same number, different real dollar meaning
  depending on sizing mode/account size). `MAX_PORTFOLIO_RISK_PCT`
  already existed but only ever applies under `USE_RISK_BASED_SIZING`,
  which is off by default -- so before this, there was literally no
  aggregate risk cap under the bot's actual default configuration.
  Portfolio-construction control, not a single-symbol signal --
  backtest.py simulates each symbol independently with its own capital,
  so there's no P&L before/after for this (same reasoning as
  `MAX_CONCURRENT_POSITIONS` already being live-only). Verified via 3 new
  unit tests instead. Enabled by default because it's purely
  restrictive -- it can only ever block a trade, never add exposure, so
  there's no downside to leaving it on.
- **Sector concentration cap (`USE_SECTOR_CONCENTRATION_CAP`, default
  true, `MAX_POSITIONS_PER_SECTOR=2`).** This is "Idea 2" from
  2026-07-31, revisited -- explicitly not built then for the same
  backtest-can't-measure-it reason, still true today. A hardcoded
  ~55-symbol `SECTOR_MAP` in `trading_bot.py`, fails OPEN (never blocks)
  on unmapped symbols since most scanner picks are small/micro-caps that
  were never going to be in it. Verified via 12 new unit tests. Same
  "purely restrictive, safe to default on" reasoning as the heat cap.

**Built, tested, kept OFF -- real backtest evidence says no:**
- **SPY broad-market regime gate (`USE_SPY_REGIME_GATE`).** Net negative
  on every metric on both universes at once (megacap PF 1.79->1.74,
  scanner PF 1.15->1.12, scanner DD 1.8%->2.6%) -- cut trade count ~1/3
  without the survivors being any higher quality. The idea itself (don't
  fight the broad tape) is reasonable; SPY's own ADX/EMA regime reading
  on this bot's 15-min bars over this window wasn't a clean enough
  signal for "bad time to open a long" to earn its keep.
- **Close-beyond-level breakout confirmation
  (`USE_CLOSE_BEYOND_LEVEL_CONFIRMATION`).** The original hypothesis
  (breakout might be checking HIGH instead of CLOSE) was wrong --
  `breakout_at` already compared CLOSE against a rolling max, always
  has. The in-scope version tested (wick-based HIGH as the level, still
  requiring a CLOSE beyond it) made megacap worse on every metric except
  trade count (PF 1.79->1.66) and was a wash on scanner. Trades that
  stopped qualifying as breakout mostly got reallocated to
  gap_continuation at lower quality, not eliminated.
- **ADX regime-switch hysteresis (`USE_ADX_HYSTERESIS`,
  `ADX_HYSTERESIS_BAND=3`).** Made every metric worse on both universes
  (megacap WR 60%->56%/PF 1.79->1.68, scanner WR 49%->48%/PF
  1.15->1.10), and counterintuitively trade count went UP, not down --
  the sticky band delays regime flips rather than suppressing them, and
  since regime state gates both entries AND exits for open positions, a
  delayed flip cascades into the whole downstream entry/exit sequence
  for every later bar, not just the flip itself.

All 6 kept in the code (toggleable, off where rejected) rather than
deleted, matching this project's existing convention (`USE_RVOL_SPIKE`,
`USE_ROSS_HOOK`) -- the evidence against them is specific to this
90-day window and this exact strategy mix, not proof the ideas can never
work. Full merged test suite: 91/91 pass (67 original + 24 new). Default-
config backtest re-run after merging (all 4 rejected/pending toggles at
their off default) reproduces the pre-merge baseline numbers almost
exactly (100 vs 99 trades megacap, identical WR/PF/DD -- the 1-trade
difference is normal day-to-day drift since backtest.py always uses "last
90 days from today"), confirming the merge introduced no accidental
default-behavior change or cross-candidate interaction.

## 2026-08-09: "horrible week" traced to 2 already-fixed bugs, not a strategy problem; MAX_CONCURRENT_POSITIONS raised 5->10

User reported the past week was bad, asked to find and remove whatever
was losing money, and asked to enable using the full paper account per
trade for higher returns. Reconstructed every round-trip trade since the
bot went live (2026-07-24) from Alpaca's own order history (not
trades.csv alone, which still has real gaps -- see the 2026-08-02 entry)
and analyzed by week/strategy/symbol.

**The numbers told a very different story than "bad week."** Week of
2026-08-04 to 08-07: net -$198.35 -- but $1,206.55 of that is the single
UFPT trade from the 2026-08-05 sizing-bug day (already reverted).
Excluding that one trade, the week was +$1,008.20, led by breakout
(PLTR +$794.13, AAPL +$208.74). The week of 2026-07-27 to 07-31 (which
the user believed was better) was actually WORSE in raw dollars
(-$158.97), root-caused to VEEE/TRAX getting rebought within 2-20
minutes of stopping out (real timestamps confirmed this, not assumed) --
exactly the pattern `SYMBOL_COOLDOWN_MINUTES` was built to stop, and
this happened before that fix was live. Neither week's loss was a live,
current strategy problem -- both trace to bugs already found and fixed
in this project's own history. No strategy was cut, because no current
mistake-maker was found; an honest null result here matters more than
manufacturing a change to look responsive.

**Declined, with the numbers, using the full account per trade.**
Recomputed the UFPT trade (-6.1% move) at three sizes: correct $500
sizing would have cost $30.34; what the sizing bug actually did
(~$19,900) cost the real $1,206.55; the full $99,613 account on that
same trade would have cost $6,044.77. Bigger size doesn't create edge,
it multiplies whatever's already there -- including bugs -- in both
directions. Explained this directly rather than complying, then offered
the safe version of "deploy more capital": more $500 positions running
concurrently (more diversified, same risk per trade) instead of bigger
individual ones.

**Shipped: `MAX_CONCURRENT_POSITIONS` 5 -> 10, `MAX_PORTFOLIO_HEAT_USD`
200 -> 250** (user chose this option explicitly). The heat cap is raised
in lockstep specifically so it doesn't become the binding constraint
before the new position-count cap does -- 250 is exactly 10 positions'
worth of the default $500-at-5%-stop ($25/position) risk, matching the
new slot count 1:1, same math the original 200 used for the old 5-slot
default (also corrected a pre-existing comment error found while doing
this -- 200 was actually 8 positions' worth, not the "4" a prior
session's comment claimed; arithmetic error in documentation only, the
code itself was never wrong). Net effect: up to $5,000 total notional
open at once (5% of the account) versus $2,500 (2.5%) before -- more of
the account genuinely at work, without touching per-trade risk or
reintroducing anything resembling the 2026-08-05 failure shape. 91/91
tests still pass.
