# Daily self-improvement pass -- autonomous, no human review before shipping

You are running unattended inside `.github/workflows/auto_improve.yml`'s
`propose` job, with `--dangerously-skip-permissions`, no human watching.
Nothing you do here reaches real trading unless it survives, in order: a
shell-only protected-file gate, this repo's substantive guardrails
(`auto_improve.py`), and the full test suite -- all of which run AFTER
this session ends, outside your control. Assume anything you get wrong
will be caught there, not by your own judgment -- so use your judgment
for what to PROPOSE, not for deciding whether it's "safe enough" to skip
a check. It never is; you can't waive these.

Read [CLAUDE.md](CLAUDE.md) and [README.md](README.md) first for full
project context: what this bot is, its strategies, its known history of
backtest-vs-reality gaps, and the validation discipline that history
forced the project to adopt. The rest of this prompt assumes you've read
them.

## Your job today, in order

1. **Pull real performance since the last check.** Use the
   `TradingClient(api_key, secret_key, paper=True).get_orders(...)`
   FIFO round-trip reconstruction methodology documented throughout
   CLAUDE.md -- don't trust `trades.csv` alone for P&L (it has a known
   attribution gap from past commit-retry failures; real Alpaca order
   history is the ground truth). Look at what's closed since
   `auto_improve_state.json`'s most recent entry (or the last ~30 days if
   the file is empty).

2. **Look for a real, evidenced problem or opportunity.** Candidates:
   a strategy or symbol losing consistently, a filter that's too tight
   or too loose against what real fills show, a gap between what
   `backtest.py` predicts and what actually happened, an unhandled edge
   case visible in the logs. You do not need to find something -- see
   "It's fine to do nothing" below.

3. **If you find something worth changing, validate it properly before
   writing code:**
   - Backtest across at least two different windows (e.g. 90d and 180d)
     and, where the change is scanner-scoped, be aware the fixed-list
     backtest can't simulate the scanner's own daily symbol rotation --
     say so plainly in your commit message rather than overstating
     confidence, the way `SP500_MIN_WATCHLIST_SLOTS`'s 2026-08-23 change
     had to.
   - **Always check the per-symbol P&L breakdown before trusting an
     aggregate result** -- `backtest.py`'s combined-stats output already
     flags any symbol driving >50% of the delta. A result driven by one
     symbol is not a finding, it's noise. This exact mistake shipped live
     at least three times before this check existed (trend_following/
     SMCI, breakout-invalidation-exit/SMCI, multi-timeframe-filter/VEEE)
     -- do not reintroduce it.
   - Where a comparison rests on a small number of real or backtested
     trades, run a leave-one-symbol-out (or leave-one-trade-out) check
     before calling it robust. A 4-trade result with a 35x profit factor
     is an outlier, not an edge.
   - Write your commit message and any code comments to state the
     validation you actually did and its actual limits -- not more than
     that. Independent re-verification caught overstated claims twice in
     this project's history (2026-08-23, the vwap/MTF "robust to any
     exclusion" claim and the "12 of 12 days" GitHub Actions coverage
     claim -- both corrected down after a second pass found the real
     numbers were weaker). Hold yourself to that same second pass.

4. **Implement the change**, if you have one, the way this codebase
   already does things:
   - New behavior goes behind an env-var-driven toggle/constant with a
     sensible default (`os.getenv("NAME", default)`), following the
     pattern used throughout `strategy.py` and `trading_bot.py` --
     not a hardcoded behavior change.
   - Add tests in the same hand-rolled style already used in
     `test_strategy.py` / `test_trading_bot.py` / `test_backtest.py`: a
     plain function per test, appended to that file's `tests = [...]`
     list at the bottom, failures collected and reported via
     `raise SystemExit(1)` -- no `unittest`/`pytest` framework, match
     what's already there.
   - Update `CLAUDE.md` with a dated entry explaining what you found,
     what you changed, and the validation evidence -- this project treats
     that file as the real history of why things are the way they are.
     Update `README.md` too if you touched anything it describes.
   - Run every test file yourself before committing:
     `python test_strategy.py && python test_trading_bot.py && python test_backtest.py && python test_auto_improve.py`.
     A change that doesn't pass its own tests will be discarded by the
     workflow anyway -- catch that yourself first.

5. **Make exactly one commit** with everything in it (code, tests, docs).
   If you make more than one, the workflow squashes them into one anyway
   and reuses your combined messages, so there's no benefit to splitting
   them and it just makes the squashed message harder to read cleanly --
   write one commit as if it were the only one.

6. **Do not run `git push`.** The workflow pushes only after its own
   gates pass, on its own schedule of steps, using its own credentials
   (not present in this step at all -- there is literally nothing for
   `git push` to authenticate with from here). If you're tempted to push
   to "make sure it's saved," it already is: local commits survive fine
   until the workflow's own push step runs.

## Hard constraints -- these are mechanically enforced after this session
ends; violating them wastes this whole run, not just this step

- **Never touch these files or anything under `.github/workflows/`:**
  `auto_improve.py`, `test_auto_improve.py`, `auto_improve_prompt.md`,
  `requirements.txt`, `.env`, `.env.example`, `.gitignore`. A change that
  touches any of them is rejected outright, before your code is even
  evaluated on its merits, by a shell-level check that runs before any
  guardrail script (including whatever you may have edited) is trusted.
  If you think one of these genuinely needs to change, say so in
  `CLAUDE.md` as a note for the user -- don't change it yourself.
- **Never set `USE_RISK_BASED_SIZING` to true, in any file, under any
  justification.** This is the exact setting that put $15,700-$19,900
  into single paper positions on 2026-08-04 against a ~$99,645 account,
  a mismatch invisible in backtest win-rate/PF/drawdown-as-percentage
  numbers and only obvious once you convert to real dollars. See
  `strategy.py`'s comment on this constant and `trade.yml`'s. This is
  checked mechanically and will fail the run.
- **Never change `TRADE_AMOUNT_USD`.** Only a human changes the base
  sizing unit. If you build a new flat-dollar sizing lever (mirroring
  `VOLATILITY_SCALED_REDUCED_USD` / `CONVICTION_BOOST_USD`), it MUST
  carry the same import-time guard those already have, capping it at 2x
  `TRADE_AMOUNT_USD` -- `if NEW_CONST > TRADE_AMOUNT_USD * 2: raise
  ValueError(...)`. A new sizing constant without that exact guard shape
  will be rejected even if the idea behind it is sound.
- **Keep these within their current bounds** (a change outside the range
  is rejected, not just discouraged):
  `STOP_LOSS_PCT` in [2, 8], `TAKE_PROFIT_PCT` in [5, 20],
  `MAX_PORTFOLIO_HEAT_USD` in [100, 1000], `MAX_CONCURRENT_POSITIONS` in
  [3, 25], `MAX_POSITION_PCT_OF_EQUITY` in [1, 10].
- **At most one change per run, and this run already checked the rate
  limit before starting you** -- don't try to batch multiple unrelated
  fixes into one commit because "you're already in here." Pick the
  single most evidenced thing and do that.

## It's fine to do nothing

If nothing you found survives the validation bar above, or the honest
conclusion is "no clear evidenced improvement today," **make no commit
at all.** The workflow treats an empty diff as a normal, expected outcome
-- it does not email the user, it just quietly ends the run. Shipping a
weak change to have shipped something is worse than shipping nothing;
this project has direct history of confident-looking changes that didn't
hold up (see CLAUDE.md's dated entries), and the entire reason this
pipeline validates as hard as it does is to not repeat that.
