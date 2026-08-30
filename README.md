# Adaptive Intraday Paper Trading Bot

This bot watches **many** stocks **while the US stock market is open**,
checks them every few minutes, and automatically decides between several
strategies depending on how each stock is behaving. Every buy is
automatically protected with a stop-loss and take-profit. It only
trades with **fake money** (Alpaca's paper trading account) — nothing
here risks real funds.

**What this bot is, and isn't**: no code can *predict* which stocks
will be profitable — nothing can. What this bot does is look at more
signals (price, volume, gaps, relative strength, news activity), across
more stocks, more often, to find higher-*probability* setups, and cuts
losers faster when it's wrong. That's a real edge in process, not a
guarantee of outcome. Treat every number below the same way this
project always has: as something to verify with `backtest.py`, not
something to trust on description alone.

## The files in this project

- `trading_bot.py` — the live bot. Run this to actually trade (paper money).
- `strategy.py` — the actual decision-making logic (indicators, buy/sell
  rules). Both the live bot and the backtester use this exact same file,
  so a backtest genuinely reflects what the live bot would do.
- `backtest.py` — replays the strategy against historical data in
  minutes instead of days, so you can test ideas fast.
- `test_strategy.py` — synthetic/engineered-data regression tests for
  `strategy.py`. Run with `py test_strategy.py` any time you change
  strategy logic.
- `test_trading_bot.py` — mocked tests (no real API calls) for
  `trading_bot.py`'s own control flow: position-cap tracking, the
  portfolio risk cap, daily-risk-state persistence, entry gating order.
- `export_trades.py` — pulls your full order history and account equity
  curve into CSV files for outside analysis.
- `daily_summary.py` / `test_daily_summary.py` — emails a daily trade
  summary (see "Daily email summary" below) and its tests.
- `trade_recorder.py` / `test_trade_recorder.py` — appends every trade to
  `trades.csv` with the indicator context behind the decision (see "Logs
  and trade history" below), and its tests.
- `.env` — your settings and API keys (never share this file).
- `requirements.txt` — what to install.

## What it actually does, in plain words

Every few minutes, for each stock on the watchlist, the bot checks for
several kinds of opportunity, in this order (first one to fire wins).
Order was tuned after backtesting (see CLAUDE.md) to put strategies with
a demonstrated edge ahead of the weaker, high-frequency ones, and three
strategies are currently OFF by default after showing a real, repeated
loss pattern in backtesting (kept in the code, not deleted, in case more
data tells a different story — see CLAUDE.md for the numbers behind
each call):

1. **VWAP mean-reversion.** Price is stretched well below the session's
   volume-weighted average price (a "must already be turning back up"
   requirement existed here but was removed by default 2026-08-23 after a
   sensitivity sweep found it costing real profit on both universes with
   no real safety benefit — `USE_VWAP_REVERSION_TURN_UP_CONFIRMATION=true`
   restores it). The standout strategy across every backtest so far
   (63-80% win rate each time).
2. **Opening Range Breakout (ORB).** Price breaks above the high of
   the first 15 minutes of the session, once that range is complete.
3. **Gap Pattern (Type A).** Stock gapped up meaningfully at the open
   vs. yesterday's close, and price is pushing through the opening
   bar's high rather than filling the gap back down. Still barely
   tested — these 5 symbols rarely gap enough to trigger it.
4. **A fresh breakout, confirmed by volume.** Price pushed above its
   recent range AND trading volume is unusually high (a stricter volume
   threshold than it started with, after backtesting showed the
   original was too loose). "Unusually high" is judged against the
   historical average for that SAME time of day (`USE_TIME_OF_DAY_
   VOLUME_NORM`, on by default), not a flat trailing average — a bar's
   volume being "unusual" should be judged against what's normal for
   that time of day, since intraday volume follows a well-documented
   U-shape (heaviest at the open, thinnest at midday). 90-day backtest,
   both universes: profit factor, return, and drawdown all improved
   together (see CLAUDE.md for the numbers) — enabled 2026-08-06 on a
   single 90-day window's evidence, sooner than this project's usual
   longer-track-record convention, by explicit user decision.
5. **Relative volume (RVOL) spike** *(off by default — `USE_RVOL_SPIKE=false`)*.
   An unusual volume surge on a green bar that also closes strong (not
   just barely green). Net negative in two straight backtests, the
   second one AFTER a fix attempt aimed specifically at this problem —
   worth knowing that fix didn't work if you re-enable it.
6. **Smash Day reversal** *(off by default — `USE_SMASH_DAY_PATTERN=false`)*.
   A sharp, likely-overdone breakdown that immediately fails and
   reclaims its own high — read as a reversal.
7. **Ross Hook** *(off by default — `USE_ROSS_HOOK=false`)*.
   A classic 1-2-3 swing reversal (low, bounce high, higher low)
   confirmed by a break above the "hook" bar's high.
8. **If none of the above: is this stock trending, or bouncing around
   sideways?** ADX answers that, then picks between **trend-following**
   (fast/slow moving average crossover) or **mean-reversion** (RSI
   oversold/overbought, tightened to require price is already turning
   back up before buying — not just an oversold reading).

This switching is the "adapting to the market" behavior. Exits (selling
a position already held, if the stop-loss/take-profit hasn't already
closed it) are governed by whichever of strategy 8's two approaches is
currently active — this keeps exit logic simple rather than needing to
remember "why" it bought something.

Every strategy above is individually toggleable in `.env`
(`USE_BREAKOUT`, `USE_GAP_PATTERN`, `USE_SMASH_DAY_PATTERN`,
`USE_ROSS_HOOK`, `USE_ORB`, `USE_RVOL_SPIKE`, `USE_VWAP_REVERSION`) if
you want to isolate one for testing.

## Setup

1. Install Python 3.10+, download all the files into one folder.
2. `pip install -r requirements.txt`
3. Rename `.env.example` to `.env`, fill in your **paper** API keys from
   https://app.alpaca.markets/paper/dashboard/overview
4. `py trading_bot.py` to run the live bot.

To stop it, press `Ctrl+C`. Your PC needs to stay on and awake while it
runs (screen can turn off, just not full sleep).

## Running unattended (no PC required): GitHub Actions

`.github/workflows/trade.yml` runs the bot on a schedule in GitHub's cloud
instead of a continuous local process — nothing needs to stay on at
home. State that needs to survive between runs (`watchlist_state.json`,
`daily_risk_state.json`) is committed back to the repo at the end of
every run, so the next run picks up where the last one left off — this
is why those two files are tracked in git rather than ignored.

**Don't trust GitHub's cron to be the bot's heartbeat.** This originally
ran `trading_bot.py --once` on a `*/5` schedule, on the assumption that
meant a check every 5 minutes. It didn't. On 2026-07-24 GitHub fired
**5 of ~96 scheduled ticks** (gaps of 65–111 minutes), and the first one
landed at 14:58 UTC — 88 minutes after the 13:30 open — so the
opening-range and gap strategies never got a chance to run at all.
GitHub documents scheduled workflows as best-effort and drops them under
load, and a high-frequency cron makes that worse rather than better.

So the workflow now runs `trading_bot.py --duration-minutes 150`: one job
stays alive for 2.5 hours and runs its own cycle every
`CHECK_INTERVAL_MINUTES` from inside that window. The cron only decides
how often a new *window* starts. Because GitHub allows one running plus
one queued run per concurrency group, the queued job starts the instant
the current one ends, so coverage stays continuous even when most ticks
are dropped. Jobs start before the open so one is already alive at the
bell, and exit early on their own once the session is over.

You will see a lot of grey **cancelled** runs in the Actions tab — that's
expected and healthy. GitHub keeps only the most recent queued run per
concurrency group and cancels the older ones, which is exactly what stops
a backlog from building up.

`trading_bot.py --once` (a single cycle, then exit) still exists for any
external scheduler that genuinely fires reliably.

Setup (one-time):
1. Create a GitHub repo and push this project to it.
2. Repo Settings → Actions → General → Workflow permissions → **Read and
   write permissions** (needed so the workflow can commit state back).
3. Repo Settings → Secrets and variables → Actions → New repository
   secret: add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` with your real
   paper keys. These are encrypted and never appear in code or logs.
4. That's it — it starts running on the next scheduled tick. Trigger a
   run immediately from the repo's Actions tab (`workflow_dispatch`) to
   verify it works without waiting.

The workflow's `env:` block mirrors this project's `.env` tuning
explicitly (not left to code defaults) specifically so a future code
change can't silently change what the unattended bot does — if you
retune something locally and want it live, update both places.

**Note on GitHub's own indexing**: if a brand-new workflow file doesn't
show up under the repo's Actions tab (or `.../actions/workflows/trade.yml`
says "This workflow does not exist") even though it's clearly committed,
that's a known GitHub quirk with workflows added in a repo's very first
commit. Fix: make any trivial edit to the file and push again — GitHub
re-indexes it and it shows up within seconds.

## Daily email summary

`.github/workflows/daily_summary.yml` runs `daily_summary.py` once a
day after market close and emails what was traded, how much was
invested, and how much was made or lost — pulled directly from Alpaca's
real order history (not from local logs, which don't persist between
GitHub Actions runs). Each buy order is tagged with the strategy that
triggered it (via Alpaca's `client_order_id` field) specifically so
this email can attribute each trade to a strategy, not just show raw
numbers.

Provider-agnostic — the workflow is currently set up for **iCloud Mail**
(`smtp.mail.me.com`), sending an account's own email to itself.

Setup (in addition to the two Alpaca secrets above):
1. On the iCloud account you want to send from/to: go to
   appleid.apple.com → Sign-In and Security → **App-Specific Passwords**
   → generate one, name it anything (e.g. "trading bot"). This is NOT
   your normal Apple ID password, and your normal password won't work
   for this.
2. Repo Settings → Secrets and variables → Actions → add two more
   secrets: `EMAIL_ADDRESS` (your full `@icloud.com` address) and
   `EMAIL_APP_PASSWORD` (the app-specific password from step 1).
3. By default it emails that same address. To send somewhere else,
   also add an `EMAIL_TO` secret with the destination address.
4. Trigger it manually once from the Actions tab to confirm it sends
   before waiting for the schedule.

To use a different provider instead (e.g. Gmail), set `EMAIL_ADDRESS`/
`EMAIL_APP_PASSWORD` to that account's own app password, and override
`EMAIL_SMTP_SERVER` in the workflow (defaults to `smtp.gmail.com` if
unset, so removing the `EMAIL_SMTP_SERVER` line entirely would also
work for Gmail).

## Autonomous self-improvement pipeline (built, currently INACTIVE)

`.github/workflows/auto_improve.yml` is a second, separate pipeline that
researches this bot's own real performance and can ship a code or
parameter change on its own, no human review per change. Briefly ran on
a schedule (21:45 UTC weekdays) starting 2026-08-26; turned back off
2026-08-30 once the real recurring Anthropic API cost of a daily
agentic-coding-session run was priced out — see
[`CLAUDE.md`](CLAUDE.md)'s 2026-08-25/26/30 entries. `workflow_dispatch`
still works for a manual, one-off run any time (e.g. "analyze this week
and improve" on whatever cadence you want to ask for it).

What it does each run: checks real Alpaca performance since any change
it shipped before (auto-reverting one that's lost more than 5% of its
deployed capital over 10+ real trades), then — if nothing needed
reverting and it's not rate-limited — runs a headless
[Claude Code](https://claude.com/claude-code) pipeline
([`auto_improve_prompt.md`](auto_improve_prompt.md)) that may propose,
implement, test, and ship exactly one change. Every proposed change has
to clear, in order: a shell-only check that it didn't touch its own
guardrails or the CI config, a substantive check that it didn't loosen
any hard sizing/risk limit (`auto_improve.py`), and the full test suite
— before it's pushed. You get one email (reusing the same SMTP setup as
the daily summary above) whenever a run actually ships, blocks, or
reverts a change; a quiet day sends nothing.

To run it manually: Actions tab → "Autonomous self-improvement
(guardrailed)" → Run workflow. Still requires an `ANTHROPIC_API_KEY`
secret under Repo Settings → Secrets and variables → Actions — every
other secret it needs already exists from the sections above. Without
it, the run fails cleanly with an actionable error at the very first
relevant step.

## Risk management

### Position sizing: flat dollar amount by default

```
USE_RISK_BASED_SIZING=false
TRADE_AMOUNT_USD=500
```

Every trade spends exactly `TRADE_AMOUNT_USD`, regardless of account
size or how volatile the stock is. Simple, predictable, and what this
bot actually runs live.

### Conviction sizing: a bigger, bounded pool for strong setups

```
USE_CONVICTION_SIZING=true
MAX_DAILY_DEPLOYED_CAPITAL_USD=25000
CONVICTION_ADX_THRESHOLD=35
CONVICTION_VOLUME_RATIO_THRESHOLD=2.5
CONVICTION_TIER1_USD=5000
HIGH_CONVICTION_STRATEGIES=
```

On top of the flat `TRADE_AMOUNT_USD` baseline, a trade can size up when
it shows real, measured confirmation — turned on 2026-08-26 at the
user's explicit request, giving the bot a **shared $25,000/day pool** to
draw from (this is paper-money research, not a live account — see
`CLAUDE.md`'s 2026-08-26 entry for the full reasoning and how it relates
to the risk-based-sizing incident below). A trade earns up to 3 points —
one each for its strategy being in `HIGH_CONVICTION_STRATEGIES` (empty
by default — no strategy has consistent real-world evidence of an edge
yet), its ADX reading clearing `CONVICTION_ADX_THRESHOLD`, and its
volume clearing `CONVICTION_VOLUME_RATIO_THRESHOLD` times its own
trailing average. **Zero points → `TRADE_AMOUNT_USD`. One point →
`CONVICTION_TIER1_USD`. Two or three points → the full remaining pool**
— deliberately reachable on two signals rather than gated behind all
three, since the empty strategy list would otherwise make the top tier
permanently unreachable. `MAX_DAILY_DEPLOYED_CAPITAL_USD` is the hard
ceiling regardless: it applies to every trade that day, not just
boosted ones, and once it's spent, new entries size at $0 (skipped)
until the next trading day resets it.

**Risk-based sizing exists as an option but is off by default, on a
real lesson learned.** With `USE_RISK_BASED_SIZING=true`, each position
is sized so a stop-loss hit costs roughly `RISK_PER_TRADE_PCT` of
current equity instead of a flat dollar amount, and `MAX_POSITION_PCT_
OF_EQUITY` caps notional size regardless. Backtesting genuinely
supports it — it cut a single volatile symbol's (MSTR) 42% drawdown to
1.0% with no measured cost to win rate or profit factor. It was turned
on live on 2026-08-04 on that evidence. One live day later
(2026-08-05), real Alpaca order history showed individual positions of
$15,700–$19,900 on a ~$99,645 account — `MAX_POSITION_PCT_OF_EQUITY=25`
means up to ~25% of your account in ONE position, which is easy to miss
when the backtest numbers are all expressed as win rate / profit factor
/ drawdown-as-a-percentage rather than dollars. That's not a bug in the
sizing math — it did exactly what it was configured to do — it's a gap
between what the config says and what it means on an actual account
size. Reverted to flat sizing the same day. If you turn this back on,
set `MAX_POSITION_PCT_OF_EQUITY` to a number whose dollar value on YOUR
account you're actually comfortable with — don't reuse 25 without doing
that math first.

### Stop-loss / take-profit

Every buy includes an automatic stop-loss and take-profit, submitted
together as one "bracket" order — so the position is protected even if
the bot isn't actively running the moment the price moves.

```
STOP_LOSS_PCT=5
TAKE_PROFIT_PCT=10
```

By default these are fixed percentages, the same for every stock. You
can instead let them scale with each stock's own recent volatility
(ATR — Average True Range) so calm stocks get tighter stops and wild
ones get more room to breathe:

```
USE_ATR_STOPS=false        # opt-in -- off by default, backtest first
ATR_PERIOD=14
ATR_STOP_MULTIPLIER=1.5
ATR_TARGET_MULTIPLIER=3.0
```

Whichever happens first (stop or target) cancels the other. Bracket
orders need whole-share quantities, so buys always round down to the
largest whole number of shares your budget covers.

### Position and account-level limits

```
MAX_CONCURRENT_POSITIONS=18
MAX_PORTFOLIO_RISK_PCT=5.0
MAX_DAILY_LOSS_PCT=3
```

`MAX_CONCURRENT_POSITIONS` caps how many positions can be open at once
— with a broader/faster scanner checking more symbols per cycle, this
stops several strategies firing in the same cycle from over-
concentrating capital. Raised twice on 2026-08-09: first 5 → 10 (more
$500 positions at once, not bigger ones), then 10 → 18 to match
`SCANNER_WATCHLIST_SIZE` exactly, once checking the real account showed
capital was never actually the constraint — this paper account's buying
power is ~$398k on ~$99.6k equity (4x margin), far more than even 18
concurrent $500 positions (~$9k) would ever need. This was always a
deliberate RISK ceiling, not a reflection of available cash.
`MAX_PORTFOLIO_HEAT_USD` below bounds the account's real worst-case
aggregate exposure regardless of this number.

`MAX_PORTFOLIO_RISK_PCT` caps the SUM of $-at-risk across every open
position at once, read from their real stop-loss orders. This is what
actually bounds correlated exposure — the scanner tends to find similar
high-beta names, so "5 positions" isn't automatically "5 independent
bets." Since each position's risk is already capped individually by
risk-based sizing, the aggregate can't blow past this cap no matter how
correlated the underlying stocks turn out to be, without needing to
compute correlation directly.

**`MAX_PORTFOLIO_RISK_PCT` only ever applies when `USE_RISK_BASED_SIZING=true`.**
With this bot's actual default (flat `TRADE_AMOUNT_USD` sizing), there is
otherwise no aggregate cap on open risk at all — a burst of same-cycle
entries in correlated names could stack up unlimited combined risk. This
toggle caps it, in FIXED DOLLARS rather than a percentage, on by default:

```
USE_PORTFOLIO_HEAT_CAP=true
MAX_PORTFOLIO_HEAT_USD=2000
```

A percentage-of-equity ceiling here would risk repeating the exact
2026-08-05 incident above — the same number meaning a harmless amount
under one sizing mode and a large real dollar figure under another.
`MAX_PORTFOLIO_HEAT_USD` sidesteps that by being a plain dollar figure:
before opening a new position, the bot adds up (entry − stop) × qty
across every currently open position PLUS what the new one would add, and
skips the entry if that total would exceed this ceiling. Raised from an
original 450 to 2,000 on 2026-08-26 alongside conviction sizing above —
a single top-tier $25,000 conviction trade carries $1,250 of heat alone
at the standard 5% stop, which the old ceiling would have silently
blocked before that trade was ever evaluated on its own merits. 2,000
covers one such trade plus room for smaller concurrent ones. Still a
hard, fixed-dollar ceiling either way — at most 2% of this account's
equity can be at risk across every open position combined, no matter how
many slots are technically available. On by default (unlike most new
toggles in this project) because it's purely restrictive: it can only
ever block a trade, never add exposure, so there's no downside to
leaving it active.

```
USE_SECTOR_CONCENTRATION_CAP=true
MAX_POSITIONS_PER_SECTOR=2
```

`USE_SECTOR_CONCENTRATION_CAP` (on by default, same reasoning as the heat
cap above) caps how many **open** positions may share the same sector,
checked only when opening a **new** position — it never affects
closing/selling an existing one. `MAX_PORTFOLIO_RISK_PCT`/
`MAX_PORTFOLIO_HEAT_USD` already bound aggregate $-at-risk, but say
nothing about how *concentrated* that risk is: five same-sector
positions can each pass their own risk-cap check individually while the
account is really making one large correlated bet five times over, and
the scanner and the S&P 500 backstop can both independently gravitate
toward the same crowded trade (e.g. several semiconductor names all
showing up as "today's biggest movers" on the same news cycle). Sector
is looked up from a small hardcoded map of common large-caps/ETFs
(`SECTOR_MAP` in `trading_bot.py`) — a symbol with no known sector is
exempt from the cap (fails open) rather than blocked, since most of the
scanner's own picks are small/micro-caps that were never going to be in
a hardcoded map, and the cost of skipping this one check for an unmapped
symbol is far lower than the cost of blocking real trades over
incomplete metadata.

`MAX_DAILY_LOSS_PCT` is a daily circuit breaker: once account equity is
down more than this % versus where it started the trading day, the bot
stops opening **new** positions for the rest of the day. Existing
positions are still watched and exited normally (same pattern as the
pre-close entry cutoff below) — this only blocks fresh entries. This
state is saved to `daily_risk_state.json` and restored on restart, so a
crash mid-bad-day can't silently un-trip an already-tripped breaker.

**Limits on this page (position cap, sector cap, portfolio risk cap,
portfolio heat cap, daily loss breaker) only work while `trading_bot.py`
is actually running** — they live in the bot's own loop, not on Alpaca's
servers (Alpaca has no account-level equivalent to set). The
stop-loss/take-profit on each individual position is different: that's a
real order sitting on Alpaca's side, so it stays protective even if the
bot crashes.

## Handling extreme moves and messy data

A few things that come up specifically because this bot favors volatile
stocks:

- **Trading halts**: exchanges briefly pause a stock that's moving very
  fast (a circuit breaker). If a buy hits this, the bot logs it plainly
  and just tries again next check — no crash, no special action needed
  from you.
- **Leveraged ETFs excluded by default**: funds like SOXL or TQQQ are
  built to move 2-3x their underlying index, so they show up in "biggest
  movers" scans constantly without anything unusual actually happening.
  Set `EXCLUDE_LEVERAGED_ETFS=false` if you want them included anyway.
- **Watchlist survives restarts**: the scanner's current picks are saved
  to `watchlist_state.json`. Restarting the bot no longer forces an
  immediate re-scan — it picks up where it left off, so frequent restarts
  don't cause the watchlist to jump around.

## End-of-day behavior

```
FLATTEN_BEFORE_CLOSE=true
FLATTEN_MINUTES_BEFORE_CLOSE=10
```

By default, the bot closes every open position about 10 minutes before
the market closes, so nothing carries risk overnight. Set to `false` to
let positions ride overnight instead — just know the strategies are
built around short intraday price checks with no special overnight-gap
awareness, and stop-loss/take-profit orders are day orders that expire
at the close too, so overnight positions temporarily lose that
protection until the next session's checks resume.

**New positions also stop earlier than that:**

```
STOP_NEW_ENTRIES_MINUTES_BEFORE_CLOSE=90
```

Backtesting on real data showed the last ~90 minutes of the session
performing meaningfully worse than the rest of the day (-25% win rate
vs. +56% for the rest of the session, in the test that prompted this) —
likely because a position opened that late doesn't have enough runway
before getting forced closed by the end-of-day flatten. The bot still
watches and exits existing positions normally in this window; it just
won't open anything new. Set to `0` to disable this and allow entries
right up until the flatten window.

**There's also a mid-day window it avoids:**

```
ENTRY_BLACKOUT_START_MINUTES=210   # 1:00pm ET
ENTRY_BLACKOUT_END_MINUTES=270     # 2:00pm ET
```

Backtesting found entries between 1-2pm ET were a net drag (negative
combined P&L, mediocre win rate), while 11am-12pm ET was the best-
performing window despite getting far fewer trades — most entries
cluster right at the 9:30 open instead. Same pattern as above: existing
positions are still managed normally, this only blocks new entries.

## Picking which stocks to trade: automatic scanner

By default, the bot scans the market itself every 30 minutes: pulls the
biggest gainers/losers, filters out anything under `SCANNER_MIN_PRICE` or
already moved more than `SCANNER_MAX_EXTENSION_PCT` today, requires at
least `SCANNER_MIN_DOLLAR_VOLUME` of average daily dollar volume (a
liquidity check against thin/manipulated stocks), excludes leveraged and
inverse ETFs, and —
if `USE_NEWS_FILTER` is on — drops candidates with no recent news
behind them (a mover with zero news is more likely thin-volume noise
than a real catalyst; this is a presence/frequency check, **not** AI
sentiment scoring, so it stays fast and simple). It then trades the top
`SCANNER_WATCHLIST_SIZE` survivors. If fewer than
`SCANNER_MIN_WATCHLIST_SIZE` names qualify, the list is topped up from
`SYMBOLS` so a thin scan day doesn't leave the bot with nothing to watch.

**S&P 500 liquidity backstop.** The movers scan above ranks by SIZE OF
MOVE, which structurally excludes megacaps — they rarely move enough in
a day to place in a top-50 gainers/losers list. But a 90-day backtest
(2026-07-28, see CLAUDE.md) found this system performs markedly *better*
on liquid stocks than on its own scanner picks (profit factor 1.52 on
megacaps vs 1.08 on scanner picks). So when `USE_SP500_UNIVERSE` is on
(default), at least `SP500_MIN_WATCHLIST_SLOTS` (default **10**, raised
from 6 on 2026-08-23 — see CLAUDE.md's full-system regression finding:
real trading and an honest, slippage-inclusive backtest both independently
confirmed the backstop is this bot's only demonstrated source of edge,
while the momentum-mover side currently is not, even with 2026-08-23's
own new scanner-quality filters) of the
watchlist are always reserved for S&P 500 names, ranked by trailing
dollar volume — regardless of whether anything in the index happens to
be a big mover that day. The constituent list is fetched from a
community-maintained CSV mirror (`SP500_LIST_URL`) and cached in-process
for `SP500_REFRESH_HOURS` (default 24 — the index changes only a handful
of times a year). These reserved slots come out of the existing
`SCANNER_WATCHLIST_SIZE` budget, not on top of it.

**Multi-timeframe confirmation, scanner picks only** *(off by default —
`USE_MULTI_TIMEFRAME_FILTER=false`, tested and reverted)*. When on, a
non-S&P-500 symbol needs the PRIOR trading day's daily EMA(9) above its
EMA(21) — an uptrend, on the daily timeframe — before the bot will take a
new intraday entry in it. S&P 500 names (whether they came from the
backstop above or the `SYMBOLS` fallback list) are always exempt,
regardless of their own daily trend. This shipped ON from 2026-07-31 on a
single 90-day backtest calling it "an unambiguous win" for scanner picks
— that characterization did not survive a 2026-08-23 re-test with a
mandatory leave-one-symbol-out check across 4 independent comparisons (2
time windows × 2 symbol universes, one of them the bot's actual current
live watchlist rather than a stale fixed list): 3 of 4 robustly favor
OFF, including both tests run against real, currently-traded symbols
(OFF wins there on return, profit factor, AND drawdown at once). Reverted
to off — see CLAUDE.md's 2026-08-23 entry for the full leave-one-out
numbers and why the original result doesn't hold up.

**`vwap_reversion` volume confirmation, scanner picks only.** When
`USE_VWAP_VOLUME_CONFIRMATION` is on (default), a non-S&P-500 symbol's
`vwap_reversion` entry also needs volume at least
`VWAP_REVERSION_MIN_VOLUME_MULT` (default 1.2×) the recent average on the
entry bar — it was the only enabled strategy with no volume check at all.
Same S&P 500 exemption, same reason: first tested with `vwap_reversion`
running alone, which looked like a clean win everywhere (profit factor
1.74 → 3.53 on liquid names). Tested again in the full priority chain and
that turned out to be misleading — tightening `vwap_reversion`'s entries
means fewer of its signals fire, so more bars fall through to
`trend_following` (next in priority), which is weaker on megacaps. Same-
moment A/B on the full system: megacap profit factor 1.60 → 1.56 (worse),
scanner 1.18 → 1.30 (better). Same split shape as the multi-timeframe
filter, same fix.

**SPY regime gate** *(off by default — `USE_SPY_REGIME_GATE=false`, tested and rejected)*.
When on, this vetoes new long entries in EVERY symbol whenever SPY itself
is in a confirmed downtrend on this same `BAR_MINUTES` timeframe — reusing
the exact same ADX/EMA machinery every symbol's own trend-following regime
already runs on (SPY's ADX ≥ `ADX_TREND_THRESHOLD` AND its fast EMA below
its slow EMA), just read off SPY's own bars instead of a new indicator.
The theory — don't fight the broad market's own tape — is reasonable, but
a 90-day backtest (2026-08-06) came back net negative on BOTH universes,
on every combined metric at once:

```
             trades   win rate   total return   profit factor   max DD
megacap:     99→69    60%→58%    +7.6%→+4.9%     1.79→1.74      1.7%→1.8%
scanner:     96→70    49%→46%    +1.3%→+0.9%     1.15→1.12      1.8%→2.6%
```

It cut trade count by roughly a third in both universes without the
surviving trades being any higher quality — it caught real winners along
with the losers it was meant to filter. Kept in code and toggleable, same
as `USE_RVOL_SPIKE`/`USE_ROSS_HOOK` above — re-test before re-enabling,
ideally against a different ADX threshold or a broader index than SPY.

**Sector-relative mean reversion filter** *(off by default —
`USE_SECTOR_RELATIVE_MEAN_REVERSION=false`, inconclusive so far)*. When
on, a `mean_reversion` entry (RSI-oversold-and-turning-up) also requires
the candidate's own return over the last `SECTOR_RELATIVE_LOOKBACK_BARS`
bars (default: `RSI_PERIOD`, the same window the oversold read itself is
judging) to trail its own sector's SPDR ETF's return over that same
window by at least `SECTOR_RELATIVE_MIN_UNDERPERFORMANCE_PCT` (default
2.0 percentage points) — sourced from short-term-reversal research
(Avellaneda & Lee; Quantpedia): a stock reading oversold in isolation is
weaker evidence than one that's genuinely lagging its own peer group over
the same window, not just moving with an ordinary soft sector day. Sector
→ ETF uses the same `SECTOR_MAP` as the concentration cap above, mapped
to the standard SPDR for that GICS sector (XLK, XLF, XLE, XLV, XLI, XLP,
XLU, XLY, XLB, XLRE, XLC); a symbol with no known sector, or a sector
with no mapped ETF, is exempt (fails open), same philosophy as the
concentration cap's own unmapped-symbol handling.

Not the same idea as the SPY regime gate above, despite both comparing
against an external reference series — that gate vetoes *every* symbol on
one binary market-wide read; this one only ever touches `mean_reversion`
entries and asks a comparative question (this stock vs. its own sector,
same window) rather than a directional one about the whole tape. Worth
its own honest test rather than assuming the SPY result predicts this
one — and it got one: a 90-day backtest (2026-08-11) came back byte-for-
byte identical ON vs. OFF on both universes' combined numbers, because
`mean_reversion` is a rare entry in this bot's priority chain (0 trades
in the megacap universe, 2 in the scanner universe, over the whole
window). Of those 2, one (SRAD) has no `SECTOR_MAP` entry and failed open
regardless of threshold; the other (SMCI, a loser) really was evaluated —
sweeping the threshold confirmed its actual underperformance vs. XLK that
window was between 3–5 percentage points, so it narrowly cleared the
2.0pp default and traded anyway. The filter is doing real,
threshold-sensitive work; there just isn't a large enough sample yet
(one single evaluable trade) to say whether 2.0pp is well-calibrated.
Kept in code and toggleable — re-test once `mean_reversion` fires often
enough, on a symbol set `SECTOR_MAP` actually covers, to produce a real
sample.

**Gap-quality volume filter** *(off by default —
`USE_GAP_QUALITY_FILTER=false`, net negative in testing)*. When on,
`gap_continuation` entries (see Gap Pattern above) also require the gap
day's own opening-bar volume to clear `GAP_QUALITY_VOLUME_MULT` (default
1.5x) times this same symbol's own historical opening-bar volume for that
time-of-day bucket — sourced from overnight/intraday return-decomposition
research (Lou, Polk & Skouras; Cooper, Cliff & Gulen): gaps backed by real
volume are theorized to hold better than thin, sentiment-driven gaps that
tend to fade. A 90-day backtest (2026-08-11) found the opposite in this
window: the filter screened out most `gap_continuation` signals as
intended, but the trades it kept were NOT higher quality than the ones it
removed — combined return fell on both universes (megacap 7.3%→6.1%,
scanner 2.3%→2.1%) and `gap_continuation`'s own win rate dropped (megacap
54%→43%, scanner 59%→50%). `gap_continuation` was already this bot's
least-tested strategy, and the filter roughly halves its trade count
again, so this reads as a real negative in a thin sample, not a settled
verdict — kept in code and toggleable, worth retesting with more data or a
different threshold rather than assumed permanently dead.

**Volatility-scaled sizing** *(off by default —
`USE_VOLATILITY_SCALED_SIZING=false`, real drawdown reduction but a real
dollar cost)*. When on, a symbol's own high realized-volatility tercile
(ranked against that SAME symbol's trailing 90 bars of ATR-as-%-of-price —
never cross-symbol) trades at `VOLATILITY_SCALED_REDUCED_USD` (default
$350, a 30% cut) instead of the flat `TRADE_AMOUNT_USD` — sourced from
Moreira & Muir (2017, "Volatility-Managed Portfolios"): scaling exposure
down when trailing realized vol is high, independent of trend strength
(ADX), improves risk-adjusted returns. Hardened against the exact
2026-08-05 sizing incident above: this is always a flat dollar figure,
never a fraction of equity or of `TRADE_AMOUNT_USD`, and `strategy.py`
raises `ValueError` at import time if it's ever configured above
`TRADE_AMOUNT_USD`, proven by a subprocess test that imports the module
with a deliberately bad env var. A 90-day backtest (2026-08-11) on the
scanner universe (the clean read — no symbol there is priced high enough
to round to 0 shares at $350) found every trade and outcome identical
toggle-off vs. toggle-on, just smaller high-vol-tercile positions: max
drawdown fell 31% (1.6%→1.1%), profit factor and win rate held flat, but
total dollar return fell 18% ($150→$122), because the high-vol tercile
happened to be mildly profitable this window rather than a drag. The
megacap read looked more dramatic (PF 2.10→2.51, drawdown 1.4%→0.8%) but
is confounded by AMD/TSLA's share prices exceeding $350, which rounds 17
of 45 high-vol-tercile trades to 0 shares (skipped, not downsized) rather
than a clean size-scaling effect. Kept in code and toggleable — the
mechanism works as designed on the metric it targets, but isn't free, and
a longer backtest window or a price-aware minimum-share-count guard would
strengthen the case before defaulting it on.

**Breakout invalidation exit** *(off by default —
`USE_BREAKOUT_INVALIDATION_EXIT=false`, real improvement on the scanner
universe)*. Found by reconstructing real Alpaca order history: every open
position only ever exited via its bracket stop/target, the *currently
active* regime strategy's SELL signal (not necessarily related to why the
position was opened), or the end-of-day flatten — there was no "the
breakout itself failed" exit. Real data showed this mattered: all 12 real
winning breakout trades exited via a plain market order, never the
bracket's take-profit leg, meaning every real win got cut short before
reaching its target by something unrelated. When on, a breakout position
exits the moment price closes back below the same level that justified
the entry in the first place (frozen at entry, symmetric with the entry
condition itself) — a live-bot cousin of `open_position_context.json`, a
small state file, tracks which open positions were breakout entries so
this works without adding API calls. A 90-day backtest (2026-08-12,
independently reproduced by a second, adversarial reviewer — not just
self-reported) found the scanner universe's 36 breakout trades (the
meaningful sample) improved: average loss shrank about 20% (-1.96% →
-1.57%) while average win held steady, exactly the asymmetry the real
evidence pointed at — combined portfolio return rose 2.5%→2.8%, profit
factor 1.29→1.35, max drawdown 1.6%→1.4%. The 5-symbol megacap universe
only produced 7 breakout trades in the same window, too few to say
anything. Left off by default since there's no megacap-side evidence yet,
but the scanner-side evidence looked real and reproduced — worth deciding
deliberately rather than defaulting either way blindly.

**Update, 2026-08-12 → 2026-08-23**: enabled live for a few hours on the
strength of the evidence above, then reverted after a 180-day robustness
re-test with a per-symbol P&L breakdown found the ENTIRE claimed 90-day
*and* 180-day scanner-universe improvement was driven by ONE symbol,
SMCI (85-100%+ of the delta on both windows) — "independently reproduced"
had checked a second reviewer's math, not whether the result held with
outliers excluded, which turned out to be the real gap. SMCI has also
appeared in the real live watchlist exactly once in 5 weeks, so the
concentrated benefit had little real expected value regardless. Back to
off by default; see `strategy.py`'s `USE_BREAKOUT_INVALIDATION_EXIT`
comment for the full numbers, and `backtest.py`'s COMBINED section (now
prints a per-symbol P&L breakdown with an automatic outlier warning on
every run) for how to catch this class of mistake going forward.

**Conviction-boost sizing** *(off by default — `USE_CONVICTION_SIZING=false`,
`HIGH_CONVICTION_STRATEGIES` genuinely empty)*. The safe alternative to an
idea explicitly declined: staking a large chunk of the account on a trade
judged "very likely to be really good" — declined because this bot has no
real confidence score, and a similar-shaped lever (%-of-equity sizing)
already caused the real 2026-08-05 incident ($15,700–$19,900 single
positions instead of $500). This is the bounded version: a strategy in
`HIGH_CONVICTION_STRATEGIES` trades `CONVICTION_BOOST_USD` (default $750)
instead of the flat `TRADE_AMOUNT_USD`, hard-capped at import time to
never exceed 2x `TRADE_AMOUNT_USD` (never above $1,000 today) even under a
misconfigured env var — the direct, structural answer to the declined
$90k idea. Composes safely with volatility-scaled sizing above: if a trade
is both in a high-conviction strategy and its own high-volatility tercile,
the size-down always wins over the size-up, verified by deliberately
breaking the precedence and confirming the test suite catches it. Checked
the evidence honestly before building anything — comparing per-strategy
win rates across two backtest universes AND real reconstructed Alpaca
trade history found no strategy with a consistent edge across all three
(`trend_following` was a 64%-win standout on one backtest universe and the
worst performer on the other; `vwap_reversion` looked solid on both
backtests but was the worst real-money performer at 23%) — so
`HIGH_CONVICTION_STRATEGIES` ships genuinely empty, not pre-populated with
a guess. Proved inert the strong way: toggle ON with an empty list produces
byte-identical backtest output to toggle OFF (confirmed via matching MD5
hashes) on both universes — the sizing code path is structurally
unreachable, not just empirically unused today. Also proved the mechanism
actually works when populated (a throwaway test run, never shipped this
way): every affected trade's entry/exit price and timing stayed identical,
only size scaled. This is a ready, tested, safety-capped tool sitting idle
until a strategy earns its way onto the list with real, cross-validated
evidence — not a stub.

**Bar timeframe: 15 minutes, not 5.** `BAR_MINUTES` (and the matching
`CHECK_INTERVAL_MINUTES`) moved from 5 to 15 on 2026-07-31. A 90-day
backtest found fewer, higher-quality decision points beat reacting to
every 5-minute wiggle, improving BOTH universes together — rare enough to
be worth noting, since most tuning changes trade one off against the
other:

```
              win rate    profit factor    max drawdown
megacap:      55% → 61%    1.60 → 1.87      2.0% → 1.6%
scanner:      46% → 53%    1.18 → 1.24      5.7% → 3.0%
```

Verified before shipping that nothing downstream assumes 5-minute bars —
the entry blackout window and end-of-day flatten/stop-new-entries timing
are all driven by Alpaca's real wall clock, not bar boundaries, so they
needed no changes. The one real interaction: `MIN_STOP_TO_ATR_RATIO`'s
volatility guard gets meaningfully stricter at a coarser timeframe
(15-min ATR runs ~1.8x the 5-min value for the same stock), which likely
explains part of the improvement on volatile scanner picks — filtering
out more marginal trades, not a side effect fighting the result.

Two filters here have bitten this project and are worth understanding:

- **Liquidity is measured in dollars, not shares.** The original version
  required a candidate to appear in Alpaca's "most actives by volume"
  list. That list is share-count based, so it's dominated by cheap
  stocks — meaning the only names appearing in *both* the biggest-movers
  list and the most-actives list were sub-$10 penny stocks, which
  `SCANNER_MIN_PRICE` then rejected. The two filters were mutually
  exclusive by construction, so **every scan returned an empty list** and
  the bot silently ran on its fallback `SYMBOLS` list for two days
  (2026-07-22 → 07-24) without anything looking broken.
- **Leveraged ETFs are excluded by asset name, not by ticker.** A
  hardcoded ticker denylist can't keep up: 2x/3x *single-stock* ETFs
  launch constantly under new symbols, and on 2026-07-24 fourteen of the
  scanner's twenty-one surviving candidates were ones the denylist had
  never heard of ("Tradr 2X Short NBIS Daily ETF" and similar). These are
  particularly dangerous here because the stop-loss and take-profit
  percentages assume ordinary single-stock volatility, and a 2x product
  blows through them twice as fast. The name check is the real gate; the
  ticker denylist is just a free first pass.

```
SCANNER_REFRESH_HOURS=0.5
SCANNER_WATCHLIST_SIZE=18
SCANNER_CANDIDATE_POOL=50
SCANNER_MAX_EXTENSION_PCT=50
USE_NEWS_FILTER=true
NEWS_LOOKBACK_HOURS=24
MIN_NEWS_ITEMS=1
```

`SCANNER_CANDIDATE_POOL` is shown at its effective maximum (50) rather
than a round number like 100 — Alpaca's screener endpoints hard-cap the
`top` parameter at 50 server-side and reject anything higher, so setting
this above 50 has no effect (the code clamps it too, but the env var
example shouldn't imply a value that doesn't do anything).

**Worth being honest about**: ranking by size of today's move means the
scanner structurally favors stocks that have ALREADY moved a lot — it's
a momentum-chasing scanner by design, not an early-catch one (there's
no cheap way to spot a move before it starts on Alpaca's free data).
`SCANNER_MAX_EXTENSION_PCT` doesn't fix that; it just avoids the worst
version of it by excluding candidates that are likely already
exhausted rather than ranking them #1.

A stock you're already holding is never abandoned, even if it rotates
off the next scan — the bot keeps managing that position until it's
closed. Set `USE_SCANNER=false` to always trade the fixed `SYMBOLS` list
instead. All price/news data fetched for a scan cycle's whole watchlist
goes out in a small, fixed number of API calls (one batched bar request,
one screener request, one news request) regardless of how many symbols
are being checked — this is what makes a wider, more frequent scan cheap.

## Backtesting: testing ideas in minutes instead of weeks

```
py backtest.py                 # test your .env symbols, last 60 days
py backtest.py TSLA NVDA       # test specific symbols
py backtest.py TSLA --days 90  # a specific time range
```

This replays the exact same logic from `strategy.py` against historical
data, bar by bar, using only data available up to that point (no
lookahead), restricted to genuine 9:30am-4pm ET trading hours (Alpaca's
historical data includes pre-market/after-hours bars by default, which
would otherwise test trades the live bot could never actually place).
Every trade carries the live bot's real stop-loss/take-profit (there
used to be a side-by-side "without risk management" comparison here too
— dropped as unneeded scope; the with-risk-management run is the one
that matters), sized with the same risk-based position sizing the live
bot uses, compounding through a simulated equity curve per symbol that
starts at your REAL Alpaca paper account equity (fetched live) rather
than a guessed number — using $500 as a stand-in for "account equity"
under 1%-risk sizing was tried first and produced zero trades on every
symbol, since $5 (1% of $500) doesn't buy a single share of most stocks
at a normal stop distance. Results are saved to `backtest_trades.csv`,
plus a **per-strategy breakdown** (trades, win rate, total P&L for each
enabled strategy) printed to the console, so individual strategies can
be judged instead of treating the bot as one black box.

**Read before trusting the numbers**: it assumes fills at exact prices
with no slippage (real trading will be a bit worse, especially on
halted or fast-moving stocks), and it's possible to fool yourself by
tuning settings to fit what already happened. Treat results as an upper
bound and a way to rule out bad ideas quickly, not a guarantee. It also
tests each symbol in isolation — it does **not** simulate
`MAX_CONCURRENT_POSITIONS`, `MAX_PORTFOLIO_RISK_PCT`, or the daily loss
circuit breaker, since those are portfolio-level controls that only
make sense across your whole account at once (live-only, see
`trading_bot.py`).

Indicators are computed once per symbol up front instead of being
recomputed from scratch on every bar (the old version's approach), so
this now runs roughly an order of magnitude faster than the original
~20 seconds/symbol/60-days — long ranges and wide symbol lists are
practical to test routinely now.

## Exporting your trade history

```
py export_trades.py
```

Saves `orders_export.csv` (every order Alpaca processed) and
`equity_history_export.csv` (account value over time) into the folder.
Upload both to Claude any time you want a real look at performance.
Note: orders rejected instantly at submission (like a halt) never became
real Alpaca records, so they won't appear here — `trading_log.txt`
remains the record for those.

## Two things worth knowing before you go live later

1. **The old $25,000 "Pattern Day Trader" rule no longer applies.**
   Alpaca and FINRA replaced it in June 2026 with a real-time,
   risk-based margin system. Worth double-checking Alpaca's current
   documentation when you're closer to going live, since it's a fairly
   recent change.
2. **More frequent trading = more chances to be wrong, in both
   directions.** That's not inherently better or worse — it's a
   different risk/reward shape, which is exactly what backtesting and
   paper trading are for.

## What to do next

Several real backtests have happened now (see CLAUDE.md for the full
blow-by-blow), the most recent one validating a risk-management overhaul
prompted by a direct question: "why does this lose money, exactly, and
fix it." Current state, on 90 days / TSLA,NVDA,COIN,AMD,PLTR / real
$99,982 starting equity: **299 trades, 53% win rate, +$15,995.81
combined (+3.2%), profit factor 1.36, max drawdown 1.0%** — down from a
42% single-symbol drawdown under the old flat-dollar position sizing.
That drawdown drop is the single biggest result of this project so far.

Along the way: `mean_reversion` flipped from a net loser (44% win rate)
to net positive after requiring price to actually be turning up before
buying, not just an oversold RSI reading. `rvol_spike` got a targeted
fix (require a strong close, not just a green one) that HONESTLY DIDN'T
WORK — still net negative after the fix, so it's off by default now
too, alongside `smash_day`/`ross_hook`. `vwap_reversion` keeps winning
across every single backtest run so far.

Natural next steps, in order:

1. Run the live bot for real stretches and export results periodically
   — every number above is still a backtest, and this project's own
   standard has always been to not fully trust a backtest until it's
   also been watched live.
2. If you want another crack at `rvol_spike`: try raising
   `RVOL_MULTIPLIER` (a stricter volume threshold) instead of another
   close-strength tweak, since that's what didn't work last time.
3. Watch `orb` — it was the best profit-factor strategy on ~8-14 trades,
   now roughly breakeven on 85. Could be the same "small sample looked
   great" pattern that already happened once with `rvol_spike`.
4. `MAX_PORTFOLIO_RISK_PCT`, `MAX_CONCURRENT_POSITIONS`, and the daily
   loss breaker are NOT provable by backtesting harder (structurally
   can't be — single-symbol isolation). The only way to know they work
   as intended is watching the live bot actually hit them.

## Logs and trade history (post-mortem + research data)

Two files are committed back by the workflow after every run, for the
same reason the state JSON files are: **a GitHub Actions runner is
destroyed when its job ends, so anything not committed back is gone.**
That is not a theoretical concern — diagnosing the 2026-07-24 failures
meant reproducing them locally, because the bot's own logs had already
been thrown away with the runner.

### `logs/<YYYY-MM-DD>.log`
Exactly what you'd have seen in the terminal, one file per trading day,
appended across every run that day. This is the first place to look when
something fails. Old files are pruned after `LOG_RETENTION_DAYS`
(default 30) so the repo can't grow forever.

Read one straight from GitHub without cloning:

```bash
curl -s https://raw.githubusercontent.com/chule305/alpaca-bot/master/logs/2026-07-27.log
```

### `trades.csv`
One append-only row per trade, with the **decision context** attached:
RSI, ADX, ATR, VWAP, EMAs, volume vs. its own average, minutes since the
open, plus the sizing and bracket levels chosen.

This is deliberately not a duplicate of Alpaca's order history. Alpaca
knows *what* was traded and at what price; it has no idea *why*. The
indicator values behind a decision exist only for the instant the bot is
looking at them, and they're what's needed to answer the questions that
actually improve the strategy:

- Does `vwap_reversion` only work when ADX is low (i.e. no strong trend)?
- Are the losers disproportionately entries made in the first 15 minutes?
- Is the ATR stop too tight on high-volatility names?

None of those are answerable from order history alone, which is why this
file exists.

```bash
py -c "import pandas as pd; d=pd.read_csv('trades.csv'); print(d.groupby('strategy')[['notional']].agg(['count','mean']))"
```

Recording is strictly best-effort: every write is wrapped so a disk error
logs a warning and nothing more. A bot that refused to place an order
because it couldn't append a CSV row would be a much worse failure than a
missing row.

Separately, a row can also go missing if the workflow's own "commit
updated state" step fails to push after trading_bot.py already recorded
it locally — the runner is destroyed either way, so anything not pushed
is gone. Confirmed happening twice (2026-07-28, 2026-07-29) when that
step exhausted its retry budget; the trades were real (visible in
Alpaca's own order history) but never made it into this file. The retry
budget was widened on 2026-08-02 to make this rarer, but Alpaca's order
history remains the authoritative record if a `trades.csv` row ever looks
like it's missing — see `export_trades.py`.
