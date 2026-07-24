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
   volume-weighted average price and just turned back up. The standout
   strategy across every backtest so far (63-80% win rate each time).
2. **Opening Range Breakout (ORB).** Price breaks above the high of
   the first 15 minutes of the session, once that range is complete.
3. **Gap Pattern (Type A).** Stock gapped up meaningfully at the open
   vs. yesterday's close, and price is pushing through the opening
   bar's high rather than filling the gap back down. Still barely
   tested — these 5 symbols rarely gap enough to trigger it.
4. **A fresh breakout, confirmed by volume.** Price pushed above its
   recent range AND trading volume is unusually high (a stricter volume
   threshold than it started with, after backtesting showed the
   original was too loose).
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
home. `trading_bot.py --once` runs a single check cycle and exits;
GitHub's scheduler re-invokes it every 5 minutes during market hours.
State that needs to survive between runs (`watchlist_state.json`,
`daily_risk_state.json`) is committed back to the repo at the end of
every run, so the next run picks up where the last one left off — this
is why those two files are tracked in git rather than ignored.

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

Setup (in addition to the two Alpaca secrets above):
1. On the Gmail account you want to send FROM: turn on 2-Step
   Verification (myaccount.google.com/security) if it isn't already on.
2. Create an **App Password**: myaccount.google.com/apppasswords → name
   it anything (e.g. "trading bot") → copy the 16-character password it
   generates. This is NOT your normal Gmail password, and your normal
   password won't work for this.
3. Repo Settings → Secrets and variables → Actions → add two more
   secrets: `EMAIL_ADDRESS` (the Gmail address) and
   `EMAIL_APP_PASSWORD` (the 16-character password from step 2).
4. By default it emails that same address. To send somewhere else,
   also add an `EMAIL_TO` secret with the destination address.
5. Trigger it manually once from the Actions tab to confirm it sends
   before waiting for the schedule.

## Risk management

### Position sizing: risk-based, not a flat dollar amount

```
USE_RISK_BASED_SIZING=true
RISK_PER_TRADE_PCT=1.0
MAX_POSITION_PCT_OF_EQUITY=25
```

Each position is sized so that if the stop-loss is hit, you lose
roughly `RISK_PER_TRADE_PCT` of your CURRENT account equity — not a
flat dollar amount every time. A volatile stock with a wide stop gets
fewer shares than a calm one with a tight stop for the same dollar
risk, and sizing scales automatically as your account grows or shrinks.
`MAX_POSITION_PCT_OF_EQUITY` caps the notional size regardless, so a
very tight stop can't imply an absurdly large position.

**This mattered in practice, not just in theory**: under the old flat
`TRADE_AMOUNT_USD`-per-trade sizing, a single volatile symbol (MSTR) hit
a 42% single-position drawdown in backtesting. Switching to risk-based
sizing dropped the SAME kind of backtest's combined max drawdown to
1.0%. Set `USE_RISK_BASED_SIZING=false` to go back to flat
`TRADE_AMOUNT_USD`-per-trade sizing if you'd rather have that.

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
MAX_CONCURRENT_POSITIONS=5
MAX_PORTFOLIO_RISK_PCT=5.0
MAX_DAILY_LOSS_PCT=3
```

`MAX_CONCURRENT_POSITIONS` caps how many positions can be open at once
— with a broader/faster scanner checking more symbols per cycle, this
stops several strategies firing in the same cycle from over-
concentrating capital.

`MAX_PORTFOLIO_RISK_PCT` caps the SUM of $-at-risk across every open
position at once, read from their real stop-loss orders. This is what
actually bounds correlated exposure — the scanner tends to find similar
high-beta names, so "5 positions" isn't automatically "5 independent
bets." Since each position's risk is already capped individually by
risk-based sizing, the aggregate can't blow past this cap no matter how
correlated the underlying stocks turn out to be, without needing to
compute correlation directly.

`MAX_DAILY_LOSS_PCT` is a daily circuit breaker: once account equity is
down more than this % versus where it started the trading day, the bot
stops opening **new** positions for the rest of the day. Existing
positions are still watched and exited normally (same pattern as the
pre-close entry cutoff below) — this only blocks fresh entries. This
state is saved to `daily_risk_state.json` and restored on restart, so a
crash mid-bad-day can't silently un-trip an already-tripped breaker.

**Limits on this page (position cap, portfolio risk cap, daily loss
breaker) only work while `trading_bot.py` is actually running** — they
live in the bot's own loop, not on Alpaca's servers (Alpaca has no
account-level equivalent to set). The stop-loss/take-profit on each
individual position is different: that's a real order sitting on
Alpaca's side, so it stays protective even if the bot crashes.

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
biggest gainers/losers, keeps only ones that are also among the most
active by volume (a liquidity check against thin/manipulated stocks),
filters out anything under `SCANNER_MIN_PRICE` or already moved more
than `SCANNER_MAX_EXTENSION_PCT` today, excludes leveraged ETFs, and —
if `USE_NEWS_FILTER` is on — drops candidates with no recent news
behind them (a mover with zero news is more likely thin-volume noise
than a real catalyst; this is a presence/frequency check, **not** AI
sentiment scoring, so it stays fast and simple). It then trades the top
`SCANNER_WATCHLIST_SIZE` survivors.

```
SCANNER_REFRESH_HOURS=0.5
SCANNER_WATCHLIST_SIZE=12
SCANNER_CANDIDATE_POOL=100
SCANNER_MAX_EXTENSION_PCT=50
USE_NEWS_FILTER=true
NEWS_LOOKBACK_HOURS=24
MIN_NEWS_ITEMS=1
```

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
