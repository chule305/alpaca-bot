"""
Autonomous Self-Improvement Guardrails
-----------------------------------------
ACTIVE as of 2026-08-26 -- see .github/workflows/auto_improve.yml's
`schedule:` trigger (weekday evenings, after market close). Built and
left deliberately inert on 2026-08-25 while this enforcement layer was
reviewed, then turned on the next day at the user's explicit request.

WHAT THIS IS: the enforcement layer for a daily, LLM-driven pipeline
(see auto_improve_prompt.md) that researches the bot's real performance
and proposes/tests/ships small code or parameter changes with NO human
approval per change. Built this way on 2026-08-25 at the user's explicit
request, after a long session that repeatedly found even carefully,
individually-tested changes can combine into a real regression that only
a full-system, human-legible review caught (see CLAUDE.md's 2026-08-23
entries) -- so "autonomous" here means zero DAILY human input required,
not zero rules. The rules below are enforced in CODE, not just asked of
the LLM in a prompt: a prompt is advisory, this file is not.

THE CORE PRINCIPLE THIS FILE EXISTS TO ENFORCE: the LLM pipeline can
propose almost anything (that's deliberate -- see the design conversation
this shipped from, "loosen the research, keep the validation rigorous").
What it can NEVER do is ship something that violates a hard rule below,
no matter how good its own reasoning sounds. These checks don't trust the
LLM's judgment about whether ITS OWN change is safe -- that's exactly the
kind of self-assessment that produces the failure modes this project has
already hit for real (the 2026-08-04/05 sizing incident happened despite
a backtest that looked completely convincing).

TWO-LAYER DEFENSE, in this order (see .github/workflows/auto_improve.yml):
  1. A PURE-SHELL check in the workflow YAML itself, run BEFORE any
     Python from this checkout is trusted, that rejects the run outright
     if the diff touches a protected file (see PROTECTED_FILES below).
     This has to live in the YAML, not here -- if the LLM edited THIS
     file as part of its change, running the tampered copy to check
     itself would be checking nothing. The shell gate can't be tampered
     with the same way because it never executes repo-provided code to
     make its decision, only `git diff --name-only` + a hardcoded list.
  2. Everything else in this file -- sizing caps, the risk-based-sizing
     lock, rate limiting, the rollback check -- runs AFTER the shell gate
     has already confirmed this file itself wasn't touched, so by the
     time Python executes, these checks can be trusted to be the real,
     unmodified rules.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LAYER 1 (mirrored here for documentation/testing -- the REAL enforcement
# of this list lives in .github/workflows/auto_improve.yml's shell step,
# which runs before this file is ever imported from a possibly-modified
# checkout; see this module's docstring for why it can't live here alone).
# Never let an autonomous change touch: its own leash, the CI definition
# that runs it, secrets/credentials, or dependency pinning.
# ---------------------------------------------------------------------------
PROTECTED_FILES = [
    "auto_improve.py",
    "test_auto_improve.py",
    "auto_improve_prompt.md",
    ".github/workflows/auto_improve.yml",
    ".github/workflows/trade.yml",
    ".github/workflows/daily_summary.yml",
    "requirements.txt",
    ".env",
    ".env.example",
    ".gitignore",
]


def diff_touches_protected_file(changed_files: list[str]) -> str | None:
    """
    Returns the first protected path that was touched, or None if the
    diff is clean. `changed_files` should come from `git diff --name-only`
    against the pre-change commit. Also blocks anything under
    .github/workflows/ wholesale (not just the three named above) and
    anything whose path contains "secret" or "credential" case-
    insensitively, so a NEW workflow file or a cleverly-named credentials
    dump can't slip through by not matching the exact list.
    """
    for f in changed_files:
        f_norm = f.replace("\\", "/")
        if f_norm in PROTECTED_FILES:
            return f_norm
        if f_norm.startswith(".github/workflows/"):
            return f_norm
        lowered = f_norm.lower()
        if "secret" in lowered or "credential" in lowered:
            return f_norm
    return None


# ---------------------------------------------------------------------------
# LAYER 2: substantive rules on WHAT a change may contain, independent of
# which files it touches. These run against strategy.py/trading_bot.py/
# backtest.py's actual post-change contents (imported fresh, so this reads
# real values, not regexes trying to parse Python).
# ---------------------------------------------------------------------------

# The single most important rule in this file. USE_RISK_BASED_SIZING=true
# is the exact, specific thing that turned a dormant 25%-of-equity cap
# into real $15,700-$19,900 positions on 2026-08-04 -- see CLAUDE.md. An
# autonomous pipeline must NEVER be able to flip this on, no matter what
# backtest evidence it thinks it has, because the failure mode isn't
# visible in backtest metrics (win rate/PF/drawdown-as-percentage) at
# all -- it only shows up as a real dollar position size, which is
# exactly the kind of thing an LLM optimizing for backtest metrics has no
# natural reason to check.
FORBIDDEN_TRUE_CONSTANTS = ["USE_RISK_BASED_SIZING"]

# TRADE_AMOUNT_USD is the base per-trade dollar amount (the FLOOR).
# MAX_DAILY_DEPLOYED_CAPITAL_USD is the total $/day conviction sizing may
# ever deploy across ALL positions combined (the CEILING) -- raised from
# an earlier, much smaller 2x-TRADE_AMOUNT_USD cap to $25,000 on
# 2026-08-26 at the user's direct, explicit request (paper account, own
# research, see CLAUDE.md). Both are IMMUTABLE for the exact same reason:
# they're the two human-set numbers everything else in the sizing system
# (CONVICTION_TIER1_USD, and any future tier the pipeline proposes) is
# bounded between -- see
# check_new_sizing_constants_have_guards below. Only a human moves either
# one; the autonomous pipeline can tune how conviction is SCORED (which
# ADX/volume thresholds count, which strategies qualify) but never how
# far the resulting size can reach.
IMMUTABLE_CONSTANTS = ["TRADE_AMOUNT_USD", "MAX_DAILY_DEPLOYED_CAPITAL_USD"]

# Bounded ranges: autonomous changes MAY adjust these, but not past a
# human-set fence. Chosen to keep risk management structurally intact
# (a stop that can never trigger is not a stop) while still leaving real
# room for the pipeline to find a better balance.
BOUNDED_RANGES = {
    "STOP_LOSS_PCT": (2.0, 8.0),
    "TAKE_PROFIT_PCT": (5.0, 20.0),
    # Raised 1000 -> 3000 on 2026-08-26 alongside MAX_DAILY_DEPLOYED_
    # CAPITAL_USD -- a single top-tier conviction trade ($25,000 at a 5%
    # stop) carries $1,250 of heat on its own, which the old 1000 ceiling
    # would have blocked structurally regardless of the trade's merits.
    # See trading_bot.py's MAX_PORTFOLIO_HEAT_USD comment for the exact
    # math behind the new default (2000) this range now has headroom
    # around, in both directions.
    "MAX_PORTFOLIO_HEAT_USD": (100.0, 3000.0),
    "MAX_CONCURRENT_POSITIONS": (3, 25),
    "MAX_POSITION_PCT_OF_EQUITY": (1.0, 10.0),  # only matters if risk-based sizing is ever on; fenced anyway, belt and suspenders
}

STATE_FILE = "auto_improve_state.json"
MAX_CHANGES_PER_ROLLING_7_DAYS = 3
MIN_HOURS_BETWEEN_CHANGES = 20  # effectively "at most one per day"


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {"changes": []}
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Fails CLOSED on a corrupt state file -- treat as "rate limit
        # already hit" (empty history would UNDER-count changes and let
        # more through; a full/unknown history is the safe misread).
        return {"changes": [{"date": datetime.now(timezone.utc).isoformat(),
                              "commit_sha": "unknown", "summary": "STATE FILE UNREADABLE",
                              "reverted": False}] * MAX_CHANGES_PER_ROLLING_7_DAYS}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
        f.write("\n")


def rate_limit_blocks_new_change(state: dict, now: datetime | None = None) -> str | None:
    """Returns a reason string if a new change should be blocked, else None."""
    now = now or datetime.now(timezone.utc)
    changes = state.get("changes", [])
    if changes:
        last = changes[-1]
        last_dt = datetime.fromisoformat(last["date"])
        hours_since = (now - last_dt).total_seconds() / 3600
        if hours_since < MIN_HOURS_BETWEEN_CHANGES:
            return f"last change was only {hours_since:.1f}h ago (minimum {MIN_HOURS_BETWEEN_CHANGES}h between changes)"
    week_ago = now - timedelta(days=7)
    recent = [c for c in changes if datetime.fromisoformat(c["date"]) >= week_ago]
    if len(recent) >= MAX_CHANGES_PER_ROLLING_7_DAYS:
        return f"{len(recent)} changes already shipped in the last 7 days (max {MAX_CHANGES_PER_ROLLING_7_DAYS})"
    return None


def check_forbidden_and_immutable(module) -> list[str]:
    """
    module: the freshly-imported strategy (or trading_bot) module AFTER
    the proposed change, so this reads real post-change values.
    Returns a list of violation strings (empty = clean).
    """
    violations = []
    for name in FORBIDDEN_TRUE_CONSTANTS:
        val = getattr(module, name, None)
        if val is True:
            violations.append(f"{name} is True -- this constant may NEVER be enabled by an autonomous change")
    return violations


def check_immutable_against_baseline(module, baseline_values: dict) -> list[str]:
    violations = []
    for name in IMMUTABLE_CONSTANTS:
        if name not in baseline_values:
            continue
        current = getattr(module, name, None)
        if current != baseline_values[name]:
            violations.append(f"{name} changed from {baseline_values[name]} to {current} -- this constant is immutable, only a human may change it")
    return violations


def check_bounded_ranges(module) -> list[str]:
    violations = []
    for name, (lo, hi) in BOUNDED_RANGES.items():
        val = getattr(module, name, None)
        if val is None:
            continue
        if not (lo <= val <= hi):
            violations.append(f"{name}={val} is outside its allowed range [{lo}, {hi}]")
    return violations


def check_new_sizing_constants_have_guards(strategy_source: str, ceiling_usd: float) -> list[str]:
    """
    Best-effort static check: any NEW module-level constant whose name
    ends in _USD and looks like a flat-dollar sizing lever should have a
    corresponding `raise ValueError` guard nearby bounding it against ONE
    of the two human-set ceilings already established in this file --
    either `> MAX_DAILY_DEPLOYED_CAPITAL_USD` (the shape CONVICTION_TIER1_USD
    uses, for a lever that sizes UP) or
    `> TRADE_AMOUNT_USD` (the shape VOLATILITY_SCALED_REDUCED_USD uses,
    for a lever that only ever sizes DOWN and so only needs to stay under
    the normal flat baseline, not the much larger daily ceiling). Either
    is acceptable; what's not acceptable is neither. This is intentionally
    coarse (string scanning, not AST analysis) -- false positives just
    mean a human needs to look, which is the safe failure direction.
    """
    import re
    violations = []
    usd_constants = re.findall(r'^([A-Z][A-Z0-9_]*_USD)\s*=\s*float\(os\.getenv', strategy_source, re.MULTILINE)
    for name in usd_constants:
        if name in ("TRADE_AMOUNT_USD", "DEFAULT_BACKTEST_EQUITY", "MAX_DAILY_DEPLOYED_CAPITAL_USD"):
            continue
        size_up_guard = rf'if\s+{name}\s*>\s*MAX_DAILY_DEPLOYED_CAPITAL_USD'
        size_down_guard = rf'if\s+{name}\s*>\s*TRADE_AMOUNT_USD'
        if not re.search(size_up_guard, strategy_source) and not re.search(size_down_guard, strategy_source):
            violations.append(
                f"{name} is a new flat-dollar sizing constant with no `if {name} > "
                f"MAX_DAILY_DEPLOYED_CAPITAL_USD: raise ValueError` (for a size-up lever) or "
                f"`if {name} > TRADE_AMOUNT_USD: raise ValueError` (for a size-down lever) guard -- "
                f"every sizing lever must be structurally bounded by one of this bot's two human-set "
                f"ceilings (TRADE_AMOUNT_USD as the floor, ${ceiling_usd:,.0f} MAX_DAILY_DEPLOYED_"
                f"CAPITAL_USD as the ceiling), the same way every existing sizing constant already is."
            )
    return violations


def run_all_guardrails(strategy_module, trading_bot_module, baseline_values: dict,
                        strategy_source: str) -> list[str]:
    """The single entry point the workflow calls. Returns all violations found (empty = safe to ship)."""
    violations = []
    violations += check_forbidden_and_immutable(strategy_module)
    violations += check_forbidden_and_immutable(trading_bot_module)
    violations += check_immutable_against_baseline(strategy_module, baseline_values)
    violations += check_bounded_ranges(strategy_module)
    violations += check_bounded_ranges(trading_bot_module)
    violations += check_new_sizing_constants_have_guards(
        strategy_source, getattr(strategy_module, "MAX_DAILY_DEPLOYED_CAPITAL_USD", 25000.0))
    # trading_bot.py imports several names directly from strategy.py, so
    # the same underlying constant can trip the same check under both
    # modules -- dedupe (order-preserving) rather than show a doubled
    # violation in the blocked-change email.
    seen = set()
    deduped = []
    for v in violations:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


# ---------------------------------------------------------------------------
# Rollback: real performance since each still-active autonomous change is
# checked BEFORE proposing a new one (see auto_improve_prompt.md's step
# order), so the system never stacks a new change on top of one it's
# simultaneously trying to undo.
# ---------------------------------------------------------------------------

MIN_TRADES_BEFORE_ROLLBACK_JUDGEMENT = 10
ROLLBACK_LOSS_THRESHOLD_PCT = -5.0  # of the capital allocated to trades since the change


def real_pnl_pct_since(commit_date_iso: str) -> tuple[float, int] | None:
    """
    Real $ P&L (as a % of $ deployed) and trade count since a given
    commit's date, reconstructed from Alpaca's real order history --
    same FIFO round-trip methodology used throughout this project's own
    investigations (see CLAUDE.md). Returns None if Alpaca can't be
    reached (fails safe: no rollback decision gets made on missing data).
    """
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
    except ImportError:
        return None

    api_key = (os.getenv("ALPACA_API_KEY") or "").strip()
    secret_key = (os.getenv("ALPACA_SECRET_KEY") or "").strip()
    if not api_key or not secret_key:
        return None

    try:
        client = TradingClient(api_key, secret_key, paper=True)
        orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500))
    except Exception:
        return None

    filled = [o for o in orders if o.status.value == "filled" and o.filled_avg_price is not None]
    filled.sort(key=lambda o: o.filled_at)
    commit_dt = datetime.fromisoformat(commit_date_iso)

    open_lots: dict[str, list] = {}
    trades = []
    for o in filled:
        qty = float(o.filled_qty)
        price = float(o.filled_avg_price)
        if o.side.value == "buy":
            open_lots.setdefault(o.symbol, []).append({"qty": qty, "price": price, "time": o.filled_at})
        elif o.side.value == "sell":
            remaining = qty
            lots = open_lots.get(o.symbol, [])
            while remaining > 1e-6 and lots:
                lot = lots[0]
                matched = min(remaining, lot["qty"])
                if lot["time"] >= commit_dt:
                    trades.append({
                        "pnl": (price - lot["price"]) * matched,
                        "deployed": lot["price"] * matched,
                    })
                lot["qty"] -= matched
                remaining -= matched
                if lot["qty"] <= 1e-6:
                    lots.pop(0)

    if len(trades) < MIN_TRADES_BEFORE_ROLLBACK_JUDGEMENT:
        return (0.0, len(trades))  # not enough data yet -- caller treats this as "don't revert"

    total_pnl = sum(t["pnl"] for t in trades)
    total_deployed = sum(t["deployed"] for t in trades)
    pnl_pct = (total_pnl / total_deployed * 100) if total_deployed > 0 else 0.0
    return (pnl_pct, len(trades))


def find_change_to_revert(state: dict) -> dict | None:
    """
    Checks every active (not already reverted) autonomous change, oldest
    first, and returns the first one whose real performance since it
    shipped crosses ROLLBACK_LOSS_THRESHOLD_PCT with enough real trades
    to trust the read. None if nothing needs reverting (including the
    common case: not enough real trades have accumulated yet to judge).
    """
    for change in state.get("changes", []):
        if change.get("reverted"):
            continue
        result = real_pnl_pct_since(change["date"])
        if result is None:
            continue
        pnl_pct, n_trades = result
        if n_trades < MIN_TRADES_BEFORE_ROLLBACK_JUDGEMENT:
            continue
        if pnl_pct <= ROLLBACK_LOSS_THRESHOLD_PCT:
            change["_measured_pnl_pct"] = pnl_pct
            change["_measured_n_trades"] = n_trades
            return change
    return None


def perform_revert(commit_sha: str) -> bool:
    """git revert --no-edit <sha>. Returns True on success."""
    result = subprocess.run(["git", "revert", "--no-edit", commit_sha], capture_output=True, text=True)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Notification -- reuses daily_summary.py's already-working, already-
# configured send_email exactly (same SMTP secrets already in GitHub
# Actions for the daily summary email), rather than standing up a second
# notification channel with its own credentials.
# ---------------------------------------------------------------------------

def notify(subject: str, body: str) -> None:
    from daily_summary import send_email
    send_email(f"[auto-improve] {subject}", body)


# ---------------------------------------------------------------------------
# CLI -- this is what .github/workflows/auto_improve.yml actually calls.
# Every subcommand prints EXACTLY ONE line of JSON to stdout (diagnostics
# go to stderr) and exits 0 unless something genuinely broke (an
# exception), so the workflow branches on the JSON content -- via
# `echo "x=$(python auto_improve.py foo)" >> "$GITHUB_OUTPUT"` -- rather
# than juggling exit codes. Keeping decision LOGIC here (tested by
# test_auto_improve.py) and decision WIRING in the YAML (auditable in a
# code review, see auto_improve.yml's own comments) is deliberate: the
# YAML stays short enough that a human can read the whole control flow in
# one sitting, which matters most for something that ships code without
# asking anyone first.
# ---------------------------------------------------------------------------

def _print_json(obj: dict) -> None:
    print(json.dumps(obj))


def cmd_snapshot_baseline() -> dict:
    """Run BEFORE the LLM step touches anything, on the clean checkout."""
    import strategy
    return {name: getattr(strategy, name, None) for name in IMMUTABLE_CONSTANTS}


def cmd_rollback_check() -> dict:
    state = load_state()
    to_revert = find_change_to_revert(state)
    if to_revert:
        return {
            "action": "revert",
            "commit_sha": to_revert["commit_sha"],
            "summary": to_revert.get("summary", ""),
            "pnl_pct": round(to_revert["_measured_pnl_pct"], 2),
            "n_trades": to_revert["_measured_n_trades"],
        }
    return {"action": "none"}


def cmd_rate_limit_check() -> dict:
    state = load_state()
    reason = rate_limit_blocks_new_change(state)
    return {"blocked": reason is not None, "reason": reason}


def cmd_verify_guardrails(baseline_path: str) -> dict:
    with open(baseline_path) as f:
        baseline_values = json.load(f)
    import strategy
    import trading_bot
    with open("strategy.py", encoding="utf-8") as f:
        strategy_source = f.read()
    violations = run_all_guardrails(strategy, trading_bot, baseline_values, strategy_source)
    return {"violations": violations}


def cmd_record_change(commit_sha: str, summary: str) -> dict:
    state = load_state()
    state.setdefault("changes", []).append({
        "date": datetime.now(timezone.utc).isoformat(),
        "commit_sha": commit_sha,
        "summary": summary,
        "reverted": False,
    })
    save_state(state)
    return {"ok": True}


def cmd_record_revert(commit_sha: str) -> dict:
    state = load_state()
    found = False
    for change in state.get("changes", []):
        if change["commit_sha"] == commit_sha:
            change["reverted"] = True
            found = True
    save_state(state)
    return {"ok": found}


def cmd_notify(subject: str, body: str) -> dict:
    notify(subject, body)
    return {"sent": True}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Guardrails for the autonomous self-improvement pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("snapshot-baseline")
    sub.add_parser("rollback-check")
    sub.add_parser("rate-limit-check")

    p = sub.add_parser("verify-guardrails")
    p.add_argument("--baseline", required=True)

    p = sub.add_parser("record-change")
    p.add_argument("--sha", required=True)
    p.add_argument("--summary", required=True)

    p = sub.add_parser("record-revert")
    p.add_argument("--sha", required=True)

    p = sub.add_parser("notify")
    p.add_argument("--subject", required=True)

    args = parser.parse_args()

    if args.command == "snapshot-baseline":
        _print_json(cmd_snapshot_baseline())
    elif args.command == "rollback-check":
        _print_json(cmd_rollback_check())
    elif args.command == "rate-limit-check":
        _print_json(cmd_rate_limit_check())
    elif args.command == "verify-guardrails":
        _print_json(cmd_verify_guardrails(args.baseline))
    elif args.command == "record-change":
        _print_json(cmd_record_change(args.sha, args.summary))
    elif args.command == "record-revert":
        _print_json(cmd_record_revert(args.sha))
    elif args.command == "notify":
        body = sys.stdin.read()
        _print_json(cmd_notify(args.subject, body))


if __name__ == "__main__":
    main()
