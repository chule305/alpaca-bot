"""
Synthetic regression tests for auto_improve.py
--------------------------------------------------
No network access, no pytest dependency (matching test_strategy.py's own
convention) -- run directly with `py test_auto_improve.py`.

These specifically guard the mechanical enforcement, not just the happy
path: a forbidden constant that's missed here is a forbidden constant
that can ship for real. See auto_improve.py's module docstring for the
two-layer defense this is layer 2 of.
"""

import json
import os
import tempfile
import types
from datetime import datetime, timedelta, timezone

import auto_improve as ai

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def fake_module(**kwargs):
    return types.SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# Protected-file detection (Python-side mirror of the shell gate)
# ---------------------------------------------------------------------------

def test_diff_touches_protected_file_detects_exact_match():
    result = ai.diff_touches_protected_file(["strategy.py", "auto_improve.py"])
    check("exact protected filename is caught", result == "auto_improve.py", str(result))


def test_diff_touches_protected_file_detects_workflow_dir():
    result = ai.diff_touches_protected_file([".github/workflows/some_new_workflow.yml"])
    check("any file under .github/workflows/ is caught", result is not None)


def test_diff_touches_protected_file_detects_secret_like_name():
    result = ai.diff_touches_protected_file(["config/my_new_secrets_dump.py"])
    check("filename containing 'secret' is caught", result is not None)
    result2 = ai.diff_touches_protected_file(["CREDENTIALS_backup.txt"])
    check("filename containing 'credential' (any case) is caught", result2 is not None)


def test_diff_touches_protected_file_allows_normal_files():
    result = ai.diff_touches_protected_file(["strategy.py", "trading_bot.py", "test_strategy.py", "CLAUDE.md"])
    check("ordinary application files are not blocked", result is None, str(result))


def test_gate_yaml_lists_every_protected_file():
    """
    The REAL enforcement of PROTECTED_FILES lives in auto_improve.yml's
    shell gate, hardcoded there on purpose (see auto_improve.py's module
    docstring for why it can't read this Python list at runtime). This
    test is the only thing keeping the two lists honest over time -- it
    catches drift for a human reading test output, it is not itself a
    security control (by the time this test runs, the shell gate has
    already run first in the real workflow).
    """
    workflow_path = os.path.join(os.path.dirname(__file__), ".github", "workflows", "auto_improve.yml")
    with open(workflow_path, encoding="utf-8") as f:
        yaml_text = f.read()
    for entry in ai.PROTECTED_FILES:
        if entry.startswith(".github/workflows/"):
            continue  # covered by the wildcard `.github/workflows/*` case arm, not listed by name
        check(f"gate YAML mentions protected file '{entry}'", entry in yaml_text)


# ---------------------------------------------------------------------------
# Forbidden / immutable / bounded constants
# ---------------------------------------------------------------------------

def test_check_forbidden_and_immutable_flags_risk_based_sizing_true():
    mod = fake_module(USE_RISK_BASED_SIZING=True)
    violations = ai.check_forbidden_and_immutable(mod)
    check("USE_RISK_BASED_SIZING=True is flagged", len(violations) == 1, str(violations))


def test_check_forbidden_and_immutable_allows_false():
    mod = fake_module(USE_RISK_BASED_SIZING=False)
    violations = ai.check_forbidden_and_immutable(mod)
    check("USE_RISK_BASED_SIZING=False is not flagged", violations == [], str(violations))


def test_check_immutable_against_baseline_flags_trade_amount_change():
    mod = fake_module(TRADE_AMOUNT_USD=750.0)
    violations = ai.check_immutable_against_baseline(mod, {"TRADE_AMOUNT_USD": 500.0})
    check("changing TRADE_AMOUNT_USD is flagged", len(violations) == 1, str(violations))


def test_check_immutable_against_baseline_allows_unchanged():
    mod = fake_module(TRADE_AMOUNT_USD=500.0)
    violations = ai.check_immutable_against_baseline(mod, {"TRADE_AMOUNT_USD": 500.0})
    check("unchanged TRADE_AMOUNT_USD is not flagged", violations == [], str(violations))


def test_check_bounded_ranges_flags_out_of_range():
    mod = fake_module(STOP_LOSS_PCT=50.0, TAKE_PROFIT_PCT=10.0, MAX_PORTFOLIO_HEAT_USD=450.0,
                       MAX_CONCURRENT_POSITIONS=18, MAX_POSITION_PCT_OF_EQUITY=5.0)
    violations = ai.check_bounded_ranges(mod)
    check("out-of-range STOP_LOSS_PCT is flagged, nothing else", violations == ["STOP_LOSS_PCT=50.0 is outside its allowed range [2.0, 8.0]"], str(violations))


def test_check_bounded_ranges_allows_in_range():
    mod = fake_module(STOP_LOSS_PCT=5.0, TAKE_PROFIT_PCT=10.0, MAX_PORTFOLIO_HEAT_USD=450.0,
                       MAX_CONCURRENT_POSITIONS=18, MAX_POSITION_PCT_OF_EQUITY=5.0)
    violations = ai.check_bounded_ranges(mod)
    check("in-range values are not flagged", violations == [], str(violations))


def test_check_bounded_ranges_boundary_values_are_allowed():
    mod = fake_module(STOP_LOSS_PCT=2.0, TAKE_PROFIT_PCT=20.0, MAX_PORTFOLIO_HEAT_USD=100.0,
                       MAX_CONCURRENT_POSITIONS=25, MAX_POSITION_PCT_OF_EQUITY=1.0)
    violations = ai.check_bounded_ranges(mod)
    check("range boundaries themselves are inclusive/allowed", violations == [], str(violations))


def test_check_bounded_ranges_allows_live_conviction_heat_default():
    # $2,000 is the real 2026-08-26 default (raised from $450 alongside
    # the $25k/day conviction pool -- see trading_bot.py's
    # MAX_PORTFOLIO_HEAT_USD comment). Confirms the widened [100, 3000]
    # range this same change made actually admits it.
    mod = fake_module(STOP_LOSS_PCT=5.0, TAKE_PROFIT_PCT=10.0, MAX_PORTFOLIO_HEAT_USD=2000.0,
                       MAX_CONCURRENT_POSITIONS=18, MAX_POSITION_PCT_OF_EQUITY=5.0)
    violations = ai.check_bounded_ranges(mod)
    check("the live $2,000 MAX_PORTFOLIO_HEAT_USD default is within its own range", violations == [], str(violations))


def test_check_immutable_against_baseline_flags_daily_pool_change():
    mod = fake_module(MAX_DAILY_DEPLOYED_CAPITAL_USD=50000.0)
    violations = ai.check_immutable_against_baseline(mod, {"MAX_DAILY_DEPLOYED_CAPITAL_USD": 25000.0})
    check("raising MAX_DAILY_DEPLOYED_CAPITAL_USD is flagged -- only a human moves the daily ceiling",
          len(violations) == 1, str(violations))


UNGUARDED_SOURCE = """
TRADE_AMOUNT_USD = float(os.getenv("TRADE_AMOUNT_USD", 500))
NEW_BOOST_USD = float(os.getenv("NEW_BOOST_USD", 100))
"""

GUARDED_SOURCE = """
TRADE_AMOUNT_USD = float(os.getenv("TRADE_AMOUNT_USD", 500))
NEW_BOOST_USD = float(os.getenv("NEW_BOOST_USD", 100))
if NEW_BOOST_USD > TRADE_AMOUNT_USD * 2:
    raise ValueError("NEW_BOOST_USD too large")
"""


GUARDED_SOURCE_SIZE_UP = """
TRADE_AMOUNT_USD = float(os.getenv("TRADE_AMOUNT_USD", 500))
MAX_DAILY_DEPLOYED_CAPITAL_USD = float(os.getenv("MAX_DAILY_DEPLOYED_CAPITAL_USD", 25000))
NEW_TIER_USD = float(os.getenv("NEW_TIER_USD", 15000))
if NEW_TIER_USD > MAX_DAILY_DEPLOYED_CAPITAL_USD:
    raise ValueError("NEW_TIER_USD too large")
"""


def test_check_new_sizing_constants_have_guards_flags_unguarded():
    violations = ai.check_new_sizing_constants_have_guards(UNGUARDED_SOURCE, 500.0)
    check("new _USD constant with no ValueError guard is flagged", len(violations) == 1, str(violations))


def test_check_new_sizing_constants_have_guards_allows_size_down_guard_shape():
    violations = ai.check_new_sizing_constants_have_guards(GUARDED_SOURCE, 500.0)
    check("a size-down lever guarded against TRADE_AMOUNT_USD is not flagged", violations == [], str(violations))


def test_check_new_sizing_constants_have_guards_allows_size_up_guard_shape():
    violations = ai.check_new_sizing_constants_have_guards(GUARDED_SOURCE_SIZE_UP, 25000.0)
    check("a size-up lever guarded against MAX_DAILY_DEPLOYED_CAPITAL_USD is not flagged",
          violations == [], str(violations))


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_rate_limit_allows_empty_state():
    reason = ai.rate_limit_blocks_new_change({"changes": []})
    check("no prior changes -> not blocked", reason is None, str(reason))


def test_rate_limit_blocks_when_too_recent():
    now = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
    state = {"changes": [{"date": (now - timedelta(hours=1)).isoformat(), "commit_sha": "a", "reverted": False}]}
    reason = ai.rate_limit_blocks_new_change(state, now=now)
    check("a change 1h ago blocks a new one (min 20h)", reason is not None, str(reason))


def test_rate_limit_allows_after_cooldown():
    now = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
    state = {"changes": [{"date": (now - timedelta(hours=25)).isoformat(), "commit_sha": "a", "reverted": False}]}
    reason = ai.rate_limit_blocks_new_change(state, now=now)
    check("a single change 25h ago does not block", reason is None, str(reason))


def test_rate_limit_blocks_at_weekly_cap():
    now = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
    state = {"changes": [
        {"date": (now - timedelta(days=1, hours=1)).isoformat(), "commit_sha": "a", "reverted": False},
        {"date": (now - timedelta(days=3)).isoformat(), "commit_sha": "b", "reverted": False},
        {"date": (now - timedelta(days=5)).isoformat(), "commit_sha": "c", "reverted": False},
    ]}
    reason = ai.rate_limit_blocks_new_change(state, now=now)
    check("3 changes already in the last 7 days blocks a 4th even past cooldown", reason is not None, str(reason))


def test_rate_limit_ignores_changes_older_than_a_week_for_the_cap():
    now = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
    state = {"changes": [
        {"date": (now - timedelta(days=10)).isoformat(), "commit_sha": "a", "reverted": False},
        {"date": (now - timedelta(days=9)).isoformat(), "commit_sha": "b", "reverted": False},
        {"date": (now - timedelta(days=8)).isoformat(), "commit_sha": "c", "reverted": False},
        {"date": (now - timedelta(days=1, hours=1)).isoformat(), "commit_sha": "d", "reverted": False},
    ]}
    reason = ai.rate_limit_blocks_new_change(state, now=now)
    check("changes older than 7 days don't count toward the weekly cap", reason is None, str(reason))


# ---------------------------------------------------------------------------
# State file persistence -- uses a temp file, never touches the real
# auto_improve_state.json this repo ships with.
# ---------------------------------------------------------------------------

def _with_temp_state_file(fn):
    original = ai.STATE_FILE
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)  # load_state should handle "file does not exist yet"
    ai.STATE_FILE = path
    try:
        fn(path)
    finally:
        ai.STATE_FILE = original
        if os.path.exists(path):
            os.remove(path)


def test_load_state_missing_file_returns_empty():
    def run(path):
        state = ai.load_state()
        check("missing state file loads as empty changes list", state == {"changes": []}, str(state))
    _with_temp_state_file(run)


def test_save_and_load_state_round_trip():
    def run(path):
        original = {"changes": [{"date": "2026-08-25T00:00:00+00:00", "commit_sha": "abc123",
                                  "summary": "test change", "reverted": False}]}
        ai.save_state(original)
        loaded = ai.load_state()
        check("state round-trips through save/load unchanged", loaded == original, str(loaded))
    _with_temp_state_file(run)


def test_load_state_corrupt_file_fails_closed():
    def run(path):
        with open(path, "w") as f:
            f.write("{not valid json,,,")
        state = ai.load_state()
        reason = ai.rate_limit_blocks_new_change(state)
        check("a corrupt state file fails CLOSED (blocks new changes)", reason is not None, str(reason))
    _with_temp_state_file(run)


# ---------------------------------------------------------------------------
# Rollback check -- without real Alpaca credentials in this test's
# environment (deliberately not provided when the test suite runs -- see
# auto_improve.yml, no step that runs the test suite has ALPACA_API_KEY
# in scope), real_pnl_pct_since must fail safe (None), never raise and
# never fabricate a number.
# ---------------------------------------------------------------------------

def test_real_pnl_pct_since_returns_none_without_credentials():
    saved = {k: os.environ.pop(k, None) for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")}
    try:
        result = ai.real_pnl_pct_since(datetime.now(timezone.utc).isoformat())
        check("no credentials -> real_pnl_pct_since returns None, doesn't raise", result is None, str(result))
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_find_change_to_revert_returns_none_when_data_unavailable():
    saved = {k: os.environ.pop(k, None) for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")}
    try:
        state = {"changes": [{"date": datetime.now(timezone.utc).isoformat(), "commit_sha": "abc",
                               "summary": "x", "reverted": False}]}
        result = ai.find_change_to_revert(state)
        check("no credentials -> find_change_to_revert returns None, doesn't crash the whole check", result is None, str(result))
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_find_change_to_revert_skips_already_reverted():
    state = {"changes": [{"date": "2020-01-01T00:00:00+00:00", "commit_sha": "abc",
                           "summary": "x", "reverted": True}]}
    result = ai.find_change_to_revert(state)
    check("an already-reverted change is never proposed for revert again", result is None, str(result))


# ---------------------------------------------------------------------------
# run_all_guardrails -- the actual entry point the CLI calls
# ---------------------------------------------------------------------------

def test_run_all_guardrails_clean_module_has_no_violations():
    strategy_mod = fake_module(USE_RISK_BASED_SIZING=False, TRADE_AMOUNT_USD=500.0,
                                STOP_LOSS_PCT=5.0, TAKE_PROFIT_PCT=10.0, MAX_POSITION_PCT_OF_EQUITY=5.0)
    trading_bot_mod = fake_module(MAX_CONCURRENT_POSITIONS=18, MAX_PORTFOLIO_HEAT_USD=450.0)
    violations = ai.run_all_guardrails(strategy_mod, trading_bot_mod, {"TRADE_AMOUNT_USD": 500.0}, GUARDED_SOURCE)
    check("an unmodified, in-bounds config produces zero violations", violations == [], str(violations))


def test_run_all_guardrails_catches_multiple_independent_problems():
    strategy_mod = fake_module(USE_RISK_BASED_SIZING=True, TRADE_AMOUNT_USD=999.0,
                                STOP_LOSS_PCT=5.0, TAKE_PROFIT_PCT=10.0, MAX_POSITION_PCT_OF_EQUITY=5.0)
    trading_bot_mod = fake_module(MAX_CONCURRENT_POSITIONS=18, MAX_PORTFOLIO_HEAT_USD=450.0)
    violations = ai.run_all_guardrails(strategy_mod, trading_bot_mod, {"TRADE_AMOUNT_USD": 500.0}, UNGUARDED_SOURCE)
    check("independent violations (forbidden flag + immutable const + unguarded sizing) all surface together",
          len(violations) == 3, str(violations))


if __name__ == "__main__":
    tests = [
        test_diff_touches_protected_file_detects_exact_match,
        test_diff_touches_protected_file_detects_workflow_dir,
        test_diff_touches_protected_file_detects_secret_like_name,
        test_diff_touches_protected_file_allows_normal_files,
        test_gate_yaml_lists_every_protected_file,
        test_check_forbidden_and_immutable_flags_risk_based_sizing_true,
        test_check_forbidden_and_immutable_allows_false,
        test_check_immutable_against_baseline_flags_trade_amount_change,
        test_check_immutable_against_baseline_allows_unchanged,
        test_check_bounded_ranges_flags_out_of_range,
        test_check_bounded_ranges_allows_in_range,
        test_check_bounded_ranges_boundary_values_are_allowed,
        test_check_bounded_ranges_allows_live_conviction_heat_default,
        test_check_immutable_against_baseline_flags_daily_pool_change,
        test_check_new_sizing_constants_have_guards_flags_unguarded,
        test_check_new_sizing_constants_have_guards_allows_size_down_guard_shape,
        test_check_new_sizing_constants_have_guards_allows_size_up_guard_shape,
        test_rate_limit_allows_empty_state,
        test_rate_limit_blocks_when_too_recent,
        test_rate_limit_allows_after_cooldown,
        test_rate_limit_blocks_at_weekly_cap,
        test_rate_limit_ignores_changes_older_than_a_week_for_the_cap,
        test_load_state_missing_file_returns_empty,
        test_save_and_load_state_round_trip,
        test_load_state_corrupt_file_fails_closed,
        test_real_pnl_pct_since_returns_none_without_credentials,
        test_find_change_to_revert_returns_none_when_data_unavailable,
        test_find_change_to_revert_skips_already_reverted,
        test_run_all_guardrails_clean_module_has_no_violations,
        test_run_all_guardrails_catches_multiple_independent_problems,
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
