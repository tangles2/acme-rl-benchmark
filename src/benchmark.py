"""
Benchmark harness for the Acme finance operations workflow.

Implements the full scoring contract from the assignment:
  - case status and resolution match expected output
  - integer cent amounts match
  - required evidence IDs are cited
  - cited IDs were actually observed in prior tool results
  - forbidden evidence IDs are not cited
  - no unsafe mutations (resolve without invoice+payment evidence)
  - tool efficiency (broad scan penalty)

Run via run_benchmark.py, not directly.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents import BaselineAgent, PolicyAgent
    from src.environment import MockEnvironment

FIXTURES = Path(__file__).parent.parent / "fixtures" / "rl-finetuning"

BROAD_SCAN_PENALTY = 0.05  # points deducted per broad scan


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tasks() -> list[dict]:
    tasks = []
    with open(FIXTURES / "tasks.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def load_expected() -> dict[str, dict]:
    with open(FIXTURES / "expected_outputs.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def score_task(
    task: dict,
    env: "MockEnvironment",
    expected: dict,
) -> dict[str, object]:
    """
    Score one completed task run against its expected output.

    Returns a dict with one boolean per criterion and a float 'total' in [0,1].
    A task is a 'strict pass' only when every criterion is True.
    """
    task_id    = task["task_id"]
    exp        = expected[task_id]
    case_id    = exp["case_id"]
    case_state = env.cases.get(case_id, {})
    final_ans  = env.get_final_answer() or {"answer": "", "evidence_ids": []}

    observed_ids      = env.observed_ids
    cited_ids         = set(final_ans.get("evidence_ids", []))
    required_ids      = set(exp.get("required_evidence_ids", []))
    forbidden_ids     = set(exp.get("forbidden_evidence_ids", []))

    # ---- individual criteria ----------------------------------------

    status_ok = case_state.get("status") == exp["expected_status"]
    resolution_ok = case_state.get("resolution") == exp["expected_resolution"]

    # Amount accuracy: check against final_answer text and case exception.
    amounts_ok = _check_amounts(case_state, final_ans, exp, trace=env.trace)

    # Required evidence cited in final_answer.
    evidence_ok = required_ids.issubset(cited_ids)

    # All cited evidence was returned by a prior tool call.
    observed_evidence_ok = cited_ids.issubset(observed_ids)

    # No forbidden IDs cited.
    forbidden_ok = len(cited_ids & forbidden_ids) == 0

    # No unsafe mutation flag was set during the run.
    unsafe_mutation_ok = not env.unsafe_mutation

    # Tool efficiency: penalise broad scans and redundant identical narrow calls.
    redundant = getattr(env, "redundant_call_count", 0)
    efficiency = max(0.0, 1.0 - (env.broad_scan_count + redundant) * BROAD_SCAN_PENALTY)
    efficiency_ok = env.broad_scan_count == 0 and redundant == 0

    # ---- aggregate ---------------------------------------------------

    criteria = {
        "status":            status_ok,
        "resolution":        resolution_ok,
        "amounts":           amounts_ok,
        "evidence":          evidence_ok,
        "observed_evidence": observed_evidence_ok,
        "forbidden_evidence": forbidden_ok,
        "unsafe_mutation":   unsafe_mutation_ok,
        "tool_efficiency":   efficiency_ok,
    }
    strict_pass = all(criteria.values())
    # Weighted total: binary criteria average + efficiency score averaged in.
    binary_score = sum(1 for v in criteria.values() if v) / len(criteria)
    # Replace efficiency criterion with the continuous score in the total.
    total = (binary_score * len(criteria) - (1 if efficiency_ok else 0) + efficiency) / len(criteria)

    return {
        "task_id":           task_id,
        "strict_pass":       strict_pass,
        "criteria":          criteria,
        "total":             round(total, 3),
        "broad_scans":       env.broad_scan_count,
        "redundant_calls":   getattr(env, "redundant_call_count", 0),
        "tool_calls":        len(env.trace),
        "cited_ids":         sorted(cited_ids),
        "observed_ids":      sorted(observed_ids),
    }


def _extract_from_trace(trace: list, tool_name: str) -> list:
    """Return the result list from the most recent call to tool_name, or []."""
    for step in reversed(trace):
        if step["tool"] == tool_name:
            r = step["result"]
            return r if isinstance(r, list) else []
    return []


def _check_amounts(case_state: dict, final_ans: dict, exp: dict, trace: list = None) -> bool:
    """
    Verify integer cent amounts.

    For resolved cases: check no exception is open AND that the observed
    payment + credit memo totals match expected_amount_paid_cents /
    expected_credit_cents to the cent.

    For exception/escalated cases: check exception amount matches expected.
    """
    expected_status = exp["expected_status"]
    trace = trace or []

    if expected_status == "resolved":
        # No exception should be open.
        if case_state.get("exception") is not None:
            return False

        # Verify actual payment and credit-memo totals from the trace.
        exp_invoice_id   = exp.get("expected_invoice_id", "")
        exp_paid_cents   = exp.get("expected_amount_paid_cents")
        exp_credit_cents = exp.get("expected_credit_cents", 0)

        if exp_paid_cents is not None:
            payments = _extract_from_trace(trace, "search_payments")
            relevant = [p for p in payments if p.get("invoice_id") == exp_invoice_id]
            total_paid = sum(int(p["amount_cents"]) for p in relevant)
            if total_paid != exp_paid_cents:
                return False

        if exp_credit_cents:
            memos = _extract_from_trace(trace, "search_credit_memos")
            relevant_m = [m for m in memos if m.get("invoice_id") == exp_invoice_id]
            total_credit = sum(int(m["amount_cents"]) for m in relevant_m)
            if total_credit != exp_credit_cents:
                return False

        return True

    if expected_status == "exception_open":
        exc = case_state.get("exception")
        if exc is None:
            return False
        expected_amt = exp.get("expected_amount_remaining_cents")
        if expected_amt is not None and exc.get("amount_cents") != expected_amt:
            return False
        return True

    if expected_status == "escalated":
        exc = case_state.get("exception")
        if exc is None:
            return False
        expected_amt = exp.get("expected_amount_due_cents")
        if expected_amt is not None and exc.get("amount_cents") != expected_amt:
            return False
        return True

    return True


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class Benchmark:
    def __init__(self, env: "MockEnvironment"):
        self.env      = env
        self.tasks    = load_tasks()
        self.expected = load_expected()

    def run_agent(self, agent) -> list[dict]:
        """Run agent on all tasks. Returns list of per-task score dicts."""
        results = []
        for task in self.tasks:
            self.env.reset()
            agent.run(task, self.env)
            score = score_task(task, self.env, self.expected)
            results.append(score)
        return results

    @staticmethod
    def aggregate(results: list[dict]) -> dict:
        n = len(results)
        strict = sum(1 for r in results if r["strict_pass"])

        # Per-criterion accuracy.
        criteria_keys = list(results[0]["criteria"].keys()) if results else []
        criteria_acc  = {
            k: round(sum(1 for r in results if r["criteria"][k]) / n, 3)
            for k in criteria_keys
        }

        avg_total  = round(sum(r["total"] for r in results) / n, 3)
        avg_calls  = round(sum(r["tool_calls"] for r in results) / n, 1)
        avg_scans  = round(sum(r["broad_scans"] for r in results) / n, 1)

        return {
            "strict_pass_rate": f"{strict}/{n}",
            "strict_pass_pct":  round(strict / n, 3),
            "avg_score":        avg_total,
            "avg_tool_calls":   avg_calls,
            "avg_broad_scans":  avg_scans,
            "criteria_accuracy": criteria_acc,
        }

    @staticmethod
    def print_report(agent_name: str, results: list[dict], agg: dict):
        sep = "-" * 60
        print(f"\n{'=' * 60}")
        print(f"  Agent: {agent_name.upper()}")
        print(f"{'=' * 60}")

        for r in results:
            icon = "PASS" if r["strict_pass"] else "FAIL"
            print(f"\n[{icon}] {r['task_id']}")
            print(f"  score={r['total']}  calls={r['tool_calls']}  broad_scans={r['broad_scans']}")
            for k, v in r["criteria"].items():
                mark = "OK" if v else "XX"
                print(f"    {mark}  {k}")

        print(f"\n{sep}")
        print(f"  AGGREGATE — {agent_name}")
        print(sep)
        print(f"  Strict pass rate : {agg['strict_pass_rate']}  ({agg['strict_pass_pct']*100:.0f}%)")
        print(f"  Avg score        : {agg['avg_score']}")
        print(f"  Avg tool calls   : {agg['avg_tool_calls']}")
        print(f"  Avg broad scans  : {agg['avg_broad_scans']}")
        print(f"  Criteria accuracy:")
        for k, v in agg["criteria_accuracy"].items():
            print(f"    {k:22s} {v*100:5.1f}%")
        print(sep)
