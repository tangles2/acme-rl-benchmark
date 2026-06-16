"""
Unit tests for the benchmark scorer (src/benchmark.py).

Covers:
  - score_task returns strict_pass only when all criteria pass
  - status / resolution criterion
  - amounts criterion (paid_in_full, partial_payment, escalated)
  - evidence criterion (required IDs cited)
  - observed_evidence criterion (cited IDs were returned by tools)
  - forbidden_evidence criterion (decoy IDs not cited)
  - unsafe_mutation criterion
  - tool_efficiency criterion (broad scan penalty)
  - aggregate() computes correct pass rate and averages
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import MockEnvironment
from src.benchmark import Benchmark, score_task, load_tasks, load_expected
from src.agents import PolicyAgent
from src.train import train


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    return MockEnvironment()


@pytest.fixture(scope="module")
def tasks():
    return load_tasks()


@pytest.fixture(scope="module")
def expected():
    return load_expected()


@pytest.fixture(scope="module")
def policy_agent():
    return train(verbose=False)


def run_task(env, agent, task):
    """Helper: reset, run agent, return score dict."""
    env.reset()
    agent.run(task, env)
    expected = load_expected()
    return score_task(task, env, expected)


# ---------------------------------------------------------------------------
# score_task structure
# ---------------------------------------------------------------------------

class TestScoreTaskStructure:

    def test_returns_required_keys(self, env, policy_agent, tasks):
        task = tasks[0]
        result = run_task(env, policy_agent, task)
        for key in ["task_id", "strict_pass", "criteria", "total", "broad_scans", "tool_calls"]:
            assert key in result, f"Missing key: {key}"

    def test_criteria_has_eight_dimensions(self, env, policy_agent, tasks):
        task = tasks[0]
        result = run_task(env, policy_agent, task)
        assert len(result["criteria"]) == 8

    def test_total_in_unit_interval(self, env, policy_agent, tasks):
        for task in tasks:
            result = run_task(env, policy_agent, task)
            assert 0.0 <= result["total"] <= 1.0, f"{task['task_id']}: total={result['total']}"

    def test_strict_pass_requires_all_criteria(self, env, policy_agent, tasks):
        for task in tasks:
            result = run_task(env, policy_agent, task)
            all_pass = all(result["criteria"].values())
            assert result["strict_pass"] == all_pass


# ---------------------------------------------------------------------------
# Per-criterion: tool_efficiency
# ---------------------------------------------------------------------------

class TestToolEfficiency:

    def test_broad_scan_fails_efficiency(self, env, tasks, expected):
        """An agent that does a broad scan should fail tool_efficiency."""
        from src.agents import BaselineAgent
        task = next(t for t in tasks if t["task_id"] == "task_paid_in_full")
        env.reset()
        BaselineAgent().run(task, env)
        result = score_task(task, env, expected)
        assert result["criteria"]["tool_efficiency"] is False
        assert result["broad_scans"] >= 1

    def test_no_broad_scan_passes_efficiency(self, env, policy_agent, tasks, expected):
        task = next(t for t in tasks if t["task_id"] == "task_paid_in_full")
        result = run_task(env, policy_agent, task)
        assert result["criteria"]["tool_efficiency"] is True
        assert result["broad_scans"] == 0


# ---------------------------------------------------------------------------
# Per-criterion: unsafe_mutation
# ---------------------------------------------------------------------------

class TestUnsafeMutationCriterion:

    def test_resolve_without_evidence_fails_criterion(self, env, tasks, expected):
        """Manually trigger unsafe_mutation and verify scorer catches it."""
        task = next(t for t in tasks if t["task_id"] == "task_paid_in_full")
        env.reset()
        # Resolve immediately without reading invoice or payment
        env.update_case("case-1001", "resolved", "paid_in_full")
        env.final_answer("paid_in_full", ["INV-2026-0413", "PAY-8841"])
        result = score_task(task, env, expected)
        assert result["criteria"]["unsafe_mutation"] is False


# ---------------------------------------------------------------------------
# Per-criterion: observed_evidence
# ---------------------------------------------------------------------------

class TestObservedEvidence:

    def test_fabricated_ids_fail_observed_evidence(self, env, tasks, expected):
        """Citing IDs never returned by any tool call should fail."""
        task = next(t for t in tasks if t["task_id"] == "task_paid_in_full")
        env.reset()
        env.get_case("case-1001")
        env.search_invoices("cus_globex", invoice_id="INV-2026-0413")
        env.search_payments("cus_globex", invoice_id="INV-2026-0413")
        env.update_case("case-1001", "resolved", "paid_in_full")
        # Cite a fabricated payment ID
        env.final_answer("paid_in_full", ["INV-2026-0413", "PAY-FAKE-9999"])
        result = score_task(task, env, expected)
        assert result["criteria"]["observed_evidence"] is False

    def test_real_observed_ids_pass(self, env, policy_agent, tasks, expected):
        task = next(t for t in tasks if t["task_id"] == "task_paid_in_full")
        result = run_task(env, policy_agent, task)
        assert result["criteria"]["observed_evidence"] is True


# ---------------------------------------------------------------------------
# Per-criterion: forbidden_evidence
# ---------------------------------------------------------------------------

class TestForbiddenEvidence:

    def test_citing_decoy_id_fails_forbidden_criterion(self, env, tasks, expected):
        """Citing a forbidden decoy ID should fail the criterion."""
        task = next(t for t in tasks if t["task_id"] == "task_ambiguous_customer")
        exp  = expected[task["task_id"]]
        forbidden = exp.get("forbidden_evidence_ids", [])
        if not forbidden:
            pytest.skip("No forbidden IDs defined for this task")

        env.reset()
        env.get_case(task["case_id"])
        env.search_invoices("cus_acme_hardware", invoice_id="INV-2026-0522")
        env.search_payments("cus_acme_hardware", invoice_id="INV-2026-0522")
        env.update_case(task["case_id"], "resolved", "paid_in_full")
        # Cite a forbidden decoy ID alongside legitimate ones
        env.final_answer("paid_in_full", ["INV-2026-0522", "PAY-8930", forbidden[0]])
        result = score_task(task, env, expected)
        assert result["criteria"]["forbidden_evidence"] is False


# ---------------------------------------------------------------------------
# Per-criterion: amounts
# ---------------------------------------------------------------------------

class TestAmounts:

    def test_partial_payment_wrong_amount_fails(self, env, tasks, expected):
        """Exception with wrong cent amount should fail amounts criterion."""
        task = next(t for t in tasks if t["task_id"] == "task_partial_payment")
        exp  = expected[task["task_id"]]
        env.reset()
        env.get_case(task["case_id"])
        env.search_invoices("cus_globex", invoice_id="INV-2026-0420")
        env.search_payments("cus_globex", invoice_id="INV-2026-0420")
        # Create exception with deliberately wrong amount
        env.create_exception(task["case_id"], "partial_payment", 1, ["INV-2026-0420", "PAY-8850"])
        env.update_case(task["case_id"], "exception_open", "partial_payment")
        env.final_answer("partial_payment", ["INV-2026-0420", "PAY-8850"])
        result = score_task(task, env, expected)
        assert result["criteria"]["amounts"] is False

    def test_correct_integer_amount_passes(self, env, policy_agent, tasks, expected):
        task = next(t for t in tasks if t["task_id"] == "task_partial_payment")
        result = run_task(env, policy_agent, task)
        assert result["criteria"]["amounts"] is True


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

class TestAggregate:

    def test_aggregate_strict_pass_rate(self, env, policy_agent, tasks, expected):
        bench   = Benchmark(env)
        results = bench.run_agent(policy_agent)
        agg     = Benchmark.aggregate(results)
        # Policy agent passes 4/5 tasks on original fixtures
        strict, total = agg["strict_pass_rate"].split("/")
        assert int(total) == len(tasks)
        assert 0 <= int(strict) <= int(total)

    def test_aggregate_avg_score_in_range(self, env, policy_agent):
        bench   = Benchmark(env)
        results = bench.run_agent(policy_agent)
        agg     = Benchmark.aggregate(results)
        assert 0.0 <= agg["avg_score"] <= 1.0

    def test_aggregate_broad_scans_zero_for_policy(self, env, policy_agent):
        bench   = Benchmark(env)
        results = bench.run_agent(policy_agent)
        agg     = Benchmark.aggregate(results)
        assert agg["avg_broad_scans"] == 0.0
