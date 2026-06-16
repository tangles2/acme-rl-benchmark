"""
Synthetic task variants and trace collection for V2 training data expansion.

Generates 8 new tasks across 4 task types (paid_in_full x2, partial_payment x2,
credit_memo x3, ambiguous_customer x1). Runs PolicyAgent on each to collect
labeled traces. Combines with original traces to give ~91 training examples
vs 28 in V1 -- enough to fix the search_credit_memos class imbalance.

task_missing_evidence type is intentionally excluded from synthetic generation
and used as the hidden eval split.

Usage:
    from src.v2.synthetic import build_combined_traces, SyntheticMockEnvironment, V2Benchmark
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT     = Path(__file__).parent.parent.parent
FIXTURES = ROOT / "fixtures" / "rl-finetuning"

# ---------------------------------------------------------------------------
# Synthetic fixture data
# ---------------------------------------------------------------------------

SYNTHETIC_CUSTOMERS = [
    {"customer_id": "cus_pied_piper",        "name": "Pied Piper LLC",        "crm_account_id": "crm_007", "region": "NA", "collections_channel": "#collections"},
    {"customer_id": "cus_vandelay",          "name": "Vandelay Industries",   "crm_account_id": "crm_008", "region": "NA", "collections_channel": "#collections"},
    {"customer_id": "cus_hooli",             "name": "Hooli Inc",             "crm_account_id": "crm_009", "region": "NA", "collections_channel": "#collections"},
    {"customer_id": "cus_bluth",             "name": "Bluth Company",         "crm_account_id": "crm_010", "region": "NA", "collections_channel": "#collections"},
    {"customer_id": "cus_prestige",          "name": "Prestige Worldwide",    "crm_account_id": "crm_011", "region": "NA", "collections_channel": "#collections"},
    {"customer_id": "cus_dunder",            "name": "Dunder Mifflin",        "crm_account_id": "crm_012", "region": "NA", "collections_channel": "#collections"},
    {"customer_id": "cus_gekko",             "name": "Gekko Capital",         "crm_account_id": "crm_013", "region": "NA", "collections_channel": "#collections"},
    {"customer_id": "cus_sterling_cooper",   "name": "Sterling Cooper",       "crm_account_id": "crm_014", "region": "NA", "collections_channel": "#collections"},
    {"customer_id": "cus_sterling_partners", "name": "Sterling Partners LLC", "crm_account_id": "crm_015", "region": "EU", "collections_channel": "#collections-emea"},
]

SYNTHETIC_INVOICES = [
    {"invoice_id": "INV-2026-0701", "customer_id": "cus_pied_piper",        "invoice_month": "2026-07", "amount_cents": "9500000",  "currency": "USD", "due_date": "2026-08-15", "status": "issued"},
    {"invoice_id": "INV-2026-0702", "customer_id": "cus_vandelay",          "invoice_month": "2026-07", "amount_cents": "7250000",  "currency": "USD", "due_date": "2026-08-15", "status": "issued"},
    {"invoice_id": "INV-2026-0703", "customer_id": "cus_hooli",             "invoice_month": "2026-07", "amount_cents": "5800000",  "currency": "USD", "due_date": "2026-08-15", "status": "issued"},
    {"invoice_id": "INV-2026-0704", "customer_id": "cus_bluth",             "invoice_month": "2026-07", "amount_cents": "8100000",  "currency": "USD", "due_date": "2026-08-15", "status": "issued"},
    {"invoice_id": "INV-2026-0705", "customer_id": "cus_prestige",          "invoice_month": "2026-07", "amount_cents": "6000000",  "currency": "USD", "due_date": "2026-08-15", "status": "issued"},
    {"invoice_id": "INV-2026-0706", "customer_id": "cus_dunder",            "invoice_month": "2026-07", "amount_cents": "10500000", "currency": "USD", "due_date": "2026-08-15", "status": "issued"},
    {"invoice_id": "INV-2026-0707", "customer_id": "cus_gekko",             "invoice_month": "2026-07", "amount_cents": "4800000",  "currency": "USD", "due_date": "2026-08-15", "status": "issued"},
    {"invoice_id": "INV-2026-0708", "customer_id": "cus_sterling_cooper",   "invoice_month": "2026-07", "amount_cents": "3500000",  "currency": "USD", "due_date": "2026-08-15", "status": "issued"},
    {"invoice_id": "INV-2026-0709", "customer_id": "cus_sterling_partners", "invoice_month": "2026-07", "amount_cents": "3500000",  "currency": "USD", "due_date": "2026-08-15", "status": "issued"},
]

SYNTHETIC_PAYMENTS = [
    {"payment_id": "PAY-9001", "customer_id": "cus_pied_piper",        "invoice_id": "INV-2026-0701", "amount_cents": "9500000", "currency": "USD", "received_at": "2026-07-15", "bank_ref": "BNK-PP-0701"},
    {"payment_id": "PAY-9002", "customer_id": "cus_vandelay",          "invoice_id": "INV-2026-0702", "amount_cents": "7250000", "currency": "USD", "received_at": "2026-07-16", "bank_ref": "BNK-VD-0702"},
    {"payment_id": "PAY-9003", "customer_id": "cus_hooli",             "invoice_id": "INV-2026-0703", "amount_cents": "3200000", "currency": "USD", "received_at": "2026-07-15", "bank_ref": "BNK-HL-0703"},
    {"payment_id": "PAY-9004", "customer_id": "cus_bluth",             "invoice_id": "INV-2026-0704", "amount_cents": "4500000", "currency": "USD", "received_at": "2026-07-17", "bank_ref": "BNK-BL-0704"},
    {"payment_id": "PAY-9005", "customer_id": "cus_prestige",          "invoice_id": "INV-2026-0705", "amount_cents": "5200000", "currency": "USD", "received_at": "2026-07-14", "bank_ref": "BNK-PR-0705"},
    {"payment_id": "PAY-9006", "customer_id": "cus_dunder",            "invoice_id": "INV-2026-0706", "amount_cents": "9000000", "currency": "USD", "received_at": "2026-07-15", "bank_ref": "BNK-DM-0706"},
    {"payment_id": "PAY-9007", "customer_id": "cus_gekko",             "invoice_id": "INV-2026-0707", "amount_cents": "4200000", "currency": "USD", "received_at": "2026-07-16", "bank_ref": "BNK-GK-0707"},
    {"payment_id": "PAY-9008", "customer_id": "cus_sterling_cooper",   "invoice_id": "INV-2026-0708", "amount_cents": "3500000", "currency": "USD", "received_at": "2026-07-15", "bank_ref": "BNK-SC-0708"},
    {"payment_id": "PAY-9009", "customer_id": "cus_sterling_partners", "invoice_id": "INV-2026-0709", "amount_cents": "3500000", "currency": "USD", "received_at": "2026-07-16", "bank_ref": "BNK-SP-0709"},
]

SYNTHETIC_CREDIT_MEMOS = [
    {"credit_memo_id": "CM-201", "customer_id": "cus_prestige", "invoice_id": "INV-2026-0705", "amount_cents": "800000",  "reason": "service_credit",            "issued_at": "2026-07-05"},
    {"credit_memo_id": "CM-202", "customer_id": "cus_dunder",   "invoice_id": "INV-2026-0706", "amount_cents": "1500000", "reason": "prior_month_adjustment",    "issued_at": "2026-07-06"},
    {"credit_memo_id": "CM-203", "customer_id": "cus_gekko",    "invoice_id": "INV-2026-0707", "amount_cents": "600000",  "reason": "service_credit",            "issued_at": "2026-07-07"},
]

SYNTHETIC_CASES = [
    {"case_id": "case-2001", "customer_id": "cus_pied_piper",      "customer_name": "Pied Piper LLC",      "invoice_id": "INV-2026-0701", "month": "2026-07", "status": "open", "resolution": None, "exception": None, "slack_draft": None},
    {"case_id": "case-2002", "customer_id": "cus_vandelay",        "customer_name": "Vandelay Industries", "invoice_id": "INV-2026-0702", "month": "2026-07", "status": "open", "resolution": None, "exception": None, "slack_draft": None},
    {"case_id": "case-2003", "customer_id": "cus_hooli",           "customer_name": "Hooli Inc",           "invoice_id": "INV-2026-0703", "month": "2026-07", "status": "open", "resolution": None, "exception": None, "slack_draft": None},
    {"case_id": "case-2004", "customer_id": "cus_bluth",           "customer_name": "Bluth Company",       "invoice_id": "INV-2026-0704", "month": "2026-07", "status": "open", "resolution": None, "exception": None, "slack_draft": None},
    {"case_id": "case-2005", "customer_id": "cus_prestige",        "customer_name": "Prestige Worldwide",  "invoice_id": "INV-2026-0705", "month": "2026-07", "status": "open", "resolution": None, "exception": None, "slack_draft": None},
    {"case_id": "case-2006", "customer_id": "cus_dunder",          "customer_name": "Dunder Mifflin",      "invoice_id": "INV-2026-0706", "month": "2026-07", "status": "open", "resolution": None, "exception": None, "slack_draft": None},
    {"case_id": "case-2007", "customer_id": "cus_gekko",           "customer_name": "Gekko Capital",       "invoice_id": "INV-2026-0707", "month": "2026-07", "status": "open", "resolution": None, "exception": None, "slack_draft": None},
    {"case_id": "case-2008", "customer_id": "cus_sterling_cooper", "customer_name": "Sterling Cooper",     "invoice_id": "INV-2026-0708", "month": "2026-07", "status": "open", "resolution": None, "exception": None, "slack_draft": None},
]

SYNTHETIC_TASKS = [
    {"task_id": "task_s_paid_1",    "case_id": "case-2001", "difficulty": "easy",   "user_request": "Pied Piper LLC says their July invoice INV-2026-0701 has been settled. Confirm full payment and resolve the case with supporting evidence.",                                                        "expected_intent": "resolve_paid_invoice"},
    {"task_id": "task_s_paid_2",    "case_id": "case-2002", "difficulty": "easy",   "user_request": "Vandelay Industries claims their July invoice is closed out. Verify full payment of INV-2026-0702 and resolve if confirmed.",                                                                         "expected_intent": "resolve_paid_invoice"},
    {"task_id": "task_s_partial_1", "case_id": "case-2003", "difficulty": "medium", "user_request": "Hooli Inc says they paid invoice INV-2026-0703 but it is still open. Check payment against the invoice amount and open an exception if there is a shortfall.",                                      "expected_intent": "open_partial_payment_exception"},
    {"task_id": "task_s_partial_2", "case_id": "case-2004", "difficulty": "medium", "user_request": "Bluth Company has an open collections case for INV-2026-0704. Verify whether the payment covers the full invoice amount.",                                                                           "expected_intent": "open_partial_payment_exception"},
    {"task_id": "task_s_credit_1",  "case_id": "case-2005", "difficulty": "medium", "user_request": "Prestige Worldwide shows a payment shortfall on INV-2026-0705. Check whether a credit memo covers the difference before opening an exception.",                                                     "expected_intent": "resolve_credit_memo_reconciliation"},
    {"task_id": "task_s_credit_2",  "case_id": "case-2006", "difficulty": "medium", "user_request": "Dunder Mifflin invoice INV-2026-0706 shows partial payment. Verify if any credit memos apply to the balance before escalating.",                                                                    "expected_intent": "resolve_credit_memo_reconciliation"},
    {"task_id": "task_s_credit_3",  "case_id": "case-2007", "difficulty": "medium", "user_request": "Gekko Capital has a shortfall on INV-2026-0707. Check whether a credit memo covers the difference. Resolve if fully settled, open an exception if not.",                                           "expected_intent": "resolve_credit_memo_reconciliation"},
    {"task_id": "task_s_ambiguous", "case_id": "case-2008", "difficulty": "hard",   "user_request": "Sterling Cooper has an open case on INV-2026-0708. Use the CRM customer ID from the case to avoid mixing up with Sterling Partners LLC.",                                                            "expected_intent": "resolve_ambiguous_customer_case"},
]

SYNTHETIC_EXPECTED = {
    "task_s_paid_1": {
        "case_id": "case-2001", "expected_status": "resolved", "expected_resolution": "paid_in_full",
        "expected_invoice_id": "INV-2026-0701", "expected_payment_ids": ["PAY-9001"],
        "expected_credit_memo_ids": [], "expected_amount_due_cents": 9500000,
        "expected_amount_paid_cents": 9500000, "expected_exception": None,
        "required_evidence_ids": ["INV-2026-0701", "PAY-9001"],
    },
    "task_s_paid_2": {
        "case_id": "case-2002", "expected_status": "resolved", "expected_resolution": "paid_in_full",
        "expected_invoice_id": "INV-2026-0702", "expected_payment_ids": ["PAY-9002"],
        "expected_credit_memo_ids": [], "expected_amount_due_cents": 7250000,
        "expected_amount_paid_cents": 7250000, "expected_exception": None,
        "required_evidence_ids": ["INV-2026-0702", "PAY-9002"],
    },
    "task_s_partial_1": {
        "case_id": "case-2003", "expected_status": "exception_open", "expected_resolution": "partial_payment",
        "expected_invoice_id": "INV-2026-0703", "expected_payment_ids": ["PAY-9003"],
        "expected_credit_memo_ids": [], "expected_amount_due_cents": 5800000,
        "expected_amount_paid_cents": 3200000, "expected_amount_remaining_cents": 2600000,
        "expected_exception": {"reason": "partial_payment", "amount_cents": 2600000},
        "required_evidence_ids": ["INV-2026-0703", "PAY-9003"],
    },
    "task_s_partial_2": {
        "case_id": "case-2004", "expected_status": "exception_open", "expected_resolution": "partial_payment",
        "expected_invoice_id": "INV-2026-0704", "expected_payment_ids": ["PAY-9004"],
        "expected_credit_memo_ids": [], "expected_amount_due_cents": 8100000,
        "expected_amount_paid_cents": 4500000, "expected_amount_remaining_cents": 3600000,
        "expected_exception": {"reason": "partial_payment", "amount_cents": 3600000},
        "required_evidence_ids": ["INV-2026-0704", "PAY-9004"],
    },
    "task_s_credit_1": {
        "case_id": "case-2005", "expected_status": "resolved", "expected_resolution": "paid_after_credit_memo",
        "expected_invoice_id": "INV-2026-0705", "expected_payment_ids": ["PAY-9005"],
        "expected_credit_memo_ids": ["CM-201"], "expected_amount_due_cents": 6000000,
        "expected_amount_paid_cents": 5200000, "expected_credit_cents": 800000,
        "expected_exception": None, "required_evidence_ids": ["INV-2026-0705", "PAY-9005", "CM-201"],
    },
    "task_s_credit_2": {
        "case_id": "case-2006", "expected_status": "resolved", "expected_resolution": "paid_after_credit_memo",
        "expected_invoice_id": "INV-2026-0706", "expected_payment_ids": ["PAY-9006"],
        "expected_credit_memo_ids": ["CM-202"], "expected_amount_due_cents": 10500000,
        "expected_amount_paid_cents": 9000000, "expected_credit_cents": 1500000,
        "expected_exception": None, "required_evidence_ids": ["INV-2026-0706", "PAY-9006", "CM-202"],
    },
    "task_s_credit_3": {
        "case_id": "case-2007", "expected_status": "resolved", "expected_resolution": "paid_after_credit_memo",
        "expected_invoice_id": "INV-2026-0707", "expected_payment_ids": ["PAY-9007"],
        "expected_credit_memo_ids": ["CM-203"], "expected_amount_due_cents": 4800000,
        "expected_amount_paid_cents": 4200000, "expected_credit_cents": 600000,
        "expected_exception": None, "required_evidence_ids": ["INV-2026-0707", "PAY-9007", "CM-203"],
    },
    "task_s_ambiguous": {
        "case_id": "case-2008", "expected_status": "resolved", "expected_resolution": "paid_in_full",
        "expected_invoice_id": "INV-2026-0708", "expected_payment_ids": ["PAY-9008"],
        "expected_credit_memo_ids": [], "expected_amount_due_cents": 3500000,
        "expected_amount_paid_cents": 3500000, "expected_exception": None,
        "required_evidence_ids": ["cus_sterling_cooper", "INV-2026-0708", "PAY-9008"],
        "forbidden_evidence_ids": ["cus_sterling_partners", "INV-2026-0709", "PAY-9009"],
    },
}

# task_missing_evidence is held out as the hidden eval split
EVAL_TASK_IDS = {"task_missing_evidence"}


# ---------------------------------------------------------------------------
# SyntheticMockEnvironment
# ---------------------------------------------------------------------------

class SyntheticMockEnvironment:
    """
    MockEnvironment extended with synthetic fixture data.
    Subclasses MockEnvironment by injecting extra rows after initial load.
    """

    def __new__(cls):
        from src.environment import MockEnvironment
        instance = MockEnvironment.__new__(MockEnvironment)
        return instance

    def __init__(self):
        from src.environment import MockEnvironment
        MockEnvironment.__init__(self)
        # Inject synthetic rows into workbooks
        self._workbooks["customers"]    += SYNTHETIC_CUSTOMERS
        self._workbooks["invoices"]     += SYNTHETIC_INVOICES
        self._workbooks["payments"]     += SYNTHETIC_PAYMENTS
        self._workbooks["credit_memos"] += SYNTHETIC_CREDIT_MEMOS
        # Inject synthetic cases
        for case in SYNTHETIC_CASES:
            self._initial_cases[case["case_id"]] = copy.deepcopy(case)
        self.reset()


# ---------------------------------------------------------------------------
# V2Benchmark
# ---------------------------------------------------------------------------

class V2Benchmark:
    """
    Benchmark that runs on both original tasks and synthetic tasks.
    Reports train-task and eval-task scores separately (hidden eval split).
    """

    def __init__(self, env):
        from src.benchmark import load_tasks, load_expected
        self.env      = env
        self.tasks    = load_tasks() + SYNTHETIC_TASKS
        self.expected = {**load_expected(), **SYNTHETIC_EXPECTED}

    def run_agent(self, agent):
        results = []
        for task in self.tasks:
            self.env.reset()
            agent.run(task, self.env)
            from src.benchmark import score_task
            score = score_task(task, self.env, self.expected)
            results.append(score)
        return results

    def run_split(self, agent):
        """Returns (train_results, eval_results) split by EVAL_TASK_IDS."""
        all_results = self.run_agent(agent)
        train = [r for r in all_results if r["task_id"] not in EVAL_TASK_IDS]
        held  = [r for r in all_results if r["task_id"] in EVAL_TASK_IDS]
        return train, held

    @staticmethod
    def aggregate(results):
        from src.benchmark import Benchmark
        return Benchmark.aggregate(results)

    @staticmethod
    def print_report(agent_name, results, agg):
        from src.benchmark import Benchmark
        Benchmark.print_report(agent_name, results, agg)


# ---------------------------------------------------------------------------
# Synthetic trace collection
# ---------------------------------------------------------------------------

def collect_synthetic_traces(policy_agent):
    """
    Run PolicyAgent on all synthetic tasks and return traces in the same
    format as training_traces.jsonl -- ready to be combined with originals.
    """
    env    = SyntheticMockEnvironment()
    traces = []

    for task in SYNTHETIC_TASKS:
        env.reset()
        policy_agent.run(task, env)
        messages = []
        for step in env.trace:
            messages.append({
                "tool_call": {
                    "name":      step["tool"],
                    "arguments": step["args"],
                },
                "tool_result": step["result"],
            })
        traces.append({"input": task["user_request"], "messages": messages})

    return traces


def build_combined_traces():
    """
    Load original training_traces.jsonl, generate synthetic traces,
    and return the combined list.
    """
    original = []
    with open(FIXTURES / "training_traces.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                original.append(json.loads(line))

    from src.train import train
    policy_agent = train(verbose=False)

    synthetic = collect_synthetic_traces(policy_agent)
    combined  = original + synthetic

    print(f"[synthetic] Original traces : {len(original)}")
    print(f"[synthetic] Synthetic traces: {len(synthetic)}")
    total_steps = sum(len(t["messages"]) for t in combined)
    print(f"[synthetic] Combined total  : {len(combined)} traces, {total_steps} training steps")
    return combined
