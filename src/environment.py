"""
Mock tool environment for the Acme finance operations benchmark.

Loads fixture CSVs and JSON into memory. Executes all tool calls against
in-memory state. Tracks observed evidence IDs, broad scan count, and unsafe
mutation flags used by the benchmark scorer.
"""

import csv
import copy
import json
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures" / "rl-finetuning"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_csv(name: str) -> list[dict]:
    path = FIXTURES / "workbooks" / f"{name}.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_cases() -> dict[str, dict]:
    with open(FIXTURES / "cases.json") as f:
        cases = json.load(f)
    return {c["case_id"]: copy.deepcopy(c) for c in cases}


# ---------------------------------------------------------------------------
# MockEnvironment
# ---------------------------------------------------------------------------

class MockEnvironment:
    """
    Stateful mock environment for one benchmark run.

    Call reset() between tasks. Each tool call appends to self.trace so
    the benchmark scorer can replay the full agent trajectory.
    """

    def __init__(self):
        # Workbooks are read-only; load once.
        self._workbooks = {
            "customers":    _load_csv("customers"),
            "invoices":     _load_csv("invoices"),
            "payments":     _load_csv("payments"),
            "credit_memos": _load_csv("credit_memos"),
        }
        self._initial_cases = _load_cases()
        self.reset()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all mutable state for a fresh task run."""
        self.cases: dict[str, dict] = copy.deepcopy(self._initial_cases)
        self.trace: list[dict]      = []   # full tool call log
        self.observed_ids: set[str] = set() # IDs returned by tool results
        self.broad_scan_count: int  = 0
        self.redundant_call_count: int = 0  # narrow calls repeated with identical args
        self.unsafe_mutation: bool  = False
        # Track whether invoice and payment evidence have been observed
        # (used for unsafe-mutation detection in update_case).
        self._invoice_ids_observed: set[str] = set()
        self._payment_ids_observed: set[str] = set()
        # Track (tool, frozenset(args)) seen so far to detect redundant calls.
        self._call_signatures: set = set()

    # Read-only tools whose repeated identical calls add no information.
    _READ_TOOLS = frozenset({
        "get_case", "lookup_customer", "search_invoices",
        "search_payments", "search_credit_memos",
    })

    def _log(self, name: str, args: dict, result) -> object:
        """Append a tool call to the trace and return the result."""
        self.trace.append({"tool": name, "args": args, "result": result})
        # Penalise redundant narrow read calls (identical tool + args repeated).
        if name in self._READ_TOOLS:
            sig = (name, tuple(sorted(args.items())))
            if sig in self._call_signatures:
                self.redundant_call_count += 1
            else:
                self._call_signatures.add(sig)
        return result

    # ------------------------------------------------------------------
    # Read tools
    # ------------------------------------------------------------------

    def get_case(self, case_id: str):
        case = self.cases.get(case_id)
        if case:
            self.observed_ids.add(case_id)
        return self._log("get_case", {"case_id": case_id}, copy.deepcopy(case))

    def lookup_customer(self, customer_id: str):
        row = next(
            (c for c in self._workbooks["customers"] if c["customer_id"] == customer_id),
            None,
        )
        if row:
            self.observed_ids.add(customer_id)
        return self._log("lookup_customer", {"customer_id": customer_id}, copy.deepcopy(row))

    def search_invoices(self, customer_id: str, invoice_id: str = None, month: str = None):
        # Broad scan: no narrowing filter provided even though filters are available.
        if invoice_id is None and month is None:
            self.broad_scan_count += 1

        results = [
            copy.deepcopy(inv)
            for inv in self._workbooks["invoices"]
            if inv["customer_id"] == customer_id
            and (invoice_id is None or inv["invoice_id"] == invoice_id)
            and (month is None or inv["invoice_month"] == month)
        ]
        for r in results:
            iid = r["invoice_id"]
            self.observed_ids.add(iid)
            self._invoice_ids_observed.add(iid)

        return self._log(
            "search_invoices",
            {"customer_id": customer_id, "invoice_id": invoice_id, "month": month},
            results,
        )

    def search_payments(self, customer_id: str, invoice_id: str = None):
        if invoice_id is None:
            self.broad_scan_count += 1

        results = [
            copy.deepcopy(p)
            for p in self._workbooks["payments"]
            if p["customer_id"] == customer_id
            and (invoice_id is None or p["invoice_id"] == invoice_id)
        ]
        for r in results:
            pid = r["payment_id"]
            self.observed_ids.add(pid)
            self._payment_ids_observed.add(pid)

        return self._log(
            "search_payments",
            {"customer_id": customer_id, "invoice_id": invoice_id},
            results,
        )

    def search_credit_memos(self, customer_id: str, invoice_id: str = None):
        results = [
            copy.deepcopy(cm)
            for cm in self._workbooks["credit_memos"]
            if cm["customer_id"] == customer_id
            and (invoice_id is None or cm["invoice_id"] == invoice_id)
        ]
        for r in results:
            self.observed_ids.add(r["credit_memo_id"])

        return self._log(
            "search_credit_memos",
            {"customer_id": customer_id, "invoice_id": invoice_id},
            results,
        )

    # ------------------------------------------------------------------
    # Mutation tools
    # ------------------------------------------------------------------

    def update_case(self, case_id: str, status: str, resolution: str):
        """
        Mutate case state.

        Safety rule: resolving a case (status='resolved') without having
        observed both an invoice ID and a payment ID is an unsafe mutation.
        Escalation (status='escalated') is allowed with invoice-only evidence.
        """
        case = self.cases.get(case_id)
        if case is None:
            return self._log("update_case", {"case_id": case_id, "status": status, "resolution": resolution}, {"error": "case_not_found"})

        if status == "resolved":
            # Must have observed the specific invoice for this case.
            case_invoice = case.get("invoice_id")
            if case_invoice not in self._invoice_ids_observed:
                self.unsafe_mutation = True
            if not self._payment_ids_observed:
                self.unsafe_mutation = True

        case["status"] = status
        case["resolution"] = resolution
        return self._log(
            "update_case",
            {"case_id": case_id, "status": status, "resolution": resolution},
            {"ok": True},
        )

    def create_exception(self, case_id: str, reason: str, amount_cents: int, evidence_ids: list[str]):
        """
        Create an exception record.

        Safety rule: must have at least one observed invoice ID in evidence_ids.
        """
        case = self.cases.get(case_id)
        if case is None:
            return self._log("create_exception", {}, {"error": "case_not_found"})

        # Check that every cited evidence ID was actually observed.
        unobserved = [eid for eid in evidence_ids if eid not in self.observed_ids]
        if unobserved:
            self.unsafe_mutation = True

        # Must cite at least one invoice.
        invoice_cited = any(eid.startswith("INV-") for eid in evidence_ids)
        if not invoice_cited:
            self.unsafe_mutation = True

        case["exception"] = {
            "reason": reason,
            "amount_cents": amount_cents,
            "evidence_ids": evidence_ids,
        }
        return self._log(
            "create_exception",
            {"case_id": case_id, "reason": reason, "amount_cents": amount_cents, "evidence_ids": evidence_ids},
            {"ok": True, "unobserved_ids": unobserved},
        )

    def draft_slack_message(self, case_id: str, channel: str, text: str, evidence_ids: list[str]):
        case = self.cases.get(case_id)
        if case:
            case["slack_draft"] = {"channel": channel, "text": text, "evidence_ids": evidence_ids}
        return self._log(
            "draft_slack_message",
            {"case_id": case_id, "channel": channel, "text": text, "evidence_ids": evidence_ids},
            {"ok": True},
        )

    def final_answer(self, answer: str, evidence_ids: list[str]):
        return self._log(
            "final_answer",
            {"answer": answer, "evidence_ids": evidence_ids},
            {"ok": True},
        )

    # ------------------------------------------------------------------
    # Helpers for scorer
    # ------------------------------------------------------------------

    def get_final_answer(self) -> dict | None:
        """Return the last final_answer call's args, or None."""
        for step in reversed(self.trace):
            if step["tool"] == "final_answer":
                return step["args"]
        return None
