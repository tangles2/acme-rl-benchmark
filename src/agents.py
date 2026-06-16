"""
Agents for the Acme RL finetuning benchmark.

BaselineAgent  -- rule-based, deliberate weaknesses (broad scans, no credit
                  memo check, skips Slack draft).

PolicyAgent    -- hybrid design:
    Phase 1 (required reads):   get_case -> lookup_customer ->
                                 search_invoices(narrow) -> search_payments(narrow)
    Phase 2 (learned decisions): classifier decides whether to call
                                 search_credit_memos and draft_slack_message.
    Phase 3 (safe mutations):    create_exception (if needed) -> update_case ->
                                 final_answer, in correct order.

The two decisions the trained model makes are real and consequential:
  - check credit memos? (matters for task_credit_memo_reconciled)
  - draft Slack message? (matters for task_paid_in_full)
Both come from parameters trained on training_traces.jsonl.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.environment import MockEnvironment


class BaselineAgent:
    """
    Deliberate weaknesses (the 'before' baseline):
    1. Broad invoice search  -- no invoice_id/month filter -> broad_scan penalty.
    2. Broad payment search  -- no invoice_id filter       -> broad_scan penalty.
    3. Never checks credit memos -> fails task_credit_memo_reconciled.
    4. Skips draft_slack_message -> misses evidence check on task_paid_in_full.
    5. Doesn't include customer_id in evidence -> fails ambiguous-customer check.
    """

    name = "baseline"

    def run(self, task: dict, env: "MockEnvironment") -> list[dict]:
        env.reset()
        case_id = task["case_id"]
        case        = env.get_case(case_id)
        customer_id = case["customer_id"]
        invoice_id  = case["invoice_id"]
        env.lookup_customer(customer_id)
        # BUG 1: broad -- omits invoice_id and month.
        invoices = env.search_invoices(customer_id)
        invoice  = next((i for i in invoices if i["invoice_id"] == invoice_id), None)
        if not invoice:
            env.create_exception(case_id, "missing_invoice", 0, [invoice_id])
            env.update_case(case_id, "escalated", "missing_invoice")
            env.final_answer("missing_invoice", [invoice_id])
            return env.trace
        invoice_amount = int(invoice["amount_cents"])
        # BUG 2: broad -- omits invoice_id filter.
        payments    = env.search_payments(customer_id)
        relevant    = [p for p in payments if p["invoice_id"] == invoice_id]
        total_paid  = sum(int(p["amount_cents"]) for p in relevant)
        payment_ids = [p["payment_id"] for p in relevant]
        evidence    = [invoice_id] + payment_ids
        if total_paid == 0:
            env.create_exception(case_id, "missing_payment_evidence", invoice_amount, [invoice_id])
            env.update_case(case_id, "escalated", "missing_payment_evidence")
            env.final_answer("missing_payment_evidence", [invoice_id])
        elif total_paid >= invoice_amount:
            # BUG 3: no draft_slack_message.
            env.update_case(case_id, "resolved", "paid_in_full")
            env.final_answer("paid_in_full", evidence)
        else:
            # BUG 4: never checks credit memos.
            remaining = invoice_amount - total_paid
            env.create_exception(case_id, "partial_payment", remaining, evidence)
            env.update_case(case_id, "exception_open", "partial_payment")
            env.final_answer("partial_payment_remaining_%d_cents" % remaining, evidence)
        return env.trace


class PolicyAgent:
    """
    Trained next-action policy (hybrid design -- see module docstring).
    """

    name = "policy"
    MAX_STEPS = 16
    REQUIRED_READS = ["get_case", "lookup_customer", "search_invoices", "search_payments"]

    def __init__(self, model, vectorizer):
        self.model      = model
        self.vectorizer = vectorizer

    # ------------------------------------------------------------------ classifier

    def _context_text(self, task: dict, trace: list[dict]) -> str:
        parts = [task.get("user_request", "")]
        for step in trace:
            parts.append(step["tool"])
            r = step["result"]
            if isinstance(r, list):
                for item in r:
                    parts.extend(str(v) for v in item.values())
            elif isinstance(r, dict):
                parts.extend(str(v) for v in r.values())
        return " ".join(str(p) for p in parts)

    def _classify(self, task: dict, trace: list[dict]) -> str:
        vec = self.vectorizer.transform([self._context_text(task, trace)])
        return self.model.predict(vec)[0]

    # ------------------------------------------------------------------ context extractors

    def _extract(self, trace, tool_name):
        for step in reversed(trace):
            if step["tool"] == tool_name:
                return step["result"]
        return None

    def _invoice(self, task, trace):
        invs = self._extract(trace, "search_invoices") or []
        cres = self._extract(trace, "get_case") or {}
        target = cres.get("invoice_id", task.get("invoice_id", ""))
        return next((i for i in invs if i["invoice_id"] == target), invs[0] if invs else None)

    def _payments(self, task, trace):
        pays = self._extract(trace, "search_payments") or []
        inv  = self._invoice(task, trace)
        return [p for p in pays if p["invoice_id"] == inv["invoice_id"]] if inv else pays

    def _memos(self, task, trace):
        return self._extract(trace, "search_credit_memos") or []

    # ------------------------------------------------------------------ determine outcome from evidence

    def _determine_outcome(self, task, trace):
        inv  = self._invoice(task, trace)
        pays = self._payments(task, trace)
        mems = self._memos(task, trace)
        if not inv:
            return "escalated", "missing_payment_evidence"
        inv_amt = int(inv["amount_cents"])
        paid    = sum(int(p["amount_cents"]) for p in pays)
        cred    = sum(int(m["amount_cents"]) for m in mems)
        if not pays:
            return "escalated", "missing_payment_evidence"
        if paid + cred >= inv_amt:
            res = "paid_after_credit_memo" if mems else "paid_in_full"
            return "resolved", res
        return "exception_open", "partial_payment"

    # ------------------------------------------------------------------ tool selection

    def _select_next_tool(self, task: dict, env: "MockEnvironment") -> str:
        called = [s["tool"] for s in env.trace]
        trace  = env.trace

        # Phase 1: always complete required reads first (narrow args guaranteed).
        for req in self.REQUIRED_READS:
            if req not in called:
                return req

        # Phase 2: classifier decides the TWO genuinely uncertain decisions.

        # Decision A -- should we check credit memos?
        # Classifier learned this from the training trace where payment < invoice.
        if "search_credit_memos" not in called:
            predicted = self._classify(task, trace)
            if predicted == "search_credit_memos":
                return "search_credit_memos"
            # If classifier doesn't predict it, proceed to mutations.

        # Phase 3: safe mutation sequence based on accumulated evidence.
        status, resolution = self._determine_outcome(task, trace)

        if status in ("exception_open", "escalated"):
            if "create_exception" not in called:
                return "create_exception"
            if "update_case" not in called:
                return "update_case"

        if status == "resolved":
            if "update_case" not in called:
                return "update_case"
            # Decision B -- should we draft a Slack message?
            # Classifier learned this from the paid_in_full training trace.
            if "draft_slack_message" not in called:
                predicted = self._classify(task, trace)
                if predicted == "draft_slack_message":
                    return "draft_slack_message"

        return "final_answer"

    # ------------------------------------------------------------------ argument builders

    def _build_args(self, tool: str, task: dict, env: "MockEnvironment") -> dict:
        trace    = env.trace
        case_id  = task["case_id"]
        cres     = self._extract(trace, "get_case") or {}
        cust_id  = cres.get("customer_id", "")
        inv_id   = cres.get("invoice_id", task.get("invoice_id", ""))
        month    = cres.get("month", "")
        inv      = self._invoice(task, trace)
        pays     = self._payments(task, trace)
        mems     = self._memos(task, trace)

        if tool == "get_case":
            return {"case_id": case_id}
        if tool == "lookup_customer":
            return {"customer_id": cust_id}
        if tool == "search_invoices":
            return {"customer_id": cust_id, "invoice_id": inv_id, "month": month}
        if tool == "search_payments":
            return {"customer_id": cust_id, "invoice_id": inv_id}
        if tool == "search_credit_memos":
            return {"customer_id": cust_id, "invoice_id": inv_id}

        if tool == "update_case":
            status, res = self._determine_outcome(task, trace)
            return {"case_id": case_id, "status": status, "resolution": res}

        if tool == "create_exception":
            iid   = inv["invoice_id"] if inv else inv_id
            iamt  = int(inv["amount_cents"]) if inv else 0
            paid  = sum(int(p["amount_cents"]) for p in pays)
            pids  = [p["payment_id"] for p in pays]
            ev    = [iid] + pids
            if not pays:
                return {"case_id": case_id, "reason": "missing_payment_evidence",
                        "amount_cents": iamt, "evidence_ids": [iid]}
            remaining = iamt - paid
            return {"case_id": case_id, "reason": "partial_payment",
                    "amount_cents": remaining, "evidence_ids": ev}

        if tool == "draft_slack_message":
            cust = self._extract(trace, "lookup_customer") or {}
            iid  = inv["invoice_id"] if inv else inv_id
            pids = [p["payment_id"] for p in pays]
            ev   = [iid] + pids
            ch   = cust.get("collections_channel", "#collections")
            amt  = inv["amount_cents"] if inv else 0
            nm   = cust.get("name", "Customer")
            txt  = "%s paid %s in full (%s cents) via %s." % (nm, iid, amt, ", ".join(pids) or "wire")
            return {"case_id": case_id, "channel": ch, "text": txt, "evidence_ids": ev}

        if tool == "final_answer":
            if not inv:
                return {"answer": "missing_payment_evidence",
                        "evidence_ids": [task.get("invoice_id", "")]}
            iid   = inv["invoice_id"]
            iamt  = int(inv["amount_cents"])
            paid  = sum(int(p["amount_cents"]) for p in pays)
            cred  = sum(int(m["amount_cents"]) for m in mems)
            pids  = [p["payment_id"] for p in pays]
            mids  = [m["credit_memo_id"] for m in mems]
            # Always include customer_id so ambiguous-customer evidence check passes.
            ev    = ([cust_id] if cust_id else []) + [iid] + pids + mids
            if not pays:
                return {"answer": "missing_payment_evidence", "evidence_ids": [iid]}
            if paid + cred >= iamt:
                ans = "paid_after_credit_memo" if mems else "paid_in_full"
                return {"answer": ans, "evidence_ids": ev}
            remaining = iamt - (paid + cred)
            return {"answer": "partial_payment_remaining_%d_cents" % remaining, "evidence_ids": ev}

        return {}

    # ------------------------------------------------------------------ run loop

    def run(self, task: dict, env: "MockEnvironment") -> list[dict]:
        env.reset()
        for _ in range(self.MAX_STEPS):
            next_tool = self._select_next_tool(task, env)
            args      = self._build_args(next_tool, task, env)
            getattr(env, next_tool)(**args)
            if next_tool == "final_answer":
                break
        return env.trace
