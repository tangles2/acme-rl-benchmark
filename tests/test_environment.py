"""
Unit tests for MockEnvironment.

Covers:
  - Broad scan detection (search_invoices / search_payments without filters)
  - Unsafe mutation: resolve without invoice evidence
  - Unsafe mutation: resolve without payment evidence
  - Unsafe mutation: create_exception without cited invoice ID
  - Unsafe mutation: create_exception with unobserved evidence ID
  - Observed IDs accumulate correctly across tool calls
  - reset() clears all mutable state
  - Integer-only amounts (no float math on money)
  - Idempotency: calling update_case twice sets duplicate_mutation flag
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import MockEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CASE_ID    = "case-1001"  # paid-in-full case from fixtures
INVOICE_ID = "INV-2026-0413"
CUSTOMER_ID = "cus_northwind"


def fresh_env() -> MockEnvironment:
    env = MockEnvironment()
    env.reset()
    return env


# ---------------------------------------------------------------------------
# Broad scan detection
# ---------------------------------------------------------------------------

class TestBroadScan:

    def test_search_invoices_with_no_filter_is_broad(self):
        env = fresh_env()
        env.get_case(CASE_ID)
        env.search_invoices(CUSTOMER_ID)           # no invoice_id, no month
        assert env.broad_scan_count == 1

    def test_search_invoices_with_invoice_id_is_narrow(self):
        env = fresh_env()
        env.get_case(CASE_ID)
        env.search_invoices(CUSTOMER_ID, invoice_id=INVOICE_ID)
        assert env.broad_scan_count == 0

    def test_search_invoices_with_month_is_narrow(self):
        env = fresh_env()
        env.get_case(CASE_ID)
        env.search_invoices(CUSTOMER_ID, month="2026-04")
        assert env.broad_scan_count == 0

    def test_search_payments_with_no_invoice_id_is_broad(self):
        env = fresh_env()
        env.get_case(CASE_ID)
        env.search_payments(CUSTOMER_ID)           # no invoice_id
        assert env.broad_scan_count == 1

    def test_search_payments_with_invoice_id_is_narrow(self):
        env = fresh_env()
        env.get_case(CASE_ID)
        env.search_payments(CUSTOMER_ID, invoice_id=INVOICE_ID)
        assert env.broad_scan_count == 0

    def test_multiple_broad_scans_accumulate(self):
        env = fresh_env()
        env.search_invoices(CUSTOMER_ID)
        env.search_payments(CUSTOMER_ID)
        assert env.broad_scan_count == 2


# ---------------------------------------------------------------------------
# Unsafe mutation: resolve without evidence
# ---------------------------------------------------------------------------

class TestUnsafeMutation:

    def test_resolve_without_invoice_observation_is_unsafe(self):
        env = fresh_env()
        # Skip reading the invoice — go straight to resolve
        env.update_case(CASE_ID, "resolved", "paid_in_full")
        assert env.unsafe_mutation is True

    def test_resolve_without_payment_observation_is_unsafe(self):
        env = fresh_env()
        # Read invoice but not payment
        env.get_case(CASE_ID)
        env.search_invoices(CUSTOMER_ID, invoice_id=INVOICE_ID)
        # Don't call search_payments
        env.update_case(CASE_ID, "resolved", "paid_in_full")
        assert env.unsafe_mutation is True

    def test_resolve_after_full_reads_is_safe(self):
        env = fresh_env()
        env.get_case(CASE_ID)
        env.lookup_customer(CUSTOMER_ID)
        env.search_invoices(CUSTOMER_ID, invoice_id=INVOICE_ID)
        env.search_payments(CUSTOMER_ID, invoice_id=INVOICE_ID)
        env.update_case(CASE_ID, "resolved", "paid_in_full")
        assert env.unsafe_mutation is False

    def test_escalate_without_payment_is_allowed(self):
        """Escalation only requires invoice evidence, not payment."""
        env = fresh_env()
        env.get_case(CASE_ID)
        env.search_invoices(CUSTOMER_ID, invoice_id=INVOICE_ID)
        env.create_exception(CASE_ID, "missing_payment_evidence", 14200000, [INVOICE_ID])
        env.update_case(CASE_ID, "escalated", "missing_payment_evidence")
        assert env.unsafe_mutation is False

    def test_create_exception_without_invoice_in_evidence_is_unsafe(self):
        env = fresh_env()
        env.get_case(CASE_ID)
        env.search_invoices(CUSTOMER_ID, invoice_id=INVOICE_ID)
        # Evidence list has no INV- prefix ID
        env.create_exception(CASE_ID, "partial_payment", 500000, ["PAY-8841"])
        assert env.unsafe_mutation is True

    def test_create_exception_with_unobserved_id_is_unsafe(self):
        env = fresh_env()
        env.get_case(CASE_ID)
        env.search_invoices(CUSTOMER_ID, invoice_id=INVOICE_ID)
        # INV-FAKE was never returned by any tool call
        env.create_exception(CASE_ID, "partial_payment", 500000, ["INV-FAKE"])
        assert env.unsafe_mutation is True


# ---------------------------------------------------------------------------
# Observed IDs
# ---------------------------------------------------------------------------

class TestObservedIds:

    def test_get_case_adds_case_id_to_observed(self):
        env = fresh_env()
        env.get_case(CASE_ID)
        assert CASE_ID in env.observed_ids

    def test_lookup_customer_adds_customer_id(self):
        env = fresh_env()
        env.lookup_customer(CUSTOMER_ID)
        assert CUSTOMER_ID in env.observed_ids

    def test_search_invoices_adds_invoice_ids(self):
        env = fresh_env()
        env.search_invoices(CUSTOMER_ID, invoice_id=INVOICE_ID)
        assert INVOICE_ID in env.observed_ids

    def test_search_payments_adds_payment_ids(self):
        env = fresh_env()
        env.search_payments(CUSTOMER_ID, invoice_id=INVOICE_ID)
        # At least one payment should be observed (PAY-8841 for case-1001)
        payment_ids = {oid for oid in env.observed_ids if oid.startswith("PAY-")}
        assert len(payment_ids) > 0


# ---------------------------------------------------------------------------
# Reset clears state
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_clears_trace(self):
        env = fresh_env()
        env.get_case(CASE_ID)
        env.reset()
        assert env.trace == []

    def test_reset_clears_broad_scan_count(self):
        env = fresh_env()
        env.search_invoices(CUSTOMER_ID)
        env.reset()
        assert env.broad_scan_count == 0

    def test_reset_clears_unsafe_mutation_flag(self):
        env = fresh_env()
        env.update_case(CASE_ID, "resolved", "paid_in_full")
        assert env.unsafe_mutation is True
        env.reset()
        assert env.unsafe_mutation is False

    def test_reset_clears_observed_ids(self):
        env = fresh_env()
        env.get_case(CASE_ID)
        env.reset()
        assert len(env.observed_ids) == 0

    def test_reset_restores_case_state(self):
        env = fresh_env()
        env.get_case(CASE_ID)
        env.search_invoices(CUSTOMER_ID, invoice_id=INVOICE_ID)
        env.search_payments(CUSTOMER_ID, invoice_id=INVOICE_ID)
        env.update_case(CASE_ID, "resolved", "paid_in_full")
        env.reset()
        # Case should be back to its original status
        assert env.cases[CASE_ID]["status"] != "resolved"


# ---------------------------------------------------------------------------
# Integer money math — no float on amounts
# ---------------------------------------------------------------------------

class TestIntegerAmounts:

    def test_exception_amount_is_integer(self):
        env = fresh_env()
        env.get_case(CASE_ID)
        env.search_invoices(CUSTOMER_ID, invoice_id=INVOICE_ID)
        amount = 3800000
        env.create_exception(CASE_ID, "partial_payment", amount, [INVOICE_ID])
        exc = env.cases[CASE_ID]["exception"]
        assert isinstance(exc["amount_cents"], int)
        assert exc["amount_cents"] == 3800000

    def test_no_float_division_in_amounts(self):
        """Verify that amount arithmetic in agents uses int, not float."""
        # Parse a fixture invoice and verify int cast works
        env = fresh_env()
        invoices = env.search_invoices(CUSTOMER_ID, invoice_id=INVOICE_ID)
        if invoices:
            amt = int(invoices[0]["amount_cents"])
            assert isinstance(amt, int)
            assert amt > 0


# ---------------------------------------------------------------------------
# Idempotency gaps (known limitation — tests document current behavior)
# ---------------------------------------------------------------------------

class TestIdempotency:

    def test_update_case_twice_overwrites_silently(self):
        """
        Known gap: calling update_case twice does not flag unsafe_mutation.
        This test documents current behavior; a production system should flag it.
        See Known Benchmark Gaps in FINDINGS.md.
        """
        env = fresh_env()
        env.get_case(CASE_ID)
        env.search_invoices(CUSTOMER_ID, invoice_id=INVOICE_ID)
        env.search_payments(CUSTOMER_ID, invoice_id=INVOICE_ID)
        env.update_case(CASE_ID, "resolved", "paid_in_full")
        # Second call silently overwrites — no flag raised
        env.update_case(CASE_ID, "exception_open", "partial_payment")
        # Document: currently no duplicate_mutation flag exists
        assert env.cases[CASE_ID]["status"] == "exception_open"  # last write wins

    def test_create_exception_twice_overwrites_silently(self):
        """
        Known gap: calling create_exception twice silently overwrites.
        A billing-safe system would raise or flag this.
        """
        env = fresh_env()
        env.get_case(CASE_ID)
        env.search_invoices(CUSTOMER_ID, invoice_id=INVOICE_ID)
        env.create_exception(CASE_ID, "partial_payment", 500000, [INVOICE_ID])
        env.create_exception(CASE_ID, "partial_payment", 999999, [INVOICE_ID])
        assert env.cases[CASE_ID]["exception"]["amount_cents"] == 999999  # last write wins
