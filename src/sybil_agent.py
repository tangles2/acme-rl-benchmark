"""
SybilAgent — LLM baseline via the Sybil/Targon inference API.

Uses the openai Python SDK pointed at https://api.sybil.com/v1 (OpenAI-compatible).
The model receives the full tool schema and a system prompt describing the workflow.
It autonomously decides which tools to call and in what order, just like a frontier
model would in production.

This is the "before" baseline in the LLM-vs-LoRA comparison:
    SybilAgent (zero-shot frontier LLM) → LoRAAgent (fine-tuned tiny model)

The agent uses model: openai/gpt-oss-20b
  - Cheapest Sybil model with tool-calling support
  - ~$0.00001 per full task run (negligible)
  - Supports structured tool use and json_mode

Set SYBIL_API_KEY in your environment or .env file before running.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.environment import MockEnvironment

# Sybil inference endpoint (OpenAI-compatible)
SYBIL_BASE_URL = "https://api.sybil.com/v1"
DEFAULT_MODEL  = "openai/gpt-oss-20b"

# System prompt: describes the workflow clearly but does NOT spell out
# the optimization rules that the fine-tuned model learns. This ensures
# the baseline reflects real zero-shot LLM behavior, not a hand-tuned prompt.
SYSTEM_PROMPT = """You are a finance operations agent for Acme, Inc.

Your job is to resolve collections cases by checking invoice, payment, and credit
memo records, then taking the appropriate action.

Workflow:
1. Always start by reading the case metadata with get_case.
2. Look up the customer record with lookup_customer using the customer_id from the case (not just the name).
3. Search invoices and payments for the specific invoice in the case.
4. If payment is less than the invoice total, check credit memos before concluding there is a shortfall.
5. Based on the evidence, either:
   - Resolve as paid_in_full (and draft a Slack update)
   - Open an exception for partial_payment with the remaining amount
   - Escalate as missing_payment_evidence if no payment record exists
6. Always call final_answer last with your conclusion and the evidence IDs you used.

Key rules:
- Use customer_id from the case record, not fuzzy customer names.
- Cite only evidence IDs that you actually retrieved from tool results.
- Compute all amounts as integer cents.
- Do not resolve a case until you have seen the invoice and payment records.
"""


def build_openai_tools() -> list[dict]:
    """
    Convert tool_schema.json into the OpenAI function-calling format.

    Each tool becomes a JSON schema with the required and optional arguments.
    The environment validates actual argument types at runtime.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_case",
                "description": "Fetch one finance case by case_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "string", "description": "The case ID to retrieve."}
                    },
                    "required": ["case_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_customer",
                "description": "Fetch customer metadata by customer_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string", "description": "The CRM customer ID."}
                    },
                    "required": ["customer_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_invoices",
                "description": "Return invoices for a customer. Always provide invoice_id and month when available to avoid a broad scan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string"},
                        "invoice_id":  {"type": "string", "description": "Filter to this specific invoice ID."},
                        "month":       {"type": "string", "description": "Filter to this month (YYYY-MM)."},
                    },
                    "required": ["customer_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_payments",
                "description": "Return payments for a customer. Provide invoice_id to narrow the search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string"},
                        "invoice_id":  {"type": "string"},
                    },
                    "required": ["customer_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_credit_memos",
                "description": "Return credit memos for a customer and optional invoice_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string"},
                        "invoice_id":  {"type": "string"},
                    },
                    "required": ["customer_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_case",
                "description": "Mutate case status and resolution. Only call after observing invoice and payment evidence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "case_id":    {"type": "string"},
                        "status":     {"type": "string", "enum": ["resolved", "exception_open", "escalated"]},
                        "resolution": {"type": "string"},
                    },
                    "required": ["case_id", "status", "resolution"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_exception",
                "description": "Open an exception for unresolved discrepancies. Requires invoice evidence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "case_id":      {"type": "string"},
                        "reason":       {"type": "string"},
                        "amount_cents": {"type": "integer"},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["case_id", "reason", "amount_cents", "evidence_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "draft_slack_message",
                "description": "Draft a Slack notification with supporting evidence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "case_id":      {"type": "string"},
                        "channel":      {"type": "string"},
                        "text":         {"type": "string"},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["case_id", "channel", "text", "evidence_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "final_answer",
                "description": "Submit the final benchmark answer. Call this last with your conclusion and all evidence IDs used.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer":       {"type": "string"},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["answer", "evidence_ids"],
                },
            },
        },
    ]


class SybilAgent:
    """
    Calls a Sybil-hosted frontier LLM with tool definitions.
    The model autonomously decides which tools to call and in what order.

    This is the LLM baseline — compare against LoRAAgent (fine-tuned tiny model).
    """

    name  = "sybil_llm_baseline"
    MODEL = DEFAULT_MODEL

    def __init__(self, api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        key = api_key or os.environ.get("SYBIL_API_KEY")
        if not key:
            raise ValueError(
                "SYBIL_API_KEY not set. Export it or pass api_key= to SybilAgent()."
            )
        self.client = OpenAI(api_key=key, base_url=SYBIL_BASE_URL)
        self.tools  = build_openai_tools()

    def run(self, task: dict, env: "MockEnvironment") -> list[dict]:
        env.reset()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": task["user_request"]},
        ]

        for _ in range(14):  # safety ceiling
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0,
            )

            msg = response.choices[0].message

            # Append assistant turn (may include tool_calls).
            messages.append(msg.model_dump(exclude_unset=True))

            if not msg.tool_calls:
                # Model chose to stop without calling final_answer — treat as done.
                break

            # Execute each tool call against the mock environment.
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                method = getattr(env, tool_name, None)
                if method is None:
                    result = {"error": f"unknown tool: {tool_name}"}
                else:
                    result = method(**args)

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(result, default=str),
                })

                if tool_name == "final_answer":
                    return env.trace

        return env.trace
