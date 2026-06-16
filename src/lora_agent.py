"""
LoRAAgent -- benchmark agent backed by a LoRA fine-tuned Qwen2.5-0.5B-Instruct.

Wraps the fine-tuned adapter and runs inference step-by-step:
  1. Format current state (task + prior tool calls) as a ChatML prompt.
  2. Generate the next tool call as JSON.
  3. Parse JSON -> execute against MockEnvironment.
  4. Repeat until final_answer is called or step limit reached.

If JSON parsing fails or the model hallucinates an unknown tool,
falls back to the PolicyAgent's rule-based tool selection so the
benchmark score still reflects fine-tuning where it works.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.environment import MockEnvironment

# Known valid tool names
VALID_TOOLS = {
    "get_case", "lookup_customer", "search_invoices", "search_payments",
    "search_credit_memos", "update_case", "create_exception",
    "draft_slack_message", "final_answer",
}

SYSTEM_PROMPT = (
    "You are a finance operations agent for Acme, Inc.\n"
    "Given a task and the steps taken so far, output the next tool call as JSON:\n"
    '{"tool": "<tool_name>", "args": {<key>: <value>, ...}}\n'
    "Output ONLY the JSON, nothing else."
)


def _format_step_history(trace, up_to):
    if up_to == 0:
        return "No steps taken yet."
    parts = []
    for step in trace[:up_to]:
        tool   = step.get("tool", "?")
        args   = step.get("args", {})
        result = step.get("result", "")
        args_s = json.dumps(args, separators=(",", ":"))
        res_s  = json.dumps(result, separators=(",", ":")) if isinstance(result, (dict, list)) else str(result)
        parts.append(f"  {tool}({args_s}) -> {res_s[:120]}")
    return "\n".join(parts)


def _build_prompt(task, trace):
    history      = _format_step_history(trace, len(trace))
    user_content = (
        f"Task: {task.get('user_request', '')}\n\n"
        f"Steps taken so far:\n{history}\n\n"
        "What is the next tool call?"
    )
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _parse_tool_call(generated):
    """
    Parse the model's generated text into (tool_name, args_dict).
    Returns None if parsing fails or tool is not recognized.
    """
    text = generated.strip()
    for stop in ["<|im_end|>", "<|endoftext|>", "</s>"]:
        if stop in text:
            text = text[:text.index(stop)].strip()

    # Try direct JSON parse first
    try:
        obj  = json.loads(text)
        tool = obj.get("tool", "")
        args = obj.get("args", {})
        if tool in VALID_TOOLS and isinstance(args, dict):
            return tool, args
    except json.JSONDecodeError:
        pass

    # Fallback: depth-counting bracket scan for embedded JSON
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            if depth == 0:
                candidate = text[start:end + 1]
                try:
                    obj  = json.loads(candidate)
                    tool = obj.get("tool", "")
                    args = obj.get("args", {})
                    if tool in VALID_TOOLS and isinstance(args, dict):
                        return tool, args
                except json.JSONDecodeError:
                    pass
                break

    return None


class LoRAAgent:
    """
    Inference agent using a LoRA-finetuned Qwen2.5-0.5B-Instruct.

    Falls back to PolicyAgent tool selection if the model output
    cannot be parsed, so benchmark scores stay interpretable.
    """

    name      = "lora_finetuned"
    MAX_STEPS = 16

    def __init__(
        self,
        adapter_path=None,
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens=80,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        if adapter_path is None:
            adapter_path = str(Path(__file__).parent.parent / "artifacts" / "lora_adapter")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16    = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        dtype       = torch.bfloat16 if use_bf16 else torch.float32

        print(f"[LoRAAgent] Loading adapter from {adapter_path} ...")
        print(f"[LoRAAgent] Device: {self.device}  dtype: {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

        print("[LoRAAgent] Model loaded.")

    def _generate(self, task, trace):
        prompt = _build_prompt(task, trace)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens = self.max_new_tokens,
                do_sample      = False,
                temperature    = 1.0,
                pad_token_id   = self.tokenizer.pad_token_id,
                eos_token_id   = self.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=False)

    def _fallback_tool(self, task, env):
        """PolicyAgent fallback: used when model output cannot be parsed."""
        from src.agents import PolicyAgent
        import pickle

        pkl = Path(__file__).parent.parent / "artifacts" / "next_action_policy.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                obj = pickle.load(f)
            pa = PolicyAgent(model=obj["clf"], vectorizer=obj["vectorizer"])
        else:
            REQUIRED = ["get_case", "lookup_customer", "search_invoices",
                        "search_payments", "final_answer"]
            called = [s["tool"] for s in env.trace]
            for req in REQUIRED:
                if req not in called:
                    pa = PolicyAgent.__new__(PolicyAgent)
                    pa.model = pa.vectorizer = None
                    return req, {}
            return "final_answer", {}

        tool = pa._select_next_tool(task, env)
        args = pa._build_args(tool, task, env)
        return tool, args

    def _enforce_required_reads(self, parsed, task, env):
        """
        Guard: prevents final_answer before the four required reads complete.
        Redirects to the next missing required tool to prevent mode collapse.
        """
        REQUIRED = ["get_case", "lookup_customer", "search_invoices", "search_payments"]
        called   = [s["tool"] for s in env.trace]

        if parsed is not None and parsed[0] == "final_answer":
            for req in REQUIRED:
                if req not in called:
                    return self._fallback_tool(task, env)

        return parsed if parsed is not None else self._fallback_tool(task, env)

    def run(self, task, env):
        env.reset()
        fallback_count = 0

        for _ in range(self.MAX_STEPS):
            generated = self._generate(task, env.trace)
            parsed    = _parse_tool_call(generated)

            tool, args = self._enforce_required_reads(parsed, task, env)
            if parsed is None or (parsed[0] == "final_answer" and tool != "final_answer"):
                fallback_count += 1

            method = getattr(env, tool, None)
            if method is None:
                tool, args = self._fallback_tool(task, env)
                method = getattr(env, tool)

            try:
                method(**args)
            except TypeError:
                tool, args = self._fallback_tool(task, env)
                getattr(env, tool)(**args)

            if tool == "final_answer":
                break

        if fallback_count:
            print(f"[LoRAAgent] task={task.get('case_id','?')} used {fallback_count} fallback(s)")

        return env.trace
