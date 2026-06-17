"""
DPO (Direct Preference Optimization) training step for V2.

Runs after LoRA SFT. Generates preference pairs from the training traces:
  - chosen  = correct tool call JSON from the trace
  - rejected = model output sampled at temperature=1.0, or a deterministic
               corruption if the model happened to get it right

Uses trl's DPOTrainer. Requires trl>=0.8.0.

Saves a DPO-refined adapter to artifacts/v2/{slug}_dpo/ and pushes to HF.

Usage:
    from src.v2.dpo_train import dpo_train
    dpo_url = dpo_train(adapter_path, base_model_name, traces)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT      = Path(__file__).parent.parent.parent
ARTIFACTS = ROOT / "artifacts" / "v2"

SYSTEM_PROMPT = (
    "You are a finance operations agent for Acme, Inc.\n"
    "Given a task and the steps taken so far, output the next tool call as JSON:\n"
    '{"tool": "<tool_name>", "args": {<key>: <value>, ...}}\n'
    "Output ONLY the JSON, nothing else."
)

# Deterministic fallback corruption: swap to a plausible wrong tool
WRONG_TOOL = {
    "get_case":            "final_answer",
    "lookup_customer":     "final_answer",
    "search_invoices":     "final_answer",
    "search_payments":     "final_answer",
    "search_credit_memos": "search_payments",
    "create_exception":    "update_case",
    "update_case":         "create_exception",
    "draft_slack_message": "final_answer",
    "final_answer":        "search_invoices",
}


def _format_step_history(messages, up_to):
    if up_to == 0:
        return "No steps taken yet."
    parts = []
    for msg in messages[:up_to]:
        tc    = msg.get("tool_call", {})
        name  = tc.get("name", "?")
        args  = json.dumps(tc.get("arguments", {}), separators=(",", ":"))
        res   = msg.get("tool_result", "")
        res_s = json.dumps(res, separators=(",", ":")) if isinstance(res, (dict, list)) else str(res)
        parts.append(f"  {name}({args}) -> {res_s[:120]}")
    return "\n".join(parts)


def _make_chat_prompt(task_input, messages, step_idx, tokenizer):
    history = _format_step_history(messages, step_idx)
    user_content = (
        f"Task: {task_input}\n\n"
        f"Steps taken so far:\n{history}\n\n"
        "What is the next tool call?"
    )
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def _sample_rejected(prompt, correct_json, model, tokenizer, device):
    """
    Sample the model at temperature=1.0. If different from correct, use it.
    Otherwise fall back to deterministic corruption.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens = 80,
            do_sample      = True,
            temperature    = 1.0,
            pad_token_id   = tokenizer.pad_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    sampled    = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

    # Strip stop tokens
    for stop in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
        if stop in sampled:
            sampled = sampled[:sampled.index(stop)].strip()

    if sampled and sampled != correct_json.strip():
        return sampled

    # Fallback: corrupt by swapping to a different tool
    try:
        obj  = json.loads(correct_json)
        tool = obj.get("tool", "final_answer")
        wrong = WRONG_TOOL.get(tool, "final_answer")
        return json.dumps({"tool": wrong, "args": {}}, separators=(",", ":"))
    except Exception:
        return json.dumps({"tool": "final_answer", "args": {"answer": "skip", "evidence_ids": []}})


def build_dpo_dataset(adapter_path, base_model_name, traces, n_samples=1):
    """
    Build a DPO preference dataset from training traces.

    For each (prompt, chosen_completion) pair, samples the model to get
    a rejected completion. Returns a HuggingFace Dataset with columns:
    prompt, chosen, rejected.
    """
    from datasets import Dataset as HFDataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if use_bf16 else torch.float32

    print(f"[dpo] Loading model for preference sampling from {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base  = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.to(device)
    model.eval()

    prompts_list  = []
    chosen_list   = []
    rejected_list = []

    for trace in traces:
        task_input = trace.get("input", "")
        messages   = trace.get("messages", [])
        for i, msg in enumerate(messages):
            tc      = msg.get("tool_call", {})
            correct = json.dumps({"tool": tc.get("name", ""), "args": tc.get("arguments", {})}, separators=(",", ":"))
            prompt  = _make_chat_prompt(task_input, messages, i, tokenizer)

            for _ in range(n_samples):
                rejected = _sample_rejected(prompt, correct, model, tokenizer, device)
                prompts_list.append(prompt)
                chosen_list.append(correct + tokenizer.eos_token)
                rejected_list.append(rejected + tokenizer.eos_token)

    print(f"[dpo] Built {len(prompts_list)} preference pairs from {len(traces)} traces")
    return HFDataset.from_dict({
        "prompt":   prompts_list,
        "chosen":   chosen_list,
        "rejected": rejected_list,
    })


def dpo_train(adapter_path, base_model_name, traces, verbose=True):
    """
    Run a DPO training pass on top of the existing LoRA adapter.
    Saves refined adapter to artifacts/v2/{slug}_dpo/.
    Returns (dpo_adapter_path, hf_url).
    """
    try:
        from trl import DPOTrainer, DPOConfig
    except ImportError:
        print("[dpo] trl not installed. Run: pip install trl>=0.8.0 --break-system-packages")
        return None, None

    slug     = Path(adapter_path).name
    dpo_path = ARTIFACTS / f"{slug}_dpo"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16
    dtype    = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base  = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.to(device)

    dataset = build_dpo_dataset(adapter_path, base_model_name, traces)

    dpo_config = DPOConfig(
        output_dir                  = str(ARTIFACTS / f"{slug}_dpo_ckpt"),
        num_train_epochs            = 2,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        learning_rate               = 5e-5,
        beta                        = 0.1,
        fp16                        = use_fp16,
        bf16                        = use_bf16,
        logging_steps               = 5,
        save_strategy               = "no",
        report_to                   = "none",
        remove_unused_columns       = False,
        max_length                  = 512,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
    )

    trainer = DPOTrainer(
        model     = model,
        args      = dpo_config,
        train_dataset = dataset,
        processing_class = tokenizer,
    )

    if verbose:
        print(f"[dpo] Starting DPO training for {slug} ({len(dataset)} preference pairs) ...")
    trainer.train()

    model.save_pretrained(str(dpo_path))
    tokenizer.save_pretrained(str(dpo_path))
    if verbose:
        print(f"[dpo] DPO adapter saved -> {dpo_path}")

    from src.v2.multi_lora import push_to_hub
    hf_url = push_to_hub(dpo_path, f"{slug}-dpo")
    return dpo_path, hf_url
