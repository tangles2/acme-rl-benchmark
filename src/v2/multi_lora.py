"""
Multi-model LoRA training for V2.

Trains one LoRA adapter per model config on the combined (original +
synthetic) trace dataset. Uses tokenizer.apply_chat_template() so the
same training code works across Qwen, SmolLM2, Phi, and others without
any format-specific hardcoding.

Adapters are saved to artifacts/v2/{slug}/ and auto-pushed to HuggingFace
if the user is logged in.

Usage:
    from src.v2.multi_lora import train_all_models, MODEL_CONFIGS
    agents = train_all_models(traces)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

ROOT      = Path(__file__).parent.parent.parent
ARTIFACTS = ROOT / "artifacts" / "v2"

# ---------------------------------------------------------------------------
# Model configs -- add more here to expand the comparison
# ---------------------------------------------------------------------------

MODEL_CONFIGS = [
    {
        "name":       "Qwen/Qwen2.5-0.5B-Instruct",
        "slug":       "qwen2.5-0.5b",
        "lora_rank":  8,
        "lora_alpha": 16,
        "epochs":     5,
        "lr":         2e-4,
    },
    {
        "name":       "Qwen/Qwen2.5-1.5B-Instruct",
        "slug":       "qwen2.5-1.5b",
        "lora_rank":  8,
        "lora_alpha": 16,
        "epochs":     5,
        "lr":         2e-4,
    },
]

SYSTEM_PROMPT = (
    "You are a finance operations agent for Acme, Inc.\n"
    "Given a task and the steps taken so far, output the next tool call as JSON:\n"
    '{"tool": "<tool_name>", "args": {<key>: <value>, ...}}\n'
    "Output ONLY the JSON, nothing else."
)

# ---------------------------------------------------------------------------
# Prompt/completion builders
# ---------------------------------------------------------------------------

def _format_step_history(messages, up_to):
    if up_to == 0:
        return "No steps taken yet."
    parts = []
    for msg in messages[:up_to]:
        tc   = msg.get("tool_call", {})
        name = tc.get("name", "?")
        args = json.dumps(tc.get("arguments", {}), separators=(",", ":"))
        res  = msg.get("tool_result", "")
        res_s = json.dumps(res, separators=(",", ":")) if isinstance(res, (dict, list)) else str(res)
        parts.append(f"  {name}({args}) -> {res_s[:120]}")
    return "\n".join(parts)


def _make_prompt(task_input, messages, step_idx, tokenizer):
    """Build a chat-template prompt for predicting the tool at step_idx."""
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


def _make_completion(msg):
    tc   = msg.get("tool_call", {})
    name = tc.get("name", "")
    args = tc.get("arguments", {})
    return json.dumps({"tool": name, "args": args}, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(traces, tokenizer, max_length=512):
    prompts     = []
    completions = []

    for trace in traces:
        task_input = trace.get("input", "")
        messages   = trace.get("messages", [])
        for i, msg in enumerate(messages):
            prompts.append(_make_prompt(task_input, messages, i, tokenizer))
            completions.append(_make_completion(msg))

    print(f"[multi_lora] Built {len(prompts)} training examples from {len(traces)} traces")

    input_ids_list = []
    labels_list    = []
    attention_list = []

    for prompt, completion in zip(prompts, completions):
        full_text = prompt + completion + tokenizer.eos_token

        prompt_enc = tokenizer(prompt,    add_special_tokens=False)
        full_enc   = tokenizer(full_text, add_special_tokens=False,
                               max_length=max_length, truncation=True)

        input_ids      = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        prompt_len     = len(prompt_enc["input_ids"])
        labels         = [-100] * prompt_len + input_ids[prompt_len:]

        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids      = input_ids      + [tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels         = labels         + [-100] * pad_len
        else:
            labels = labels[:max_length]

        input_ids_list.append(input_ids[:max_length])
        attention_list.append(attention_mask[:max_length])
        labels_list.append(labels[:max_length])

    return Dataset.from_dict({
        "input_ids":      input_ids_list,
        "attention_mask": attention_list,
        "labels":         labels_list,
    })


# ---------------------------------------------------------------------------
# HuggingFace push
# ---------------------------------------------------------------------------

def push_to_hub(adapter_path, slug):
    try:
        from huggingface_hub import HfApi, whoami
        api      = HfApi()
        user     = whoami()
        username = user["name"]
        repo_id  = f"{username}/acme-lora-v2-{slug}"
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path = str(adapter_path),
            repo_id     = repo_id,
            repo_type   = "model",
            commit_message = f"Upload V2 LoRA adapter: {slug}",
        )
        url = f"https://huggingface.co/{repo_id}"
        print(f"[push] {slug} -> {url}")
        return url
    except Exception as e:
        print(f"[push] Skipped for {slug}: {e}")
        return None


# ---------------------------------------------------------------------------
# Single model training
# ---------------------------------------------------------------------------

def train_single_model(config, traces, verbose=True):
    """
    Fine-tune one model with LoRA. Returns (adapter_path, hf_url).
    """
    name  = config["name"]
    slug  = config["slug"]
    rank  = config["lora_rank"]
    alpha = config["lora_alpha"]
    lr    = config["lr"]
    ep    = config["epochs"]

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16
    dtype    = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    if verbose:
        print(f"\n[multi_lora] Training: {name}")
        print(f"[multi_lora] rank={rank}, alpha={alpha}, epochs={ep}, dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, trust_remote_code=True)

    lora_cfg = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = rank,
        lora_alpha     = alpha,
        target_modules = ["q_proj", "v_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
    )
    model = get_peft_model(model, lora_cfg)
    if verbose:
        model.print_trainable_parameters()

    dataset = build_dataset(traces, tokenizer)

    adapter_path = ARTIFACTS / slug
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = str(ARTIFACTS / f"{slug}_ckpt"),
        num_train_epochs            = ep,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        learning_rate               = lr,
        lr_scheduler_type           = "cosine",
        warmup_steps                = 5,
        save_strategy               = "no",
        logging_steps               = 5,
        fp16                        = use_fp16,
        bf16                        = use_bf16,
        dataloader_num_workers      = 0,
        report_to                   = "none",
        remove_unused_columns       = False,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = dataset,
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, padding=True, pad_to_multiple_of=8
        ),
    )

    if verbose:
        print(f"[multi_lora] Starting training ({ep} epochs, {len(dataset)} examples) ...")

    trainer.train()

    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    if verbose:
        print(f"[multi_lora] Adapter saved -> {adapter_path}")

    hf_url = push_to_hub(adapter_path, slug)
    return adapter_path, hf_url


def train_all_models(traces, configs=None, verbose=True):
    """
    Train all models in configs list. Returns list of (config, adapter_path, hf_url).
    """
    if configs is None:
        configs = MODEL_CONFIGS
    results = []
    for config in configs:
        adapter_path, hf_url = train_single_model(config, traces, verbose=verbose)
        results.append((config, adapter_path, hf_url))
    return results


# ---------------------------------------------------------------------------
# V2LoRAAgent -- model-agnostic inference using apply_chat_template
# ---------------------------------------------------------------------------

VALID_TOOLS = {
    "get_case", "lookup_customer", "search_invoices", "search_payments",
    "search_credit_memos", "update_case", "create_exception",
    "draft_slack_message", "final_answer",
}


def _parse_tool_call(text):
    import json as _json
    text = text.strip()
    for stop in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
        if stop in text:
            text = text[:text.index(stop)].strip()

    try:
        obj  = _json.loads(text)
        tool = obj.get("tool", "")
        args = obj.get("args", {})
        if tool in VALID_TOOLS and isinstance(args, dict):
            return tool, args
    except _json.JSONDecodeError:
        pass

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
                try:
                    obj  = _json.loads(text[start:end + 1])
                    tool = obj.get("tool", "")
                    args = obj.get("args", {})
                    if tool in VALID_TOOLS and isinstance(args, dict):
                        return tool, args
                except _json.JSONDecodeError:
                    pass
                break
    return None


class V2LoRAAgent:
    """
    Inference agent wrapping a V2 LoRA adapter.
    Uses apply_chat_template so it works with any model.
    """

    MAX_STEPS = 16

    def __init__(self, adapter_path, base_model, max_new_tokens=80):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.slug = Path(adapter_path).name
        self.name = f"lora_v2_{self.slug}"

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16     = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        dtype        = torch.bfloat16 if use_bf16 else torch.float32

        print(f"[V2LoRAAgent] Loading {self.slug} from {adapter_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, trust_remote_code=True)
        self.model = PeftModel.from_pretrained(base, str(adapter_path))
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        print(f"[V2LoRAAgent] {self.slug} ready on {self.device}")

    def _build_prompt(self, task, trace):
        history = []
        for step in trace:
            tool  = step.get("tool", "?")
            args  = json.dumps(step.get("args", {}), separators=(",", ":"))
            res   = step.get("result", "")
            res_s = json.dumps(res, separators=(",", ":")) if isinstance(res, (dict, list)) else str(res)
            history.append(f"  {tool}({args}) -> {res_s[:120]}")
        history_str = "\n".join(history) if history else "No steps taken yet."
        user_content = (
            f"Task: {task.get('user_request', '')}\n\n"
            f"Steps taken so far:\n{history_str}\n\n"
            "What is the next tool call?"
        )
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def _generate(self, task, trace):
        prompt = self._build_prompt(task, trace)
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
        return self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=False)

    def _fallback(self, task, env):
        from src.agents import PolicyAgent
        import pickle
        pkl = ROOT / "artifacts" / "next_action_policy.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                obj = pickle.load(f)
            pa   = PolicyAgent(model=obj["clf"], vectorizer=obj["vectorizer"])
            tool = pa._select_next_tool(task, env)
            args = pa._build_args(tool, task, env)
            return tool, args
        REQUIRED = ["get_case", "lookup_customer", "search_invoices", "search_payments"]
        called = [s["tool"] for s in env.trace]
        for req in REQUIRED:
            if req not in called:
                return req, {"case_id": task["case_id"]} if req == "get_case" else {}
        return "final_answer", {"answer": "unknown", "evidence_ids": []}

    def _enforce_required_reads(self, parsed, task, env):
        REQUIRED = ["get_case", "lookup_customer", "search_invoices", "search_payments"]
        called   = [s["tool"] for s in env.trace]
        if parsed is not None and parsed[0] == "final_answer":
            for req in REQUIRED:
                if req not in called:
                    return self._fallback(task, env)
        return parsed if parsed is not None else self._fallback(task, env)

    def run(self, task, env):
        env.reset()
        fallback_count = 0
        for _ in range(self.MAX_STEPS):
            generated  = self._generate(task, env.trace)
            parsed     = _parse_tool_call(generated)
            tool, args = self._enforce_required_reads(parsed, task, env)
            if parsed is None or (parsed[0] == "final_answer" and tool != "final_answer"):
                fallback_count += 1
            method = getattr(env, tool, None)
            if method is None:
                tool, args = self._fallback(task, env)
                method = getattr(env, tool)
            try:
                method(**args)
            except TypeError:
                tool, args = self._fallback(task, env)
                getattr(env, tool)(**args)
            if tool == "final_answer":
                break
        if fallback_count:
            print(f"[V2LoRAAgent:{self.slug}] task={task.get('case_id','?')} used {fallback_count} fallback(s)")
        return env.trace
