"""
LoRA fine-tuning pipeline for the Acme RL benchmark.

Loads Qwen/Qwen2.5-0.5B-Instruct from HuggingFace, applies LoRA,
and fine-tunes it on (context → next_tool_call) step pairs extracted
from training_traces.jsonl.

After training the adapter is saved to artifacts/lora_adapter/.
A LoRAAgent wrapping the fine-tuned model is returned.

Usage:
    from src.lora_train import lora_train
    agent = lora_train()

    # or from CLI:
    python -m src.lora_train
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

from src.lora_agent import LoRAAgent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).parent.parent
FIXTURES      = ROOT / "fixtures" / "rl-finetuning"
ARTIFACTS     = ROOT / "artifacts"
ADAPTER_PATH  = ARTIFACTS / "lora_adapter"
TRACES_PATH   = FIXTURES / "training_traces.jsonl"

BASE_MODEL    = "Qwen/Qwen2.5-0.5B-Instruct"

# ---------------------------------------------------------------------------
# ChatML prompt template (Qwen2.5 uses ChatML natively)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a finance operations agent for Acme, Inc.\n"
    "Given a task and the steps taken so far, output the next tool call as JSON:\n"
    '{"tool": "<tool_name>", "args": {<key>: <value>, ...}}\n'
    "Output ONLY the JSON, nothing else."
)


def _format_step_history(messages: list[dict], up_to: int) -> str:
    """Build a text summary of the tool calls made before step `up_to`."""
    if up_to == 0:
        return "No steps taken yet."
    parts = []
    for msg in messages[:up_to]:
        tc   = msg.get("tool_call", {})
        name = tc.get("name", "?")
        args = json.dumps(tc.get("arguments", {}), separators=(",", ":"))
        res  = msg.get("tool_result", "")
        res_str = json.dumps(res, separators=(",", ":")) if isinstance(res, (dict, list)) else str(res)
        # Keep result compact — just the first 120 chars
        parts.append(f"  {name}({args}) → {res_str[:120]}")
    return "\n".join(parts)


def _make_prompt(task_input: str, messages: list[dict], step_idx: int) -> str:
    """Full ChatML prompt for predicting the tool at `step_idx`."""
    history = _format_step_history(messages, step_idx)
    user_content = (
        f"Task: {task_input}\n\n"
        f"Steps taken so far:\n{history}\n\n"
        "What is the next tool call?"
    )
    # Qwen2.5 ChatML format
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _make_completion(msg: dict) -> str:
    """The target completion: compact JSON of the tool call at this step."""
    tc   = msg.get("tool_call", {})
    name = tc.get("name", "")
    args = tc.get("arguments", {})
    return json.dumps({"tool": name, "args": args}, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset(tokenizer, max_length: int = 512) -> Dataset:
    """
    Extract (prompt, completion) pairs from training_traces.jsonl,
    tokenize them, and return a HuggingFace Dataset.

    Each (step_i → step_i+1) pair is one training example.
    The loss is computed only on the completion tokens (prompt tokens are masked).
    """
    prompts     = []
    completions = []

    with open(TRACES_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trace      = json.loads(line)
            task_input = trace.get("input", "")
            messages   = trace.get("messages", [])

            for i, msg in enumerate(messages):
                prompts.append(_make_prompt(task_input, messages, i))
                completions.append(_make_completion(msg))

    print(f"[lora_train] Built {len(prompts)} training examples from {TRACES_PATH.name}")

    # Tokenize: full_text = prompt + completion + eos
    input_ids_list  = []
    labels_list     = []
    attention_list  = []

    for prompt, completion in zip(prompts, completions):
        full_text = prompt + completion + tokenizer.eos_token

        prompt_enc      = tokenizer(prompt,     add_special_tokens=False)
        full_enc        = tokenizer(full_text,  add_special_tokens=False,
                                    max_length=max_length, truncation=True)

        input_ids       = full_enc["input_ids"]
        attention_mask  = full_enc["attention_mask"]
        prompt_len      = len(prompt_enc["input_ids"])

        # Mask the prompt in labels so loss is only on completion tokens
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # Pad to max_length
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids      = input_ids      + [tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels         = labels         + [-100] * pad_len
        else:
            labels         = labels[:max_length]

        input_ids_list.append(input_ids[:max_length])
        attention_list.append(attention_mask[:max_length])
        labels_list.append(labels[:max_length])

    return Dataset.from_dict({
        "input_ids":      input_ids_list,
        "attention_mask": attention_list,
        "labels":         labels_list,
    })


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def lora_train(
    base_model:   str  = BASE_MODEL,
    lora_rank:    int  = 8,
    lora_alpha:   int  = 16,
    epochs:       int  = 5,
    lr:           float = 2e-4,
    verbose:      bool  = True,
) -> LoRAAgent:
    """
    Fine-tune `base_model` with LoRA on the Acme training traces.

    Returns a LoRAAgent wrapping the fine-tuned adapter.
    Adapter is also saved to artifacts/lora_adapter/.

    Args:
        base_model:  HuggingFace model ID (default: Qwen/Qwen2.5-0.5B-Instruct)
        lora_rank:   LoRA rank r (default: 8, controls adapter capacity)
        lora_alpha:  LoRA scaling factor (default: 16)
        epochs:      Training epochs (default: 5)
        lr:          Learning rate (default: 2e-4)
        verbose:     Print progress (default: True)

    Returns:
        LoRAAgent — a benchmark-compatible agent using the fine-tuned model.
    """
    # Auto-detect GPU and choose optimal dtype
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16
    dtype    = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    if verbose:
        print(f"\n[lora_train] Loading base model: {base_model}")
        print(f"[lora_train] LoRA config: rank={lora_rank}, alpha={lora_alpha}, epochs={epochs}")
        print(f"[lora_train] Device: {'GPU (cuda)' if use_cuda else 'CPU'}  dtype: {dtype}")

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_cfg = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = lora_rank,
        lora_alpha     = lora_alpha,
        target_modules = ["q_proj", "v_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
    )
    model = get_peft_model(model, lora_cfg)
    if verbose:
        model.print_trainable_parameters()

    # Build dataset
    dataset = build_dataset(tokenizer)

    # Training arguments — auto-configured for GPU or CPU
    ARTIFACTS.mkdir(exist_ok=True)
    training_args = TrainingArguments(
        output_dir                  = str(ARTIFACTS / "lora_checkpoints"),
        num_train_epochs            = epochs,
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
            tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
        ),
    )

    if verbose:
        print(f"\n[lora_train] Starting LoRA training ({epochs} epochs, {len(dataset)} examples) ...")

    trainer.train()

    # Save adapter
    model.save_pretrained(str(ADAPTER_PATH))
    tokenizer.save_pretrained(str(ADAPTER_PATH))
    if verbose:
        print(f"[lora_train] Adapter saved →