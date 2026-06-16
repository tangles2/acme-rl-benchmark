"""
Creates and pushes a model card to the HuggingFace repo.
Run from anywhere after huggingface-cli login:
    python3 push_model_card.py
"""

from huggingface_hub import HfApi

REPO_ID = "tangles2/acme-lora-qwen2.5-0.5b"

MODEL_CARD = """---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- lora
- peft
- finance
- tool-calling
- rl-finetuning
- qwen2.5
datasets:
- custom (acme-rl-benchmark training traces)
---

# acme-lora-qwen2.5-0.5b

LoRA adapter fine-tuned on top of [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) for a finance operations tool-calling workflow.

Built as part of the [acme-rl-benchmark](https://github.com/tangles2/acme-rl-benchmark) take-home project for Manifold Labs.

## What it does

The adapter trains the base model to call a sequence of finance tools in the correct order to resolve collections cases:

- `get_case` / `lookup_customer` / `search_invoices` / `search_payments` / `search_credit_memos`
- `update_case` / `create_exception` / `draft_slack_message` / `final_answer`

Given a task description and the steps taken so far, the model outputs the next tool call as JSON:

```json
{"tool": "search_invoices", "args": {"customer_id": "cus_globex", "invoice_id": "INV-2026-0413"}}
```

## Training details

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Target modules | q_proj, v_proj |
| Trainable params | 540,672 / 494,573,440 (0.11%) |
| Training examples | 28 (4 traces x ~7 steps each) |
| Epochs | 5 |
| Learning rate | 2e-4 (cosine schedule) |
| Hardware | Targon H200, bfloat16 |
| Training time | ~22 seconds |

## Benchmark results

Evaluated on 5 finance operations tasks across 8 scoring criteria.

| Agent | Strict Pass Rate | Avg Score | Broad Scans |
|---|---|---|---|
| Rule-based baseline | 0/5 (0%) | 86.3% | 2.0 |
| sklearn PolicyAgent | 4/5 (80%) | 90.0% | 0.0 |
| **This model (LoRA)** | **3/5 (60%)** | **82.5%** | **0.0** |

The model relies heavily on a PolicyAgent fallback (4-7 fallbacks per task) due to
limited training data. With 100+ training examples the fallback rate would drop
significantly.

## Limitations

- Only 28 training examples. The model does not reliably produce valid JSON tool calls on its own yet.
- `task_credit_memo_reconciled` fails across all agents due to only 1 training example of `search_credit_memos`.
- `task_missing_evidence` fails due to the model running past the correct stopping point (hits 16-step limit).

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = "Qwen/Qwen2.5-0.5B-Instruct"
adapter    = "tangles2/acme-lora-qwen2.5-0.5b"

tokenizer = AutoTokenizer.from_pretrained(adapter)
model     = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
model     = PeftModel.from_pretrained(model, adapter)
model.eval()
```

See the [benchmark repo](https://github.com/tangles2/acme-rl-benchmark) for the full
inference loop, mock environment, and scoring harness.
"""

api = HfApi()
api.upload_file(
    path_or_fileobj=MODEL_CARD.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Add model card",
)
print(f"Model card pushed to https://huggingface.co/{REPO_ID}")
