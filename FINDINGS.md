# Run 1 Findings — Acme RL Finetuning Benchmark

**Date:** 2026-06-16  
**Hardware:** Targon GPU node (CPU fallback — CUDA driver mismatch)  
**Model:** Qwen/Qwen2.5-0.5B-Instruct + LoRA (rank=8, alpha=16, 5 epochs)  
**Training data:** 28 examples from 4 traces in `training_traces.jsonl`

---

## Results Summary

| Agent | Strict Pass Rate | Avg Score | Avg Tool Calls | Broad Scans |
|---|---|---|---|---|
| Rule-based baseline | 0/5 (0%) | 86.3% | 6.6 | 2.0 |
| sklearn PolicyAgent | 4/5 (80%) | 90.0% | 6.6 | 0.0 |
| LoRA (Qwen2.5-0.5B) | 0/5 (0%) | 57.5% | 0.0 | 0.0 |

### Criteria accuracy by agent

| Criterion | Baseline | sklearn | LoRA |
|---|---|---|---|
| status | 80% | 80% | 0% |
| resolution | 80% | 80% | 0% |
| amounts | 80% | 80% | 60% |
| evidence | 60% | 80% | 0% |
| observed_evidence | 100% | 100% | 100% |
| forbidden_evidence | 100% | 100% | 100% |
| unsafe_mutation | 100% | 100% | 100% |
| tool_efficiency | 0% | 100% | 100% |

---

## Finding 1 — LoRA model collapsed to calling `final_answer` immediately

The most significant result of Run 1 is that the fine-tuned LoRA agent called `final_answer`
as its very first action on every task, resulting in zero actual tool calls (`calls=0`).

**Why this happened:** With only 28 training examples spread across 9 tool classes,
the model did not learn the correct multi-step workflow. Instead it converged on
`final_answer` as a safe default — it never causes broad scan penalties, unsafe
mutation flags, or forbidden evidence violations. The model found the path of least
resistance rather than the correct task sequence.

This is a textbook case of **mode collapse / reward hacking**: the agent optimized
for avoiding penalties rather than completing the task.

**Evidence in the training logs:**

```
{'loss': '1.509', 'epoch': '0.71'}
{'loss': '1.232', 'epoch': '1.43'}
{'loss': '1.034', 'epoch': '2.14'}
{'loss': '0.862', 'epoch': '2.86'}
{'loss': '0.814', 'epoch': '3.57'}
{'loss': '0.771', 'epoch': '4.29'}
{'loss': '0.895', 'epoch': '5.00'}  ← loss rebounded at end
train_loss: 1.017
```

Loss decreased but rebounded at epoch 5, suggesting the model was oscillating
rather than converging — a sign of too few examples relative to model capacity.

---

## Finding 2 — sklearn outperformed LoRA despite being far simpler

The TF-IDF + LogisticRegression classifier (trained in under 1 second, CPU-only,
no GPU required) scored 4/5 vs. the LoRA model's 0/5.

This highlights an important principle: **more data trumps model size**.
With only 28 examples, a simple linear classifier generalizes better than
a 0.5B parameter generative model that can overfit or collapse.

The sklearn agent succeeds because it was designed as a hybrid — the classifier
only handles two genuinely uncertain decisions (check credit memos? send Slack?),
while deterministic rules enforce the required read sequence and safe mutation ordering.
A generative model trained on 28 examples has to learn all of this from scratch.

---

## Finding 3 — The one consistent remaining failure: `task_credit_memo_reconciled`

Both the sklearn agent and the LoRA agent fail this task. The root cause is
data imbalance: `search_credit_memos` appears only once in the 28 training
examples (1 out of 4 traces uses it), so neither model reliably learns when
to call it.

This is the canonical long-tail problem in SFT: rare behaviors require
disproportionately more training examples than common ones.

---

## Finding 4 — CUDA driver mismatch forced CPU inference

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
(found version 12080).
```

Training and inference both ran on CPU. This caused LoRA training to take
~10 minutes (625 seconds) for 35 steps — on a GPU with a matching driver
this would run in under 60 seconds. The CUDA warning did not affect correctness,
only speed.

---

## Proposed Improvements for Run 2

### 1. Add a minimum-steps guard before `final_answer` (quick fix)

The fastest fix: in `lora_agent.py`, prevent the model from calling `final_answer`
until the four required reads have been completed. This mirrors the Phase 1 logic
in PolicyAgent and eliminates mode collapse without retraining.

```python
REQUIRED_BEFORE_FINAL = ["get_case", "lookup_customer", "search_invoices", "search_payments"]

def _select_tool(self, generated, task, env):
    parsed = _parse_tool_call(generated)
    called = [s["tool"] for s in env.trace]
    # Block final_answer until required reads are done
    if parsed and parsed[0] == "final_answer":
        for req in REQUIRED_BEFORE_FINAL:
            if req not in called:
                return self._fallback_tool(task, env)
    return parsed
```

### 2. Generate more training traces (biggest impact)

28 examples is too few for a generative model. Options:
- **Synthetic traces via SybilAgent**: run the Sybil API against more task variants
  and collect successful traces as additional training data (self-play / data flywheel)
- **Manual authoring**: write 2-3 additional traces for `search_credit_memos` and
  edge cases to fix the class imbalance
- **Target**: 100-200 examples across all 9 tool classes, with at least 5-10
  examples per rare class

### 3. Increase training epochs and add early stopping

With more data, increase epochs to 15-20 and add an eval split to detect
overfitting early. The loss rebound at epoch 5 suggests the current setup
is at the edge of instability.

### 4. Add tool schema to the system prompt

The current system prompt describes the workflow in plain English.
Explicitly including the JSON schema for each tool gives the model a
structured reference, reducing hallucinated argument names.

### 5. Constrain generation with structured decoding (longer term)

Force the model output to be valid JSON matching the tool schema using
constrained beam search or grammar-based sampling (e.g., `outlines` library).
This eliminates the need for the fallback parser entirely.

### 6. Use a larger base model

Qwen2.5-0.5B is at the small end. Qwen2.5-1.5B or 3B would have significantly
better instruction-following out of the box and require fewer examples to
fine-tune reliably. The LoRA adapter stays small regardless of base model size.

---

## What to discuss with Manifold

- The mode collapse result is a real and interesting failure — lead with it, not away from it
- Explain the sklearn vs LoRA comparison and why simple models can win with limited data
- Walk through the reward hacking angle: the model found a zero-penalty strategy
  (call final_answer immediately) that scores well on safety criteria but fails on correctness
- Propose the data flywheel: use the Sybil API to generate synthetic traces at scale,
  then retrain — this is the realistic path to a production-quality agent
