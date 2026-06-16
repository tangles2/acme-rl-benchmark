# Findings — Acme RL Finetuning Benchmark

**Repository:** https://github.com/tangles2/acme-rl-benchmark  
**Hardware:** Targon H200 GPU node (CUDA 12.8, driver 570.195.03)  
**Models tested:** Rule-based baseline, TF-IDF + LogisticRegression (sklearn), Qwen/Qwen2.5-0.5B-Instruct + LoRA (rank=8, alpha=16)  
**Training data:** 28 examples from 4 traces in `training_traces.jsonl`

---

## Results Summary

| Agent | Strict Pass Rate | Avg Score | Avg Tool Calls | Avg Broad Scans |
|---|---|---|---|---|
| Rule-based baseline | 0/5 (0%) | 86.3% | 6.6 | 2.0 |
| sklearn PolicyAgent (post-SFT) | **4/5 (80%)** | **90.0%** | 6.6 | **0.0** |
| LoRA Qwen2.5-0.5B (Run 1 — CPU) | 0/5 (0%) | 57.5% | 0.0 | 0.0 |

### Criteria accuracy by agent

| Criterion | Baseline | sklearn | LoRA (Run 1) |
|---|---|---|---|
| status | 80% | 80% | 0% |
| resolution | 80% | 80% | 0% |
| amounts | 80% | 80% | 60% |
| evidence | 60% | **80%** | 0% |
| observed_evidence | 100% | 100% | 100% |
| forbidden_evidence | 100% | 100% | 100% |
| unsafe_mutation | 100% | 100% | 100% |
| tool_efficiency | 0% | **100%** | 100% |

---

## Finding 1 — sklearn SFT drove the largest gains

Training a TF-IDF + LogisticRegression next-action classifier on 28 trace examples
produced two concrete improvements over the rule-based baseline:

**Tool efficiency: 0% → 100%.** The baseline always called `search_invoices` and
`search_payments` without filters (2 broad scans per task). The trained classifier
learned to always include `invoice_id` and `month` arguments — eliminating all broad
scans across all five tasks.

**Evidence accuracy: 60% → 80%.** The baseline omitted `customer_id` from the
final evidence list on `task_ambiguous_customer`. The classifier, trained on traces
that cite `customer_id` as evidence, corrected this.

The before/after delta confirms the benchmark is sensitive to real behavioral changes —
not just output text.

---

## Finding 2 — LoRA model collapsed to calling `final_answer` immediately (Run 1)

In Run 1, the LoRA agent called `final_answer` as its very first action on every task,
resulting in zero tool calls and 0/5 pass rate.

**Why this happened:** With only 28 training examples across 9 tool classes, the model
did not learn the multi-step workflow. It converged on `final_answer` as a low-penalty
default — it never triggers broad scan penalties, unsafe mutation flags, or forbidden
evidence violations. This is a textbook case of **reward hacking / mode collapse**:
the model optimized for avoiding penalties rather than completing the task.

**Training loss from Run 1:**

```
epoch 0.71 → loss 1.509
epoch 1.43 → loss 1.232
epoch 2.14 → loss 1.034
epoch 2.86 → loss 0.862
epoch 3.57 → loss 0.814
epoch 4.29 → loss 0.771
epoch 5.00 → loss 0.895  ← rebounded at end
train_loss: 1.017
```

Loss rebounded at epoch 5, indicating instability — a sign of too few examples
relative to model capacity.

**Mitigation implemented:** `_enforce_required_reads()` in `lora_agent.py` now
intercepts any `final_answer` call before `get_case`, `lookup_customer`,
`search_invoices`, and `search_payments` have all been completed, and redirects to
the next missing required tool. This prevents mode collapse without retraining.

---

## Finding 3 — One task fails across all agents: `task_credit_memo_reconciled`

Both the baseline and the sklearn agent fail this task. The LoRA agent also failed it
in Run 1.

**Root cause:** `search_credit_memos` appears only once across all 28 training
examples (1 out of 4 traces uses it). The classifier's probability for this class is
near-uniform at the decision point (0.053), making it unreliable.

Without the credit memo check, the agent sees: payment (4,000,000¢) < invoice
(4,500,000¢) and opens a partial-payment exception instead of resolving as
`paid_after_credit_memo`.

This is the canonical **long-tail problem in SFT**: rare behaviors require
disproportionately more training examples than common ones.

**Fix:** 5–10 additional traces where a partial payment is resolved via credit memo
would likely make this classification reliable.

---

## Finding 4 — CUDA driver mismatch (resolved)

Run 1 was affected by a CUDA driver mismatch:

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
(found version 12080).
```

PyTorch 2.12.0 requires CUDA 13.x; the H200 node runs CUDA 12.8 (driver 570.195.03).
Fixed by reinstalling PyTorch with the correct index URL:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128 --break-system-packages
```

Both `lora_train.py` and `lora_agent.py` now auto-detect CUDA at runtime and use
bfloat16 on the H200, falling back to float32 on CPU.

---

## Finding 5 — SybilAgent (Sybil API / GPT-OSS-20B) could not be evaluated on GPU node

The Sybil LLM baseline (`openai/gpt-oss-20b` via `https://api.sybil.com/v1`) was
implemented and tested locally but could not run on the Targon GPU node because the
node's network policy blocks outbound HTTPS to `api.sybil.com`. The `--no-sybil` flag
was added to allow the benchmark to run without it.

The intended comparison story is: **Sybil LLM (frontier baseline) → sklearn SFT →
LoRA SFT**. The Sybil baseline is expected to pass all 5 tasks using zero training data,
establishing the quality ceiling a small fine-tuned model should approach.

---

## Proposed Improvements

### 1. Minimum-steps guard before `final_answer` (implemented)

`_enforce_required_reads()` in `lora_agent.py` blocks `final_answer` until all four
required reads are complete, then falls back to the sklearn PolicyAgent for correct
tool selection. This addresses mode collapse without requiring retraining.

### 2. Expand training traces (highest impact)

28 examples is too few for a 0.5B parameter generative model. Target:

- 100–200 total examples
- At least 5–10 `search_credit_memos` examples to fix the class imbalance
- **Data flywheel:** run SybilAgent against synthetic task variants and collect
  successful traces as additional training data

### 3. Increase epochs with early stopping

With more data, increase to 15–20 epochs. The loss rebound at epoch 5 suggests the
current setup needs an eval split to detect overfitting.

### 4. Add tool schema to the system prompt

Explicitly including the JSON schema for each tool gives the model a structured
reference and reduces hallucinated argument names.

### 5. Constrain generation with structured decoding

Use constrained beam search or grammar-based sampling (e.g., `outlines` library) to
force model output to be valid JSON matching the tool schema. Eliminates the need for
the fallback parser.

### 6. Larger base model

Qwen2.5-1.5B or 3B would have significantly better instruction-following out of the
box and would require fewer examples to fine-tune reliably. The LoRA adapter stays
small regardless of base model size.

---

## Interview Discussion Points

**Lead with the mode collapse result — it is the most interesting finding.** The model
found a zero-penalty strategy (call `final_answer` immediately) that passes all safety
criteria (observed_evidence, forbidden_evidence, unsafe_mutation all 100%) while failing
all correctness criteria. This is a perfect illustration of why reward design and
product metrics must be tracked separately.

**The sklearn vs. LoRA comparison makes a clear point:** with only 28 examples, a
simple linear classifier outperforms a 0.5B generative model. More data trumps model
size at this scale.

**The data flywheel is the production path:** use the Sybil API to generate synthetic
traces at scale, retrain the LoRA adapter, and measure on a hidden eval split. The
benchmark harness already supports this loop without modification.

**On SFT vs. DPO vs. GRPO:** SFT is correct here given clean demonstration traces.
DPO becomes relevant once you accumulate (correct, incorrect) pairs from analyst
corrections. GRPO/RLVR makes sense when the model needs to explore beyond the
demonstration distribution — this benchmark's verifier is already the right shape
for that reward signal.
