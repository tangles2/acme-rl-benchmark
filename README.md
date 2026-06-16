# Acme Finance Operations — RL Finetuning Benchmark

A benchmark harness and finetuning pipeline for a finance collections workflow,
built to the Manifold Labs take-home specification.

**LoRA adapter:** https://huggingface.co/tangles2/acme-lora-qwen2.5-0.5b

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

For LoRA fine-tuning on GPU, install PyTorch with the matching CUDA index:

```bash
# CUDA 12.8 (e.g. Targon H200 node)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CPU only
pip install torch
```

**Project layout**

```
acme-rl-benchmark/
├── fixtures/rl-finetuning/          # provided fixtures (unchanged)
│   ├── cases.json
│   ├── expected_outputs.json
│   ├── tasks.jsonl
│   ├── tool_schema.json
│   ├── training_traces.jsonl
│   └── workbooks/                   # customers, invoices, payments, credit_memos CSVs
├── src/
│   ├── environment.py               # mock tool environment (9 tools)
│   ├── agents.py                    # BaselineAgent + PolicyAgent
│   ├── benchmark.py                 # scoring harness (8 criteria)
│   ├── train.py                     # sklearn SFT pipeline
│   ├── sybil_agent.py               # LLM baseline via Sybil API (Targon GPT-OSS-20B)
│   ├── lora_train.py                # LoRA fine-tuning (Qwen2.5-0.5B-Instruct)
│   └── lora_agent.py                # inference agent wrapping LoRA adapter
├── artifacts/                       # created on first run
│   ├── next_action_policy.pkl       # sklearn trained model
│   └── lora_adapter/                # LoRA adapter weights (after training)
├── FINDINGS.md                      # run results, failure analysis, improvements
├── run_benchmark.py                 # single entry point
└── requirements.txt
```

---

## Run

```bash
# Full pipeline (requires SYBIL_API_KEY and GPU with HuggingFace access)
python3 run_benchmark.py

# Skip Sybil LLM baseline (no API key needed)
python3 run_benchmark.py --no-sybil

# Skip LoRA step (sklearn only, runs on any laptop in under 5 seconds)
python3 run_benchmark.py --no-sybil --no-lora
```

The full pipeline runs four steps in sequence:

1. **Sybil LLM baseline** — `openai/gpt-oss-20b` via the Sybil API. Establishes the
   quality ceiling a fine-tuned small model should approach. Requires `SYBIL_API_KEY`
   environment variable.
2. **sklearn SFT** — extracts (context → next-tool) pairs from `training_traces.jsonl`,
   fits a TF-IDF + LogisticRegression classifier, saves artifact to `artifacts/`.
3. **PolicyAgent** — runs the trained sklearn classifier against all five tasks;
   prints a before/after delta versus the rule-based baseline.
4. **LoRA fine-tuning + inference** — fine-tunes Qwen2.5-0.5B-Instruct with LoRA on
   the same traces, saves adapter to `artifacts/lora_adapter/`, scores all five tasks.

---

## Architecture

### Mock tool environment (`src/environment.py`)

Loads fixture CSVs into memory. Implements all nine tools from `tool_schema.json`:

`get_case` · `lookup_customer` · `search_invoices` · `search_payments` ·
`search_credit_memos` · `update_case` · `create_exception` ·
`draft_slack_message` · `final_answer`

Tracks per-run state: observed evidence IDs, broad-scan count, unsafe-mutation flag.
Resets completely before each task.

Unsafe mutations raise the flag rather than throwing, so the scorer can report them
without crashing the run:
- `update_case(status="resolved")` before observing both invoice and payment evidence.
- `create_exception` without a cited invoice ID.
- Citing evidence IDs that were never returned by any tool call.

Broad scan: any `search_invoices` call without `invoice_id` or `month`, or any
`search_payments` call without `invoice_id`, counts as one broad scan (−0.05/scan).

### Baseline agent (`BaselineAgent`)

Five deliberate weaknesses that mirror common model failure modes:

| Weakness | Effect on scoring |
|---|---|
| Broad `search_invoices` (no filter) | efficiency penalty (broad scan) |
| Broad `search_payments` (no filter) | efficiency penalty (broad scan) |
| Never calls `search_credit_memos` | fails `task_credit_memo_reconciled` |
| Skips `draft_slack_message` | missed Slack evidence |
| Omits `customer_id` from evidence | fails ambiguous-customer evidence check |

### Sybil LLM baseline (`src/sybil_agent.py`)

Uses the OpenAI-compatible Sybil API (`https://api.sybil.com/v1`) with model
`openai/gpt-oss-20b`. Converts all nine tools to OpenAI function-calling JSON schema
and runs a tool-calling loop against the mock environment (max 14 steps).

Set `SYBIL_API_KEY` to run. The agent falls back to the rule-based baseline if the
API key is missing or the endpoint is unreachable.

### sklearn PolicyAgent (`PolicyAgent`)

Hybrid design: safety invariants enforced by rules; the two genuinely uncertain
decisions are made by learned parameters.

**Phase 1 — required reads (always enforced):**
`get_case` → `lookup_customer` → `search_invoices(narrow)` → `search_payments(narrow)`

Arguments always include `invoice_id` and `month` filters; no broad scans possible.

**Phase 2 — learned decisions (trained classifier):**
- Should we call `search_credit_memos`?
- Should we call `draft_slack_message` after a paid-in-full resolution?

Both decisions come exclusively from parameters trained on `training_traces.jsonl`.

**Phase 3 — safe mutations (rule-enforced):**
`create_exception` (if needed) → `update_case` → `final_answer`.
Correct ordering is guaranteed regardless of classifier output.

### sklearn finetuning pipeline (`src/train.py`)

1. Parses `training_traces.jsonl`; each step becomes one training example.
   - **Input (X):** task description + tool names called so far + key result values
   - **Label (y):** next tool name to call
   - 4 traces × 7 steps = 28 training examples, 9 classes
2. Fits `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)`.
3. Fits `LogisticRegression(C=1.0, max_iter=1000)`.
4. Saves model to `artifacts/next_action_policy.pkl`.

Training accuracy on fixture traces: **85.7%**

### LoRA fine-tuning pipeline (`src/lora_train.py` + `src/lora_agent.py`)

Fine-tunes Qwen/Qwen2.5-0.5B-Instruct using PEFT LoRA (rank=8, alpha=16, targeting
`q_proj` and `v_proj`). Trains for 5 epochs on the same 28 (prompt, completion) pairs,
formatted as ChatML. Only 540k of 494M parameters are trained (0.11%).

Auto-detects GPU at runtime: uses bfloat16 on CUDA hardware with bf16 support,
float16 on older CUDA, float32 on CPU.

The `LoRAAgent` runs greedy inference and parses the model output as JSON tool calls.
A required-reads guard prevents mode collapse (model calling `final_answer` before
completing the four required reads). Falls back to the sklearn PolicyAgent when
generation is unparseable.

---

## Before / After Results

| Metric | Baseline | sklearn (post-SFT) |
|---|---|---|
| **Strict pass rate** | 0/5 (0%) | **4/5 (80%)** |
| Avg score | 86.3% | **90.0%** |
| Avg tool calls | 6.6 | 6.6 |
| Avg broad scans | 2.0 | **0.0** |

| Criterion | Baseline | sklearn |
|---|---|---|
| status | 80% | 80% |
| resolution | 80% | 80% |
| amounts | 80% | 80% |
| evidence | 60% | **80%** |
| observed_evidence | 100% | 100% |
| forbidden_evidence | 100% | 100% |
| unsafe_mutation | 100% | 100% |
| tool_efficiency | 0% | **100%** |

**What changed after training:**
- Broad scans eliminated (policy always uses narrow filters).
- Evidence accuracy improved: policy correctly cites `customer_id` for ambiguous-customer cases.
- Slack draft added for resolved paid-in-full cases (learned from trace).

See `FINDINGS.md` for full failure analysis, LoRA results, and improvement proposals.

---

## Failure Cases

### 1. `task_credit_memo_reconciled` — all agents fail

`search_credit_memos` appears only once in 28 training examples. The classifier
probability for this class is near-uniform (0.053) at the decision point, making
it unreliable. Without the credit memo check, the agent sees payment (4,000,000¢)
< invoice (4,500,000¢) and incorrectly opens a partial-payment exception.

### 2. `task_paid_in_full` (baseline) — efficiency-only failure

Passes all correctness checks but loses on tool efficiency (2 broad scans, no filters).
The trained policy eliminates this.

### 3. LoRA mode collapse (Run 1)

The fine-tuned LoRA agent called `final_answer` immediately on every task, scoring 0/5.
With only 28 examples, the model converged on the lowest-penalty strategy rather than
the correct workflow. See FINDINGS.md for full analysis and the mitigation implemented.

---

## Reward Hacking Risks and Mitigations

| Risk | Mitigation |
|---|---|
| **Resolve without reading invoice/payment evidence** | `update_case(status="resolved")` sets `unsafe_mutation=True` if the case invoice ID has not been observed in a prior `search_invoices` result, or if no payment ID has been observed. |
| **Always escalate to avoid wrong answers** | Escalation requires an invoice evidence ID in `create_exception`. Escalating without reading the invoice triggers `unsafe_mutation`. |
| **Read entire workbooks (broad scans)** | Any `search_invoices` call without `invoice_id`/`month`, or `search_payments` without `invoice_id`, increments `broad_scan_count`. Score is penalized −0.05 per scan. |
| **Cite IDs that were never observed** | `observed_evidence` criterion checks that every ID in `final_answer.evidence_ids` appears in `env.observed_ids`. Fabricated IDs fail this check even if final state is correct. |
| **Overfitting to public fixture names** | The ambiguous-customer task includes decoy IDs (`cus_acme_holdings`, `INV-2026-0521`) that are plausible matches. Citing them triggers `forbidden_evidence` failure. |
| **Passing final-state checks with wrong supporting evidence** | The scorer checks both `evidence` (required IDs cited) and `observed_evidence` (all cited IDs actually returned). A model cannot fabricate IDs to pass correctness checks. |

---

## Discussion Topics

**Production path:** Replace the sklearn classifier with a LoRA-finetuned small LM
trained on analyst-labeled traces. The Phase 1 / Phase 3 safety scaffolding stays the
same; only the Phase 2 classifier is upgraded.

**Collecting training traces:** Embed a trace logger in the existing analyst workflow.
When an analyst closes a case, log every lookup and decision. Flag traces where the
analyst had to revise a decision (strong negative signal).

**Hidden eval split:** Reserve one complete customer scenario that no training trace
touches. Never tune on it.

**Reward design vs. product metrics:** Product metric = analyst hours saved. Reward
metric = correct status + correct evidence in minimum tool calls. These can diverge:
a model that always escalates saves zero hours but avoids wrong resolutions. Track both
separately; alarm if escalation rate spikes.

**SFT vs. DPO vs. GRPO:** SFT is correct here given clean demonstration traces. Add
DPO once you have (correct, incorrect) pairs from analyst corrections. GRPO/RLVR makes
sense when you want the model to explore beyond the demonstration distribution — this
benchmark's verifier is already the right shape for that reward signal.

---

## Design Tradeoffs

**Hybrid agent over end-to-end LM.** The PolicyAgent splits responsibility: deterministic rules enforce the required read sequence and safe mutation ordering; a trained classifier handles only the two genuinely uncertain decisions (check credit memos? send Slack?). This keeps the benchmark honest — the trained component can only influence outcomes it should influence — but it means the system would need restructuring before a pure LM could own the whole trace.

**sklearn over LoRA for the core result.** A TF-IDF + LogisticRegression classifier is interpretable, CPU-friendly, and trainable in under a second. It scores 4/5 on the original tasks without a GPU. The LoRA path shows the production direction (small LM, adapter-based fine-tuning) but requires far more training data than the 28 fixture examples provide. Both are included because the comparison is informative: more data beats bigger model at this scale.

**Integer cents throughout.** All money values are stored and compared as `int` (cents). No divisions, no `float()` casts on amounts. This eliminates floating-point rounding as a failure mode. The one float in the codebase is the 0.05-per-scan efficiency penalty, which applies to scores, not amounts.

**Mock environment over live API.** The benchmark runs entirely in-memory against fixture CSVs. This makes runs deterministic, fast (~2 seconds), and reviewable without credentials. The tradeoff is that the mock is permissive — it accepts `None` arguments and returns empty results rather than raising errors, which can hide argument bugs. A production environment would validate inputs.

**V2 synthetic traces share tasks with their eval set.** The V2 pipeline generates training traces by running the V1 PolicyAgent on synthetic tasks, then evaluates LoRA models on those same tasks. This is a known leakage: V2 LoRA scores on synthetic tasks are optimistic. The original five tasks (V1 fixtures) are clean — they are never touched during synthetic generation — and `task_missing_evidence` is held out entirely from both generation and synthetic training. See FINDINGS.md for details.

---

## What I Would Improve With More Time

**More training data before a bigger model.** The LoRA result (0/5) shows that 28 examples is not enough for a generative model to learn this workflow. The next step is generating 200–500 traces using the data flywheel in `run_benchmark_v2.py`, not switching to a larger base model.

**Argument validation in the mock.** `search_invoices(customer_id=None)` currently silently returns nothing. Adding a `ValueError` for missing required fields would make agent bugs immediately visible rather than manifesting as wrong resolutions downstream.

**Idempotency guards on mutations.** Calling `update_case` or `create_exception` twice silently overwrites. A billing-safe environment should flag or reject duplicate mutation calls. The idempotency tests in `tests/test_environment.py` document the current behavior explicitly.

**Hidden eval split with no leakage.** The V2 synthetic tasks should be split at generation time: half used for training traces, half held out for eval only. Currently only `task_missing_evidence` is truly held out.

**Structured decoding.** The LoRA agent parses free-text JSON output with a regex fallback. Constrained generation (e.g., `outlines` library) would eliminate parse failures entirely and remove the need for the fallback to PolicyAgent.

**Cost and latency metrics.** The scorer tracks tool calls and broad scans but not wall-clock latency or token count. For a production system where Acme is choosing between models on cost vs. accuracy, these matter as much as pass rate.
