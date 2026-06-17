# Acme Finance Operations - RL Finetuning Benchmark

A benchmark harness and finetuning pipeline for a finance collections workflow,
built to the Manifold Labs take-home spec.

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
│   ├── sybil_agent.py               # LLM baseline via Sybil API
│   ├── lora_train.py                # LoRA fine-tuning (Qwen2.5-0.5B-Instruct)
│   └── lora_agent.py                # inference agent wrapping LoRA adapter
├── src/v2/                          # V2 pipeline (multi-model, synthetic traces, DPO)
│   ├── synthetic.py
│   ├── multi_lora.py
│   └── dpo_train.py
├── tests/                           # pytest unit tests
│   ├── test_environment.py
│   └── test_scorer.py
├── artifacts/                       # created on first run
│   ├── next_action_policy.pkl
│   └── lora_adapter/
├── FINDINGS.md                      # run results and failure analysis
├── run_benchmark.py                 # V1 entry point
├── run_benchmark_v2.py              # V2 entry point (multi-model + DPO)
└── requirements.txt
```

---

## Run

```bash
# Full pipeline (requires SYBIL_API_KEY and GPU with HuggingFace access)
python3 run_benchmark.py

# Skip Sybil LLM baseline (no API key needed)
python3 run_benchmark.py --no-sybil

# Skip LoRA step entirely (sklearn only, runs on any laptop in under 5 seconds)
python3 run_benchmark.py --no-sybil --no-lora

# V2 pipeline (multi-model LoRA, synthetic traces, DPO)
python3 run_benchmark_v2.py --skip-dpo
```

The V1 pipeline runs four steps:

1. **Sybil LLM baseline** - `openai/gpt-oss-20b` via the Sybil API. Sets the quality
   ceiling a fine-tuned small model should approach. Requires `SYBIL_API_KEY`.
2. **sklearn SFT** - extracts (context -> next-tool) pairs from `training_traces.jsonl`,
   fits TF-IDF + LogisticRegression, saves artifact to `artifacts/`.
3. **PolicyAgent** - runs the trained classifier on all five tasks and prints a
   before/after delta vs the rule-based baseline.
4. **LoRA fine-tuning** - fine-tunes Qwen2.5-0.5B-Instruct with LoRA on the same
   traces, saves adapter to `artifacts/lora_adapter/`, scores all five tasks.

---

## Tests

```bash
pip install pytest --break-system-packages
python3 -m pytest tests/ -v
```

40 tests covering all 8 scoring criteria, unsafe mutation detection, broad scan
counting, observed evidence checks, and integer amount math.

---

## Architecture

### Mock tool environment (`src/environment.py`)

Loads fixture CSVs into memory. Implements all nine tools from `tool_schema.json`:

`get_case` / `lookup_customer` / `search_invoices` / `search_payments` /
`search_credit_memos` / `update_case` / `create_exception` /
`draft_slack_message` / `final_answer`

Tracks per-run state: observed evidence IDs, broad scan count, unsafe mutation flag.
Resets completely before each task.

Unsafe mutations set the flag rather than throwing an exception, so the scorer can
report them without crashing the run:
- `update_case(status="resolved")` before observing both invoice and payment evidence
- `create_exception` without a cited invoice ID
- Citing evidence IDs that were never returned by a prior tool call

Broad scan: any `search_invoices` call without `invoice_id` or `month`, or any
`search_payments` call without `invoice_id`, adds one to the broad scan count (-0.05/scan).

### Baseline agent (`BaselineAgent`)

Five deliberate weaknesses that mirror common model failure modes:

| Weakness | Effect on scoring |
|---|---|
| Broad `search_invoices` (no filter) | efficiency penalty |
| Broad `search_payments` (no filter) | efficiency penalty |
| Never calls `search_credit_memos` | fails `task_credit_memo_reconciled` |
| Skips `draft_slack_message` | missed Slack evidence |
| Omits `customer_id` from evidence | fails ambiguous-customer evidence check |

### Sybil LLM baseline (`src/sybil_agent.py`)

Uses the OpenAI-compatible Sybil API with model `openai/gpt-oss-20b`. Converts all
nine tools to OpenAI function-calling schema and runs a tool-calling loop against
the mock environment (max 14 steps). Set `SYBIL_API_KEY` to run.

### sklearn PolicyAgent (`PolicyAgent`)

Hybrid design: rules handle the deterministic parts, the trained classifier handles
the two decisions that actually require learning.

**Phase 1 - required reads (always enforced):**
`get_case` -> `lookup_customer` -> `search_invoices(narrow)` -> `search_payments(narrow)`

Arguments always include filters so broad scans are not possible.

**Phase 2 - learned decisions (trained classifier):**
- Should we call `search_credit_memos`?
- Should we call `draft_slack_message` after a paid-in-full resolution?

Both come from parameters trained on `training_traces.jsonl`.

**Phase 3 - safe mutations (rule-enforced):**
`create_exception` (if needed) -> `update_case` -> `final_answer`

Ordering is guaranteed regardless of what the classifier output.

### sklearn finetuning pipeline (`src/train.py`)

1. Parses `training_traces.jsonl` - each step becomes one training example.
   - **X:** task description + tool names called so far + key result values
   - **y:** next tool name to call
   - 4 traces x 7 steps = 28 training examples, 9 classes
2. Fits `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)`
3. Fits `LogisticRegression(C=1.0, max_iter=1000)`
4. Saves model to `artifacts/next_action_policy.pkl`

Training accuracy on fixture traces: **85.7%**

### LoRA fine-tuning (`src/lora_train.py` + `src/lora_agent.py`)

Fine-tunes Qwen/Qwen2.5-0.5B-Instruct with PEFT LoRA (rank=8, alpha=16, targeting
`q_proj` and `v_proj`). Trains for 5 epochs on 28 (prompt, completion) pairs formatted
as ChatML. Only 540k of 494M parameters are updated (0.11%).

Auto-detects GPU at runtime: bfloat16 on CUDA with bf16 support, float16 on older
CUDA, float32 on CPU.

The `LoRAAgent` runs greedy inference and parses output as JSON tool calls. A
required-reads guard blocks `final_answer` until the four required reads are done,
which fixes the mode collapse issue from Run 1. Falls back to sklearn PolicyAgent
when generation is unparseable.

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

What changed after training:
- Broad scans eliminated (policy always uses narrow filters)
- Evidence accuracy up: policy correctly cites `customer_id` for ambiguous customer cases
- Slack draft added for paid-in-full resolutions (learned from trace)

**Important caveat:** A follow-up audit found that a constant "always call search_credit_memos"
rule scores 5/5 while the trained sklearn classifier only scores 4/5. The Phase 1 and Phase 3
safety scaffolding is doing most of the heavy lifting. The classifier's learned decisions are
improving on average score and efficiency, but they are not the reason the system passes. A
re-read of these results should credit the scaffolding design over the learning step.

See `FINDINGS.md` for full failure analysis, LoRA results, and what to improve next.

---

## Failure Cases

### 1. `task_credit_memo_reconciled` - all agents fail

`search_credit_memos` appears only once in 28 training examples. The classifier
probability at the decision point is near-uniform (0.053), so it rarely gets picked.
Without the credit memo check, the agent sees payment (4,000,000 cents) less than
invoice (4,500,000 cents) and opens a partial payment exception instead of resolving
as `paid_after_credit_memo`.

Note: the audit found that a constant "always call search_credit_memos" rule solves this
and scores 5/5. The synthetic V2 flywheel now generates credit memo traces via a dedicated
MemoAlwaysAgent to fix the class imbalance before LoRA training.

### 2. `task_paid_in_full` (baseline) - efficiency-only failure

Passes all correctness checks but loses on tool efficiency from 2 broad scans.
The trained policy fixes this completely.

### 3. LoRA mode collapse (Run 1)

The LoRA agent called `final_answer` first on every single task in Run 1, scoring 0/5.
With only 28 examples, the model found the lowest-penalty path rather than the correct
workflow. Fixed by adding the required-reads guard in `lora_agent.py`. See FINDINGS.md.

---

## Reward Hacking Risks and Mitigations

| Risk | Mitigation |
|---|---|
| **Resolve without reading invoice/payment** | `update_case(status="resolved")` sets `unsafe_mutation=True` if the case invoice or any payment has not been observed in a prior tool result |
| **Always escalate to avoid wrong answers** | `create_exception` requires an invoice ID in evidence. Escalating without reading the invoice triggers `unsafe_mutation` |
| **Read entire workbooks** | Any unfiltered `search_invoices` or `search_payments` call increments `broad_scan_count`. Score is penalized -0.05 per scan |
| **Cite IDs that were never observed** | `observed_evidence` checks that every ID in `final_answer.evidence_ids` appears in `env.observed_ids`. Fabricated IDs fail even if the final state is correct |
| **Overfitting to fixture names** | The ambiguous customer task includes decoy IDs (`cus_acme_holdings`, `INV-2026-0521`) that are plausible matches. Citing them triggers `forbidden_evidence` failure |
| **Wrong evidence with correct final state** | The scorer checks both `evidence` (required IDs cited) and `observed_evidence` (all cited IDs actually returned). You cannot fabricate IDs to pass |

---

## Discussion Topics

**Production path:** Swap the sklearn classifier out for a LoRA-finetuned small LM
trained on analyst-labeled traces. The Phase 1 and Phase 3 safety scaffolding stays
the same - only Phase 2 gets upgraded.

**Collecting training traces:** Add a trace logger to the existing analyst workflow.
When an analyst closes a case, log every lookup and decision. Flag traces where
the analyst revised a decision - those are strong negative signal.

**Hidden eval split:** Reserve one complete customer scenario that no training trace
touches. Never tune on it.

**Reward design vs. product metrics:** Product metric is analyst hours saved. Reward
metric is correct status plus correct evidence in minimum tool calls. These can
diverge - a model that always escalates avoids wrong resolutions but saves zero hours.
Track both separately and alert if escalation rate spikes without a drop in analyst
corrections.

**SFT vs. DPO vs. GRPO:** SFT is the right starting point with clean demonstration
traces. Add DPO once you have (correct, incorrect) pairs from analyst corrections.
GRPO makes sense when you want the model to explore beyond the demonstration
distribution - this benchmark's verifier is already the right shape for that.

---

## Design Tradeoffs

**Hybrid agent instead of end-to-end LM.** The PolicyAgent splits things up: rules
handle the required read sequence and safe mutation ordering, and the trained
classifier only touches the two uncertain decisions. The downside is that a pure LM
would need the whole structure reworked before it could own the full trace.

**sklearn for the core result.** TF-IDF + LogisticRegression is interpretable,
CPU-only, and trains in under a second. It scores 4/5 without a GPU. The LoRA path
shows where this goes in production, but it needs a lot more data than 28 examples
to work reliably. Including both shows the tradeoff clearly: at this data scale, the
simpler model wins.

**Integer cents throughout.** All money values are stored and compared as `int`.
No divisions, no `float()` calls on amounts. The only float in the codebase is the
0.05-per-scan efficiency penalty, which is applied to scores, not amounts.

**In-memory mock instead of live API.** Runs are deterministic, fast (~2 seconds),
and need no credentials. The downside is the mock is permissive - it accepts `None`
arguments and returns empty results rather than raising errors, so argument bugs can
hide. A production environment would validate inputs strictly.

**V2 synthetic traces share tasks with their eval set.** The V2 pipeline generates
training traces by running PolicyAgent on synthetic tasks, then evaluates LoRA on
those same tasks. That is a known leakage issue. The original five tasks are clean
and `task_missing_evidence` is held out entirely. V2 LoRA scores on synthetic tasks
should be read as optimistic. See FINDINGS.md for the full explanation.

---

## What I Would Improve With More Time

**More training data first.** The LoRA result (0/5 in Run 1, heavy fallback use in
Run 2) shows 28 examples is not enough. The next step is generating 200-500 traces
using the data flywheel in `run_benchmark_v2.py`, not switching to a bigger model.

**Argument validation in the mock.** Right now `search_invoices(customer_id=None)`
silently returns nothing. Adding a `ValueError` for missing required fields would
surface argument bugs immediately instead of letting them show up as wrong resolutions.

**Idempotency guards on mutations.** Calling `update_case` or `create_exception` twice
silently overwrites. A billing workflow should flag or reject that. The tests in
`tests/test_environment.py` document the current behavior as a known gap.

**Clean hidden eval split for V2.** The synthetic tasks should be split at generation
time - some for training traces, some held out for eval only. Right now only
`task_missing_evidence` is truly held out.

**Structured decoding.** The LoRA agent parses free-text JSON with a regex fallback.
Constrained generation (e.g. the outlines library) would cut the fallback rate to near
zero without retraining.

**Cost and latency metrics.** The scorer tracks tool calls and broad scans but not
wall time or token count. For a production comparison between models, those matter
as much as pass rate.
