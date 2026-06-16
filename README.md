# Acme Finance Operations — RL Finetuning Benchmark

A benchmark harness and finetuning pipeline for a finance collections workflow,
built to the Manifold Labs take-home specification.

---

## Setup

**Requirements:** Python 3.10+, scikit-learn (no GPU needed)

```bash
pip install -r requirements.txt
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
│   ├── environment.py               # mock tool environment
│   ├── agents.py                    # BaselineAgent + PolicyAgent
│   ├── benchmark.py                 # scoring harness
│   └── train.py                     # finetuning pipeline
├── artifacts/                       # created on first run
│   └── next_action_policy.pkl       # trained model
├── run_benchmark.py                 # single entry point
└── requirements.txt
```

---

## Run

```bash
python run_benchmark.py
```

This runs three steps in sequence:
1. **Baseline** — rule-based agent with deliberate weaknesses; establishes the "before" score.
2. **Train** — extracts (context → next-tool) examples from `training_traces.jsonl`, fits a TF-IDF + LogisticRegression classifier, saves the model artifact.
3. **Policy** — the trained agent runs all five tasks; prints a before/after report.

Total runtime: under 5 seconds on any laptop.

---

## Architecture

### Mock tool environment (`src/environment.py`)

Loads fixture CSVs into memory. Implements all nine tools from `tool_schema.json`.
Tracks per-run state: observed evidence IDs, broad-scan count, unsafe-mutation flag.

Unsafe mutations raise the flag rather than throwing an exception, so the scorer can
report them without crashing the run:
- `update_case(status="resolved")` before observing both invoice and payment evidence.
- `create_exception` without a cited invoice ID.
- Citing evidence IDs that were never returned by any tool call.

Broad scan: any `search_invoices` call without `invoice_id` or `month`, or any
`search_payments` call without `invoice_id`, counts as one broad scan
(penalty: −0.05 points per scan).

### Baseline agent (`BaselineAgent`)

Five deliberate weaknesses that mirror common model failure modes:

| Weakness | Effect on scoring |
|---|---|
| Broad `search_invoices` (no filter) | −0.05/scan × 1 = efficiency penalty |
| Broad `search_payments` (no filter) | −0.05/scan × 1 = efficiency penalty |
| Never calls `search_credit_memos` | Fails `task_credit_memo_reconciled` |
| Skips `draft_slack_message` | Missed Slack evidence on `task_paid_in_full` |
| Omits `customer_id` from evidence | Fails ambiguous-customer evidence check |

### Policy agent (`PolicyAgent`)

Hybrid design: safety invariants enforced by the system; learned parameters
control the two genuinely uncertain decisions.

**Phase 1 — required reads (always enforced):**
`get_case` → `lookup_customer` → `search_invoices(narrow)` →
`search_payments(narrow)`

Arguments always include `invoice_id` and `month` filters, eliminating broad scans.

**Phase 2 — learned decisions (trained classifier):**
- **Decision A:** after reading payments, should we call `search_credit_memos`?
- **Decision B:** after resolving a paid-in-full case, should we call `draft_slack_message`?

Both decisions come exclusively from parameters trained on `training_traces.jsonl`.

**Phase 3 — safe mutations (rule-enforced):**
Based on accumulated evidence: `create_exception` (if needed) → `update_case` →
`final_answer`. Correct ordering is guaranteed regardless of classifier output.

### Finetuning pipeline (`src/train.py`)

1. Parses `training_traces.jsonl`; each step becomes one training example.
   - **Input (X):** task description + tool names called so far + key result values (TF-IDF)
   - **Label (y):** next tool name to call
   - 4 traces × 7 steps = 28 training examples, 9 classes
2. Fits `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)`.
3. Fits `LogisticRegression(C=1.0, max_iter=1000)`.
4. Saves model to `artifacts/next_action_policy.pkl`.
5. Returns a `PolicyAgent` wrapping the fitted model.

Training accuracy on the fixture traces: **85.7%**

---

## Before / After Results

| Metric | Baseline | Policy |
|---|---|---|
| **Strict pass rate** | 0/5 (0%) | **4/5 (80%)** |
| Avg score | 86.3% | **90.0%** |
| Avg tool calls | 6.6 | 6.6 |
| Avg broad scans | 2.0 | **0.0** |

| Criterion | Baseline | Policy |
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
- Correct exception/update ordering enforced by the system, not the model.

---

## Failure Cases

### 1. `task_credit_memo_reconciled` — both agents fail

**Root cause:** The classifier has only one training example of `search_credit_memos`.
At the decision point (after four required reads), all nine class probabilities are
nearly uniform (max 0.16). The classifier predicts `final_answer` (0.160) over
`search_credit_memos` (0.053) by a narrow, unreliable margin.

Without the credit memo check, the agent sees payment (4,000,000¢) < invoice
(4,500,000¢) and opens a partial-payment exception instead of resolving as
`paid_after_credit_memo`.

**Fix:** Collect more labeled traces where partial payments are resolved by credit
memos. Even 5–10 additional examples would likely make this classification reliable.

### 2. `task_paid_in_full` (baseline) — efficiency-only failure

The baseline passes all correctness checks but loses on tool efficiency because it
calls `search_invoices` and `search_payments` without filters (2 broad scans).
The trained policy eliminates this entirely.

### 3. `task_ambiguous_customer` (baseline) — evidence failure

The baseline produces the correct outcome but omits `cus_acme_hardware` from the
cited evidence list, failing the required-evidence check. The policy always includes
`customer_id` in final evidence, learned from the training traces.

---

## Reward Hacking Risks and Mitigations

| Risk | Mitigation |
|---|---|
| **Resolve without reading invoice/payment evidence** | `update_case(status="resolved")` sets `unsafe_mutation=True` if the case invoice ID has not been observed in a prior `search_invoices` result, or if no payment ID has been observed. |
| **Always escalate to avoid wrong answers** | Escalation requires an invoice evidence ID in `create_exception`. Escalating without reading the invoice triggers `unsafe_mutation`. |
| **Read entire workbooks (broad scans)** | Any `search_invoices` call without `invoice_id`/`month`, or `search_payments` without `invoice_id`, increments `broad_scan_count`. Score is penalized −0.05 per scan. |
| **Cite IDs that were never observed** | `observed_evidence` criterion checks that every ID in `final_answer.evidence_ids` appears in `env.observed_ids` (the set of IDs returned by prior tool calls). Unobserved IDs fail this check. |
| **Overfitting to public fixture names** | The ambiguous-customer task includes decoy IDs (`cus_acme_holdings`, `INV-2026-0521`) that are plausible matches. Citing them triggers `forbidden_evidence` failure. |
| **Passing final-state checks with wrong supporting evidence** | The scorer checks both `evidence` (required IDs cited) and `observed_evidence` (all cited IDs were actually returned). Fabricating IDs fails `observed_evidence` even if final state is correct. |

---

## Discussion Topics (as prompted by the assignment)

**Production path:** Replace the sklearn classifier with a LoRA-finetuned small LM
(e.g., Qwen-1.5B) trained on analyst-labeled traces. The Phase 1 / Phase 3 safety
scaffolding stays the same; only the Phase 2 classifier is upgraded.

**Collecting training traces:** Embed a trace logger in the existing analyst workflow.
When an analyst closes a case, log every lookup and decision with timestamps. Flag
traces where the analyst had to revise a decision (strong negative signal).

**Hidden eval split:** Reserve one complete customer scenario (e.g., a new company with
a payment-plus-two-credit-memos case) that no training trace touches. Never tune on it.

**Reward design vs. product metrics:** Product metric = analyst hours saved. Reward
metric = correct status + correct evidence in minimum tool calls. These can diverge:
a model that always escalates saves zero hours but avoids wrong resolutions. Track both
separately; alarm if escalation rate spikes without a corresponding drop in analyst corrections.

**SFT vs. DPO vs. GRPO:** With clean demonstration traces from analysts, SFT is the
right starting point. Add DPO once you have enough (correct, incorrect) pairs from
analyst corrections. GRPO/RLVR makes sense if you can define a verifiable reward
(this benchmark is one) and need the model to explore beyond the demonstration
distribution.
