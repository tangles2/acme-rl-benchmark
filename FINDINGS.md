# Findings - Acme RL Finetuning Benchmark

**Hardware:** Targon H200 GPU node, CUDA 12.8, driver 570.195.03
**Models:** Rule-based baseline, TF-IDF + LogisticRegression (sklearn), Qwen/Qwen2.5-0.5B-Instruct + LoRA (rank=8, alpha=16, 5 epochs)
**Training data:** 28 examples from 4 traces in training_traces.jsonl

---

## Results Summary

| Agent | Strict Pass Rate | Avg Score | Avg Tool Calls | Avg Broad Scans |
|---|---|---|---|---|
| Rule-based baseline | 0/5 (0%) | 86.3% | 6.6 | 2.0 |
| sklearn PolicyAgent | 4/5 (80%) | 90.0% | 6.6 | 0.0 |
| LoRA Qwen2.5-0.5B | 3/5 (60%) | 82.5% | 8.8 | 0.0 |

### Criteria by agent

| Criterion | Baseline | sklearn | LoRA |
|---|---|---|---|
| status | 80% | 80% | 60% |
| resolution | 80% | 80% | 60% |
| amounts | 80% | 80% | 80% |
| evidence | 60% | 80% | 60% |
| observed_evidence | 100% | 100% | 100% |
| forbidden_evidence | 100% | 100% | 100% |
| unsafe_mutation | 100% | 100% | 100% |
| tool_efficiency | 0% | 100% | 100% |

---

## Finding 1 - LoRA ran on GPU but leaned heavily on the fallback

LoRA training finished in 22 seconds on the H200 using bfloat16, down from about
10 minutes on CPU in Run 1. The adapter saved correctly and inference loaded clean.

The fallback counts in the log tell the real story though:

```
task=case-1001  used 6 fallbacks
task=case-1002  used 7 fallbacks
task=case-1003  used 7 fallbacks
task=case-1004  used 6 fallbacks
task=case-1005  used 4 fallbacks
```

Each task ran 6-7 steps total. That many fallbacks per task means the model was
producing unparseable output on almost every step and the sklearn PolicyAgent had
to take over each time. The LoRA adapter is not generating reliable JSON tool calls
on its own yet.

The 3/5 pass rate is mostly the sklearn PolicyAgent covering for it, not the LoRA
model solving tasks independently. That is an honest result and worth being upfront
about.

---

## Finding 1b - Trained classifier is not load-bearing; scaffolding is

An audit of the scoring results revealed that a constant "always call search_credit_memos"
rule scores 5/5 while the trained sklearn classifier only scores 4/5. This means the
Phase 1 / Phase 3 deterministic scaffolding in PolicyAgent (required reads, safe mutation
ordering, hard-coded final_answer schema) is responsible for passing the benchmark, not
the learned classifier.

The classifier does improve average score and eliminates broad scans, but neither of those
changes crosses a task from failing to passing in the current benchmark. The practical
takeaway: if you want to know what is doing the work here, it is the scaffolding design,
not the training step.

This is worth stating clearly because the naive interpretation of "4/5 after SFT" makes it
sound like the model learned to do the job. What actually happened is the scaffolding
already knew how to do most of it, and the model learned a few stylistic improvements on
top. The classifier would need to make better decisions on the task boundaries (credit memo
vs partial payment) before it earns the credit.

The V2 synthetic flywheel (MemoAlwaysAgent) is a step toward fixing the underlying data
problem for LoRA fine-tuning.

---

## Finding 2 - sklearn still beats LoRA at this data scale

sklearn passed 4/5 tasks. LoRA passed 3/5 and actually regressed on
`task_missing_evidence`, which sklearn got right.

At 28 training examples, a TF-IDF + LogisticRegression classifier has far fewer
parameters to fit and generalizes better from a small dataset. The 0.5B parameter
generative model has to learn reliable JSON formatting, argument structure, and task
sequencing all at the same time from 28 examples. That is too much to ask of it.

Training loss from the LoRA run:

```
epoch 0.71  loss 1.528
epoch 1.43  loss 1.262
epoch 2.14  loss 1.042
epoch 2.86  loss 0.873
epoch 3.57  loss 0.816
epoch 4.29  loss 0.776
epoch 5.00  loss 0.903  <-- rebounded at end
train_loss: 1.029
```

The loss rebounding at epoch 5 is the same pattern as Run 1. More data would help
more than more epochs here.

---

## Finding 3 - Mode collapse is fixed but task_missing_evidence is a new failure

In Run 1 the LoRA agent called `final_answer` first on every task without doing any
lookups. The fix was a required-reads guard in `lora_agent.py` that blocks
`final_answer` until `get_case`, `lookup_customer`, `search_invoices`, and
`search_payments` have all been called. That worked.

But `task_missing_evidence` is now failing differently. The agent hit the 16-step
limit (`calls=16`) and still got status and resolution wrong. The model kept calling
tools past the point where it should have wrapped up. The guard is stopping it from
quitting too early, but the model has not learned when it actually should stop.

---

## Finding 4 - task_credit_memo_reconciled fails across all agents

Every agent fails this one. `search_credit_memos` appears only once in all 28
training examples. The classifier probability for this class is so low it never gets
selected at the right decision point.

Without the credit memo check, the agent sees payment (4,000,000 cents) less than
invoice (4,500,000 cents) and opens a partial payment exception instead of resolving
as `paid_after_credit_memo`.

This is a data problem. A handful of additional traces that use `search_credit_memos`
would likely fix it for both sklearn and LoRA.

The V2 flywheel now includes a MemoAlwaysAgent that forces search_credit_memos calls for
credit memo synthetic tasks, which should address this class imbalance before LoRA training.
The audit also confirmed that a constant "always check memos" rule resolves this task, so
the fix direction is clear -- the model just needs to see enough examples to learn it.

---

## Finding 5 - Sybil LLM baseline could not run on the GPU node

The Sybil API (`openai/gpt-oss-20b` via `api.sybil.com`) is implemented and was
tested locally, but the Targon GPU node blocks outbound HTTPS to that endpoint.
The `--no-sybil` flag routes around it. The intended comparison was Sybil vs sklearn
vs LoRA, with the frontier model setting the quality ceiling. That piece is still
missing from the results but the code is all there.

---

## What would actually improve the LoRA results

**More training data is the highest leverage change.** 28 examples is not enough for
a generative model to learn reliable JSON output formatting and task sequencing at the
same time. Realistically you want 100-200 examples with at least 5-10 per rare class
like `search_credit_memos` and `draft_slack_message`. The quickest way to get there is
running SybilAgent against synthetic task variants and collecting the successful traces.

**More epochs with an eval split.** The loss is still rebounding at epoch 5. With more
data, training to 15-20 epochs with early stopping would help stabilize it.

**Tool schema in the system prompt.** Right now the prompt describes the workflow in
plain English. Explicitly listing each tool and its argument names gives the model a
structured reference and would cut down on malformed JSON.

**Larger base model.** Qwen2.5-1.5B or 3B follows instructions better out of the box
and would need fewer examples to get to reliable tool call formatting. The LoRA adapter
stays small regardless of which base model you use.

**Structured decoding.** Constrained generation (e.g. the outlines library) forces
output to be valid JSON matching the tool schema. That would cut the fallback rate
dramatically without needing to retrain.

---

## What to talk about with Manifold

The mode collapse result from Run 1 is worth leading with. The model found that
calling `final_answer` immediately scores well on safety criteria (`observed_evidence`,
`forbidden_evidence`, and `unsafe_mutation` all pass) while failing all the
correctness checks. That is exactly the kind of reward hacking the benchmark is
designed to catch, and it caught it.

The sklearn vs LoRA comparison makes a clear point about data scale. At 28 examples
the simpler model wins. That is not a failure of the approach - it is an honest
measurement of where the data situation is right now.

The benchmark itself is the real deliverable. The environment resets cleanly, all 8
scoring criteria tie to business-correct outcomes, reward hacking is caught by the
verifier, and the pipeline runs from a single command. The fine-tuned model being small
is expected. The benchmark being trustworthy is what matters.

---

## Known Benchmark Gaps

These are edge cases that exist in real collections workflows but are not currently
tested. They would be the first additions before moving this to production.

### 1. Multiple credit memos on a single invoice

All fixtures have at most one credit memo per invoice. A real case might have CM-101
and CM-102 both partially offsetting the same invoice. The scorer checks
`paid + credit >= invoice_amount`, but nothing verifies the agent correctly sums
multiple memos rather than stopping at the first one. An agent that reads only the
most recent credit memo would pass every current test and fail in production.

**Fix:** Add a task where two credit memos together cover the gap but neither alone
does. The agent has to sum both results from `search_credit_memos`.

### 2. Overpayment

No task tests the case where `payment_amount > invoice_amount`. An agent that
computes `remaining = invoice - paid` naively would produce a negative exception
amount. The benchmark has no assertion for this so the bug would be invisible until
a real customer overpaid.

**Fix:** Add a task with `payment_amount = invoice_amount + N` and assert the case
resolves as `paid_in_full` with no exception opened.

### 3. Payment referencing the wrong invoice

All fixture payments have the correct `invoice_id` for the case being worked. In
production a customer might send a wire that gets applied to the wrong invoice. If
the agent accepts any payment for the right `customer_id`, it would resolve
incorrectly. The scorer does not currently check which `invoice_id` a payment
references.

**Fix:** Add a task where a payment exists for the customer but points to a different
invoice. The agent has to filter by `invoice_id` and escalate as
`missing_payment_evidence`.

### 4. Silent tool failures from bad argument types

The mock accepts any arguments and returns empty results when nothing matches. A real
API would reject `search_invoices(customer_id=None)` with a 400 error. Because the
mock is permissive, an argument bug silently produces an empty result and the only
signal is a wrong `resolution` downstream. This makes debugging harder than it needs
to be.

**Fix:** Add argument validation to the mock (`ValueError` on missing required fields)
and add a `tool_call_invalid_rate` metric to the scorer.

### 5. Repeated narrow tool calls

`MAX_STEPS` prevents infinite loops but nothing penalizes calling `search_invoices`
three times with the same arguments. The `tool_efficiency` metric only penalizes broad
scans, not redundant narrow calls. A model could call the same safe read tool
repeatedly to pad its trace before calling `final_answer` and the scorer would not
catch it.

**Fix:** Track call counts per tool per run. Penalize any tool called more than once
with identical arguments - after the first call the result is already in the trace
and there is no new information to gain.

---

## V2 Train/Eval Leakage Note

The V2 pipeline has a known leakage between the synthetic training set and the
synthetic eval set.

**What happens:** `build_combined_traces()` runs the V1 PolicyAgent on all 8 synthetic
tasks and collects those traces as training data. `V2Benchmark` then evaluates LoRA
models on those same 8 tasks. The training data is derived directly from the eval set.

**What this means:** V2 LoRA scores on synthetic tasks are optimistic. The model has
seen (via its training traces) the exact tasks it gets evaluated on. Do not read those
numbers as generalization performance.

**What is clean:**
- The 5 original tasks from the fixtures are never touched during synthetic trace
  generation. V2 scores on those 5 tasks are valid.
- `task_missing_evidence` is held out from synthetic generation entirely. Its score
  is a true out-of-distribution result.

**The fix:** Split synthetic tasks at generation time - some for training traces, the
rest held out for eval only. With the current 8 synthetic tasks this would leave only
4 training examples per type, so more task variants need to be generated first.
`SyntheticMockEnvironment` and `V2Benchmark` are already structured to support this -
it is a data-authoring task, not an architecture change.
