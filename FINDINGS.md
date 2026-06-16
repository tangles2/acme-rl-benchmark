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

LoRA training completed in 22 seconds on the H200 using bfloat16, down from roughly
10 minutes on CPU in Run 1. The adapter saved correctly and inference loaded clean.

That said, the per-task fallback counts in the log tell you what actually happened:

```
task=case-1001  used 6 fallbacks
task=case-1002  used 7 fallbacks
task=case-1003  used 7 fallbacks
task=case-1004  used 6 fallbacks
task=case-1005  used 4 fallbacks
```

Each task ran 6-7 steps. A fallback count that high means the model was generating
unparseable output on almost every step, and the PolicyAgent had to take over each
time. The LoRA adapter is not yet producing reliable JSON tool calls on its own.

So the 3/5 pass rate is mostly the sklearn PolicyAgent covering for the LoRA model,
not the LoRA model solving tasks by itself. That is an honest result worth calling out.

---

## Finding 2 - sklearn still outperforms LoRA at this data scale

The sklearn agent passed 4/5 tasks. LoRA passed 3/5 and actually regressed on
task_missing_evidence, which sklearn handled correctly.

This lines up with what you would expect at 28 training examples. A TF-IDF +
LogisticRegression classifier has far fewer parameters to fit and generalizes better
from a small dataset. The 0.5B parameter generative model needs more data to learn
reliable JSON output formatting, argument structure, and task sequencing all at once.

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

The rebound at epoch 5 is the same pattern as Run 1. The model is not converging
cleanly. More data would help here more than more epochs.

---

## Finding 3 - Mode collapse is fixed but the model still fails task_missing_evidence

In Run 1, the LoRA agent called final_answer immediately on every task without doing
any lookups. We fixed that by adding a required-reads guard in lora_agent.py that
blocks final_answer until get_case, lookup_customer, search_invoices, and
search_payments have all been called.

That fix worked. The model is no longer short-circuiting to final_answer on step 1.

But task_missing_evidence is a new failure. The agent hit the 16-step limit (calls=16)
and still got status and resolution wrong. The model kept calling tools past where it
should have wrapped up. The fallback guard is keeping it from quitting too early, but
the model is not learning when it actually should stop either.

---

## Finding 4 - task_credit_memo_reconciled fails across all agents

Every agent fails this task. The root cause is that search_credit_memos appears only
once in all 28 training examples. The classifier probability for this class is so low
that it never gets selected at the right decision point.

Without the credit memo check, the agent sees payment (4,000,000 cents) less than
invoice (4,500,000 cents) and opens a partial payment exception instead of resolving
as paid_after_credit_memo.

This is a data problem, not a model problem. A few more traces that use
search_credit_memos would likely fix this for both the sklearn and LoRA agents.

---

## Finding 5 - Sybil LLM baseline could not run on the GPU node

The Sybil API (openai/gpt-oss-20b via api.sybil.com) was implemented and tested but
the Targon GPU node blocks outbound HTTPS to that endpoint. The --no-sybil flag routes
around it. The intended comparison was Sybil LLM baseline vs sklearn vs LoRA, with
the frontier model setting the quality ceiling. That piece is still missing from the
results but the code is there.

---

## What would actually improve the LoRA results

**More training data is the highest leverage thing.** 28 examples is not enough for a
generative model to learn reliable JSON output formatting and task sequencing at the
same time. Realistically you want 100-200 examples with at least 5-10 per rare class
like search_credit_memos and draft_slack_message. The quickest way to get there is
to run SybilAgent against synthetic task variants and collect the successful traces.

**More epochs with an eval split.** The loss is still rebounding at epoch 5. With more
data, training to 15-20 epochs with early stopping would help stabilize it.

**Tool schema in the system prompt.** Right now the prompt describes the workflow in
plain English. Explicitly listing each tool and its argument names gives the model a
reference and would cut down on malformed JSON.

**Larger base model.** Qwen2.5-1.5B or 3B would follow instructions better out of the
box and need fewer examples to get to reliable tool call formatting. The LoRA adapter
stays small regardless of which base model you use.

**Structured decoding.** Constrained generation (something like the outlines library)
forces the output to be valid JSON matching the tool schema. That would cut the fallback
rate dramatically without needing to retrain.

---

## What to talk about with Manifold

The mode collapse result from Run 1 is worth leading with. The model figured out that
calling final_answer immediately scores well on safety criteria (observed_evidence,
forbidden_evidence, and unsafe_mutation all pass) while failing all the correctness
checks. That is exactly the kind of reward hacking the benchmark is designed to catch,
and it actually caught it.

The sklearn vs LoRA comparison makes a clear point about data scale. At 28 examples,
the simpler model wins. That is not a failure of the approach, it is an honest
measurement of where the data situation is right now.

The benchmark setup is the real deliverable here. The environment resets cleanly, all
8 scoring criteria are tied to business-correct outcomes, reward hacking is caught by
the verifier, and the pipeline is reproducible from a single command. The fine-tuned
model being small is expected. The benchmark being trustworthy is what matters.
