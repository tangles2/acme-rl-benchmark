"""
V2 benchmark runner.

Extends the V1 pipeline with:
  - Synthetic trace generation (8 new tasks, ~91 total training examples)
  - Multi-model LoRA training (Qwen2.5-0.5B and Qwen2.5-1.5B by default)
  - Hidden eval split (task_missing_evidence held out from training)
  - DPO refinement step on the best-scoring model (optional)
  - Auto-push of all adapters to HuggingFace

V1 code is untouched. Use run_benchmark.py for the V1 pipeline.

Usage:
    python3 run_benchmark_v2.py
    python3 run_benchmark_v2.py --skip-dpo
    python3 run_benchmark_v2.py --skip-synthetic
    python3 run_benchmark_v2.py
    python3 run_benchmark_v2.py --no-sybil --skip-dpo
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Acme RL Benchmark V2")
parser.add_argument("--no-sybil",        action="store_true", help="Skip Sybil LLM baseline")
parser.add_argument("--skip-synthetic",  action="store_true", help="Skip synthetic trace generation (use V1 traces only)")
parser.add_argument("--skip-dpo",        action="store_true", help="Skip DPO refinement step")
parser.add_argument("--skip-lora",       action="store_true", help="Skip all LoRA training (runs baselines only)")
parser.add_argument("--push-model-card", action="store_true", help="Push model card to HuggingFace after run (default: off)")
args = parser.parse_args()

print("=" * 60)
print("  Acme Finance Operations -- RL Finetuning Benchmark V2")
print("=" * 60)

# ---------------------------------------------------------------------------
# Step 1: Rule-based baseline (same as V1, for comparison)
# ---------------------------------------------------------------------------

print("\n[1/6] Running rule-based baseline ...")

from src.environment import MockEnvironment
from src.agents import BaselineAgent
from src.benchmark import Benchmark

env_v1       = MockEnvironment()
bench_v1     = Benchmark(env_v1)
baseline_ag  = BaselineAgent()
baseline_ag.name = "rule_baseline"

baseline_results = bench_v1.run_agent(baseline_ag)
baseline_agg     = Benchmark.aggregate(baseline_results)
Benchmark.print_report("rule_baseline", baseline_results, baseline_agg)

# ---------------------------------------------------------------------------
# Step 1b: Sybil LLM baseline (zero-shot frontier model)
# ---------------------------------------------------------------------------

if args.no_sybil:
    print("\n[1b] Skipping Sybil baseline (--no-sybil)")
    sybil_results = None
    sybil_agg = None
else:
    print("\n[1b] Running Sybil LLM baseline ...")
    try:
        from src.sybil_agent import SybilAgent
        sybil_agent  = SybilAgent()
        sybil_results = bench_v1.run_agent(sybil_agent)
        sybil_agg     = Benchmark.aggregate(sybil_results)
        Benchmark.print_report("sybil_llm_baseline", sybil_results, sybil_agg)
    except Exception as e:
        print(f"[1b] Sybil baseline failed (non-fatal): {e}")
        sybil_results = None
        sybil_agg = None


# ---------------------------------------------------------------------------
# Step 2: sklearn SFT on original traces (V1 policy, for comparison)
# ---------------------------------------------------------------------------

print("\n[3/7] Training sklearn policy on original traces ...")

from src.train import train as train_sklearn

sklearn_agent = train_sklearn(verbose=True)
sklearn_results = bench_v1.run_agent(sklearn_agent)
sklearn_agg     = Benchmark.aggregate(sklearn_results)
Benchmark.print_report("policy_sklearn_v1", sklearn_results, sklearn_agg)

# ---------------------------------------------------------------------------
# Step 3: Build combined training dataset
# ---------------------------------------------------------------------------

if args.skip_synthetic:
    print("\n[4/7] Skipping synthetic generation -- using original traces only ...")
    import json
    FIXTURES = Path(__file__).parent / "fixtures" / "rl-finetuning"
    combined_traces = []
    with open(FIXTURES / "training_traces.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                combined_traces.append(json.loads(line))
    print(f"[4/7] Using {len(combined_traces)} original traces")
else:
    print("\n[4/7] Generating synthetic traces ...")
    from src.v2.synthetic import build_combined_traces
    combined_traces = build_combined_traces()

# ---------------------------------------------------------------------------
# Step 4: Train LoRA models
# ---------------------------------------------------------------------------

if args.skip_lora:
    print("\n[5/7] Skipping LoRA training (--skip-lora)")
    trained_models = []
else:
    print("\n[5/7] Training LoRA models ...")
    from src.v2.multi_lora import train_all_models, MODEL_CONFIGS
    trained_models = train_all_models(combined_traces, configs=MODEL_CONFIGS, verbose=True)

# ---------------------------------------------------------------------------
# Step 5: Benchmark all LoRA models with hidden eval split
# ---------------------------------------------------------------------------

from src.v2.synthetic import SyntheticMockEnvironment, V2Benchmark, EVAL_TASK_IDS
from src.v2.multi_lora import V2LoRAAgent

env_v2   = SyntheticMockEnvironment()
bench_v2 = V2Benchmark(env_v2)

# Also run V1 sklearn on the expanded task set for apples-to-apples comparison
print("\n[6/7] Running sklearn on expanded task set (train + eval split) ...")
sklearn_train, sklearn_eval = bench_v2.run_split(sklearn_agent)
sklearn_train_agg = V2Benchmark.aggregate(sklearn_train)
sklearn_eval_agg  = V2Benchmark.aggregate(sklearn_eval)
print(f"\n  sklearn -- train tasks: {sklearn_train_agg['strict_pass_rate']}  eval tasks: {sklearn_eval_agg['strict_pass_rate']}")

lora_summaries = []

for config, adapter_path, hf_url in trained_models:
    slug  = config["slug"]
    name  = config["name"]
    print(f"\n[6/7] Benchmarking {slug} ...")
    try:
        agent = V2LoRAAgent(adapter_path=str(adapter_path), base_model=name)
        train_results, eval_results = bench_v2.run_split(agent)
        train_agg = V2Benchmark.aggregate(train_results)
        eval_agg  = V2Benchmark.aggregate(eval_results)
        V2Benchmark.print_report(f"lora_v2_{slug} (train tasks)", train_results, train_agg)
        V2Benchmark.print_report(f"lora_v2_{slug} (eval tasks)",  eval_results,  eval_agg)
        lora_summaries.append({
            "slug":       slug,
            "hf_url":     hf_url,
            "train_agg":  train_agg,
            "eval_agg":   eval_agg,
        })
    except Exception as e:
        print(f"[6/7] Benchmarking failed for {slug}: {e}")

# ---------------------------------------------------------------------------
# Step 6: DPO refinement on the best-scoring model
# ---------------------------------------------------------------------------

if args.skip_dpo or not trained_models:
    print("\n[7/7] Skipping DPO step")
    dpo_url = None
else:
    print("\n[7/7] Running DPO refinement on best model ...")
    # Pick the model with the best train-task strict pass rate
    if lora_summaries:
        best = max(lora_summaries, key=lambda x: x["train_agg"]["strict_pass_pct"])
        best_config   = next(c for c, _, _ in trained_models if c["slug"] == best["slug"])
        best_adapter  = next(p for c, p, _ in trained_models if c["slug"] == best["slug"])
        print(f"[7/7] Best model: {best['slug']} (train pass rate {best['train_agg']['strict_pass_rate']})")
        try:
            from src.v2.dpo_train import dpo_train
            dpo_path, dpo_url = dpo_train(
                adapter_path    = best_adapter,
                base_model_name = best_config["name"],
                traces          = combined_traces,
            )
            if dpo_path:
                dpo_agent = V2LoRAAgent(adapter_path=str(dpo_path), base_model=best_config["name"])
                dpo_train_r, dpo_eval_r = bench_v2.run_split(dpo_agent)
                dpo_train_agg = V2Benchmark.aggregate(dpo_train_r)
                dpo_eval_agg  = V2Benchmark.aggregate(dpo_eval_r)
                V2Benchmark.print_report(f"dpo_{best['slug']} (train tasks)", dpo_train_r, dpo_train_agg)
                V2Benchmark.print_report(f"dpo_{best['slug']} (eval tasks)",  dpo_eval_r,  dpo_eval_agg)
        except Exception as e:
            print(f"[7/7] DPO failed: {e}")
            dpo_url = None
    else:
        print("[7/7] No models trained successfully, skipping DPO")
        dpo_url = None

# ---------------------------------------------------------------------------
# Final summary table
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  V2 SUMMARY")
print("=" * 60)
print(f"\n  Training examples : {sum(len(t['messages']) for t in combined_traces)}")
print(f"  Eval task held out: {sorted(EVAL_TASK_IDS)}")
print()
print(f"  {'Agent':<28} {'Train pass':>12} {'Eval pass':>12} {'Avg score':>10}")
print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*10}")
print(f"  {'rule_baseline':<28} {baseline_agg['strict_pass_rate']:>12} {'N/A':>12} {baseline_agg['avg_score']:>10}")
print(f"  {'sklearn_v1':<28} {sklearn_agg['strict_pass_rate']:>12} {'N/A':>12} {sklearn_agg['avg_score']:>10}")
if sybil_agg:
    print(f"  {'sybil_llm_baseline':<28} {sybil_agg['strict_pass_rate']:>12} {'N/A':>12} {sybil_agg['avg_score']:>10}")
print(f"  {'sklearn (expanded)':<28} {sklearn_train_agg['strict_pass_rate']:>12} {sklearn_eval_agg['strict_pass_rate']:>12} {sklearn_train_agg['avg_score']:>10}")
for s in lora_summaries:
    print(f"  {('lora_v2_' + s['slug']):<28} {s['train_agg']['strict_pass_rate']:>12} {s['eval_agg']['strict_pass_rate']:>12} {s['train_agg']['avg_score']:>10}")

print()
print("  HuggingFace adapters:")
for s in lora_summaries:
    if s["hf_url"]:
        print(f"    {s['slug']:20s}  {s['hf_url']}")
if dpo_url:
    print(f"    {'dpo':20s}  {dpo_url}")
print()

# ---------------------------------------------------------------------------
# Push model card to HuggingFace (opt-in via --push-model-card)
# ---------------------------------------------------------------------------

if args.push_model_card:
    print("Pushing model card to HuggingFace ...")
    try:
        import subprocess
        result = subprocess.run(
            ["python3", "push_model_card.py"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  {result.stdout.strip()}")
        else:
            print(f"  Model card push failed (non-fatal): {result.stderr.strip()}")
    except Exception as e:
        print(f"  Model card push skipped: {e}")
else:
    print("[info] Model card push skipped. Pass --push-model-card to enable.")
