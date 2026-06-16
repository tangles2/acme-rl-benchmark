"""
Acme RL Finetuning Benchmark — main entry point.

Usage:
    # Full run (requires SYBIL_API_KEY, downloads Qwen2.5-0.5B on first run):
    SYBIL_API_KEY=sn4_... python run_benchmark.py

    # Skip LLM baseline:
    python run_benchmark.py --no-sybil

    # Skip LoRA (sklearn policy only):
    python run_benchmark.py --no-lora

    # Minimal (sklearn pipeline only, no API key, no model download):
    python run_benchmark.py --no-sybil --no-lora

Story:
    SybilAgent (frontier LLM, zero-shot)
        → PolicyAgent (sklearn SFT analogue, fast)
        → LoRAAgent (Qwen2.5-0.5B + LoRA fine-tuned)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.environment import MockEnvironment
from src.benchmark  import Benchmark
from src.train      import train


def failure_analysis(label_a, results_a, label_b, results_b):
    print("\n" + "=" * 60)
    print("  FAILURE ANALYSIS")
    print("=" * 60)
    for a, b in zip(results_a, results_b):
        task_id = a["task_id"]
        if not a["strict_pass"] or not b["strict_pass"]:
            print(f"\n  Task: {task_id}")
            if not a["strict_pass"]:
                failed = [k for k, v in a["criteria"].items() if not v]
                print(f"    {label_a:25s} failed: {', '.join(failed)}")
            if not b["strict_pass"]:
                failed = [k for k, v in b["criteria"].items() if not v]
                print(f"    {label_b:25s} failed: {', '.join(failed)}")
    print()


def delta_report(label_a, agg_a, label_b, agg_b):
    print("\n" + "=" * 60)
    print(f"  DELTA: {label_a}  →  {label_b}")
    print("=" * 60)

    def pct(x): return f"{x*100:.1f}%"

    rows = [
        ("Strict pass rate",  agg_a["strict_pass_rate"],    agg_b["strict_pass_rate"]),
        ("Avg score",         pct(agg_a["avg_score"]),       pct(agg_b["avg_score"])),
        ("Avg tool calls",    str(agg_a["avg_tool_calls"]),  str(agg_b["avg_tool_calls"])),
        ("Avg broad scans",   str(agg_a["avg_broad_scans"]), str(agg_b["avg_broad_scans"])),
    ]
    for label, av, bv in rows:
        print(f"  {label:22s}  before={av:>8}   after={bv:>8}")

    print(f"\n  Criteria accuracy:")
    print(f"  {'criterion':22s}  {'before':>12}   {'after':>12}")
    for k in agg_a["criteria_accuracy"]:
        av    = agg_a["criteria_accuracy"][k]
        bv    = agg_b["criteria_accuracy"][k]
        arrow = " <improved>" if bv > av else ("          " if bv == av else " <regressed>")
        print(f"  {k:22s}  {pct(av):>12}   {pct(bv):>12}{arrow}")
    print()


def run_sybil(bench):
    try:
        from src.sybil_agent import SybilAgent
        agent   = SybilAgent()
        print(f"\n[1/4] Running SybilAgent ({agent.MODEL}) — zero-shot frontier LLM ...")
        results = bench.run_agent(agent)
        agg     = bench.aggregate(results)
        bench.print_report("sybil_llm_baseline", results, agg)
        return results, agg, "sybil"
    except (ImportError, ValueError) as e:
        print(f"\n[1/4] SybilAgent skipped: {e}")
        print("       Set SYBIL_API_KEY to enable LLM baseline. Using rule-based fallback.\n")
        from src.agents import BaselineAgent
        agent   = BaselineAgent()
        results = bench.run_agent(agent)
        agg     = bench.aggregate(results)
        bench.print_report("rule_baseline (fallback)", results, agg)
        return results, agg, "baseline"


def run_lora(bench, verbose=True):
    try:
        from src.lora_train import lora_train
        if verbose:
            print("\n[4/4] LoRA fine-tuning Qwen2.5-0.5B-Instruct ...")
            print("      (first run downloads ~1 GB from HuggingFace)")
        agent   = lora_train(verbose=verbose)
        results = bench.run_agent(agent)
        agg     = bench.aggregate(results)
        bench.print_report("lora_finetuned (Qwen2.5-0.5B)", results, agg)
        return results, agg
    except Exception as e:
        print(f"\n[4/4] LoRA training failed: {e}")
        print("       Use --no-lora to skip, or check GPU/memory/HF access.\n")
        return None, None


def main():
    no_sybil = "--no-sybil" in sys.argv
    no_lora  = "--no-lora"  in sys.argv

    print("=" * 60)
    print("  Acme Finance Operations — RL Finetuning Benchmark")
    print("=" * 60)

    env   = MockEnvironment()
    bench = Benchmark(env)

    # ---- 1: LLM / rule-based baseline ----
    if no_sybil:
        from src.agents import BaselineAgent
        agent   = BaselineAgent()
        print("\n[1/4] Running rule-based baseline (--no-sybil) ...")
        baseline_results = bench.run_agent(agent)
        baseline_agg     = bench.aggregate(baseline_results)
        bench.print_report("rule_baseline", baseline_results, baseline_agg)
        baseline_label   = "baseline"
    else:
        baseline_results, baseline_agg, baseline_label = run_sybil(bench)

    # ---- 2: Train sklearn policy (fast SFT analogue) ----
    print("\n[2/4] Training sklearn next-action policy ...")
    policy = train(verbose=True)

    # ---- 3: Run sklearn policy ----
    print("\n[3/4] Running sklearn PolicyAgent ...")
    policy_results = bench.run_agent(policy)
    policy_agg     = bench.aggregate(policy_results)
    bench.print_report("policy_sklearn (post-SFT)", policy_results, policy_agg)

    # ---- Delta: baseline → sklearn ----
    delta_report(baseline_label, baseline_agg, "policy_sklearn", policy_agg)
    failure_analysis(baseline_label, baseline_results, "policy_sklearn", policy_results)

    # ---- 4: LoRA fine-tuning (optional) ----
    if not no_lora:
        lora_results, lora_agg = run_lora(bench)
        if lora_results:
            delta_report("policy_sklearn", policy_agg, "lora_finetuned", lora_agg)
            failure_analysis("policy_sklearn", policy_results, "lora_finetuned", lora_results)
    else:
        print("\n[4/4] LoRA step skipped (--no-lora).")

    print("\nDone.")
    print("  sklearn artifact : artifacts/next_action_policy.pkl")
    if not no_lora:
        print("  LoRA adapter     : artifacts/lora_adapter/")

    # ---- 5: Push model card to HuggingFace ----
    print("\n[5/5] Pushing model card to HuggingFace ...")
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


if __name__ == "__main__":
    main()
