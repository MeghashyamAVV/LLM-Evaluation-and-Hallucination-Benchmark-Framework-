from evaluator import run_evaluation, aggregate_results, save_results, STRATEGIES
from dataset import EVAL_DATASET
import json

STRATEGY_ORDER = [
    "Zero-shot (baseline)",
    "Few-shot examples",
    "RAG (context provided)",
    "Chain-of-thought",
    "RAG + structured reasoning"
]

def print_summary(summary: dict):
    print("\n" + "="*72)
    print("  FINAL RESULTS — 5 PROMPT STRATEGIES COMPARED")
    print("  Scoring: deterministic keyword-overlap (no LLM judge)")
    print("="*72)
    print(f"  {'Strategy':<32} {'Faithful':>9} {'Relevant':>9} {'Halluc↓':>9} {'Latency':>10}")
    print(f"  {'-'*68}")

    sorted_strats = sorted(
        [(k, v) for k, v in summary.items()],
        key=lambda x: x[1]["avg_hallucination"]
    )

    for name, m in sorted_strats:
        print(
            f"  {name:<32} "
            f"{m['avg_faithfulness']:>9.3f} "
            f"{m['avg_answer_relevancy']:>9.3f} "
            f"{m['avg_hallucination']:>9.3f} "
            f"{m['avg_latency_ms']:>9.0f}ms"
        )

    print("="*72)
    print("  ↓ lower hallucination = better | ↑ higher faithfulness = better")

    best  = sorted_strats[0]
    worst = sorted_strats[-1]
    zero_shot = next(((k,v) for k,v in summary.items() if "zero" in k.lower()), sorted_strats[-1])

    worst_h     = zero_shot[1]["avg_hallucination"]
    best_h      = best[1]["avg_hallucination"]
    reduction_pct = round((worst_h - best_h) / max(worst_h, 0.001) * 100)

    faith_improvement = round(
        (best[1]["avg_faithfulness"] - zero_shot[1]["avg_faithfulness"])
        / max(zero_shot[1]["avg_faithfulness"], 0.001) * 100
    )

    print(f"\n  Best strategy:   {best[0]}")
    print(f"  Baseline:        {zero_shot[0]}")
    print(f"\n  📊 Hallucination reduction ({best[0]} vs zero-shot): {reduction_pct}%")
    print(f"  📊 Faithfulness improvement ({best[0]} vs zero-shot): +{faith_improvement}%")
    print(f"\n  → Resume metric: {best[0]} reduced hallucinations")
    print(f"    by {reduction_pct}% vs zero-shot baseline across 20 Q&A samples.")
    print("="*72)

    return reduction_pct, faith_improvement, best[0]

def main():
    print("\n" + "="*72)
    print("  LLM EVALUATION & HALLUCINATION BENCHMARK")
    print(f"  {len(EVAL_DATASET)} samples × {len(STRATEGIES)} strategies = "
          f"{len(EVAL_DATASET)*len(STRATEGIES)} evaluations")
    print("="*72)
    print("  Scoring method: deterministic keyword-overlap (no LLM judge bias)")
    print("  Make sure ollama serve is running in another terminal\n")

    results = run_evaluation(EVAL_DATASET)
    save_results(results, "eval_results.json")

    summary = aggregate_results(results)
    reduction_pct, faith_pct, best = print_summary(summary)

    with open("eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  Files saved: eval_results.json, eval_summary.json")
    print("  Run dashboard:  streamlit run dashboard.py\n")


if __name__ == "__main__":
    main()