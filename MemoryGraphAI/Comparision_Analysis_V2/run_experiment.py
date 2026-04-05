"""
run_experiment.py

FULL EXPERIMENT PIPELINE

Includes:
- Vector vs Hybrid comparison
- LLM-as-judge evaluation
- Advanced metrics (graph reasoning proof)
- Final summary + JSON output
"""

import json
import os

from pipelines.vector_rag import VectorRAG
from pipelines.hybrid_rag import HybridRAG
from evaluation.eval_queries import EVAL_QUERIES
from evaluation.llm_evaluator import judge
from evaluation.advanced_metrics import compute_advanced_metrics


# ─────────────────────────────────────────────────────────────
# LOAD GRAPH
# ─────────────────────────────────────────────────────────────

with open("data/knowledge_graph.json") as f:
    graph = json.load(f)

vector = VectorRAG(graph["nodes"])
hybrid = HybridRAG(graph)


# ─────────────────────────────────────────────────────────────
# TRACKING VARIABLES
# ─────────────────────────────────────────────────────────────

hybrid_wins = 0
vector_wins = 0
ties = 0

all_results = []


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

for idx, q in enumerate(EVAL_QUERIES, start=1):

    query = q["query"]
    gt = q["ground_truth"]

    print("\n" + "=" * 80)
    print(f"[Query {idx}/{len(EVAL_QUERIES)}]")
    print("Q:", query)
    print("-" * 80)

    # ─────────────────────────────
    # VECTOR RAG
    # ─────────────────────────────
    v_nodes = vector.retrieve(query)
    v_context = " ".join(v_nodes)
    v_ans = vector.answer(query, v_nodes)

    # ─────────────────────────────
    # HYBRID RAG
    # ─────────────────────────────
    h_ans = hybrid.answer(query)

    # Extract context from hybrid answer (simple parsing)
    h_context = h_ans

    # ─────────────────────────────
    # LLM JUDGE
    # ─────────────────────────────
    result = judge(query, h_ans, v_ans, gt)
    
    if result.get("error"):
        print("⚠️ Skipping (LLM output invalid)")
        continue
    
    winner = result.get("winner", "tie").lower()

    if winner == "hybrid":
        hybrid_wins += 1
    elif winner == "vector":
        vector_wins += 1
    else:
        ties += 1

    # ─────────────────────────────
    # ADVANCED METRICS
    # ─────────────────────────────
    adv_metrics = compute_advanced_metrics(
        query=query,
        hybrid_context=h_context,
        vector_context=v_context,
        hybrid_answer=h_ans,
        vector_answer=v_ans,
        ground_truth=gt
    )

    # ─────────────────────────────
    # PRINT OUTPUT
    # ─────────────────────────────
    print("\n--- Vector Answer ---")
    print(v_ans)

    print("\n--- Hybrid Answer ---")
    print(h_ans)

    print("\n--- LLM Evaluation ---")
    print(json.dumps(result, indent=2))

    print("\n--- Advanced Metrics ---")
    print(json.dumps(adv_metrics, indent=2))

    print(f"\n🏆 Winner: {winner.upper()}")

    # ─────────────────────────────
    # STORE RESULT
    # ─────────────────────────────
    all_results.append({
        "query": query,
        "ground_truth": gt,
        "vector_answer": v_ans,
        "hybrid_answer": h_ans,
        "llm_evaluation": result,
        "advanced_metrics": adv_metrics,
        "winner": winner
    })


# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(f"\nHybrid wins : {hybrid_wins}")
print(f"Vector wins : {vector_wins}")
print(f"Ties        : {ties}")

if hybrid_wins > vector_wins:
    print("\n🔥 FINAL WINNER: HYBRID RAG (BETTER REASONING)")
elif vector_wins > hybrid_wins:
    print("\n⚠️ VECTOR RAG STILL STRONG")
else:
    print("\n🔵 RESULT: TIE")


# ─────────────────────────────────────────────────────────────
# AGGREGATE METRICS
# ─────────────────────────────────────────────────────────────

def avg(lst):
    return sum(lst) / len(lst) if lst else 0

relation_gains = []
richness_gains = []
grounding_gains = []
multihop_gains = []

for r in all_results:
    m = r["advanced_metrics"]

    relation_gains.append(m["relationship_metrics"]["relation_gain"])
    richness_gains.append(m["context_metrics"]["richness_gain"])
    grounding_gains.append(m["grounding_metrics"]["grounding_gain"])
    multihop_gains.append(m["multihop_reasoning"]["gain"])

print("\n📊 AVERAGE ADVANCED METRICS:")
print("-" * 50)
print(f"Relation Gain   : {avg(relation_gains):.2f}")
print(f"Richness Gain   : {avg(richness_gains):.2f}")
print(f"Grounding Gain  : {avg(grounding_gains):.2f}")
print(f"Multi-hop Gain  : {avg(multihop_gains):.2f}")


# ─────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────

output_path = os.path.join(os.getcwd(), "experiment_results.json")

with open(output_path, "w") as f:
    json.dump({
        "summary": {
            "hybrid_wins": hybrid_wins,
            "vector_wins": vector_wins,
            "ties": ties,
            "avg_relation_gain": avg(relation_gains),
            "avg_richness_gain": avg(richness_gains),
            "avg_grounding_gain": avg(grounding_gains),
            "avg_multihop_gain": avg(multihop_gains)
        },
        "details": all_results
    }, f, indent=2)

print(f"\nResults saved to → {output_path}")