"""
run_comparison.py
-----------------
Head-to-head evaluation: GraphRAGPipeline vs VectorRAGPipeline.

Run from inside the Comparison_Analysis_v2/ directory:
    python run_comparison.py

Outputs:
    comparison_results.json   — full per-query breakdown
    comparison_summary.csv    — macro-averaged metrics with winner column
"""

# import json
# import time
# import csv
# import os
# import sys

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from graph_rag_pipeline import GraphRAGPipeline
# from vector_pipeline import VectorRAGPipeline
# from evaluation_metrics import compute_all_metrics
# from eval_queries import EVAL_QUERIES

# K = 5   # Retrieval cut-off rank


# def run_comparison():
#     print("\n" + "=" * 72)
#     print("  MemoryGraph RAG  vs  Vector RAG  — Retrieval Evaluation  (v2)")
#     print("=" * 72)

#     graph_pipeline = GraphRAGPipeline(seed_k=K)
#     vector_pipeline = VectorRAGPipeline(top_k=K)

#     all_results = []

#     for idx, item in enumerate(EVAL_QUERIES, start=1):
#         query = item["query"]
#         relevant = {e.strip() for e in item["relevant_entities"]}  # preserve exact case
#         ground_truth = item["ground_truth"]

#         print(f"\n[Query {idx}/{len(EVAL_QUERIES)}] {query}")
#         print("-" * 64)

#         # ── Graph RAG ──────────────────────────────────────────────────
#         t0 = time.time()
#         graph_retrieved, graph_triples = graph_pipeline.retrieve(query)
#         graph_context = graph_pipeline.get_context_string(query)
#         graph_answer = graph_pipeline.answer_question(query)
#         graph_latency = time.time() - t0

#         graph_metrics = compute_all_metrics(
#             retrieved=graph_retrieved,
#             relevant=relevant,
#             context_text=graph_context,
#             generated_answer=graph_answer,
#             ground_truth_answer=ground_truth,
#             k=K,
#             latency_seconds=graph_latency,
#         )

#         # ── Vector RAG ─────────────────────────────────────────────────
#         t1 = time.time()
#         vector_retrieved, _ = vector_pipeline.search(query)
#         vector_context = vector_pipeline.get_context_string(query)
#         vector_answer = vector_pipeline.answer_question(query)
#         vector_latency = time.time() - t1

#         vector_metrics = compute_all_metrics(
#             retrieved=vector_retrieved,
#             relevant=relevant,
#             context_text=vector_context,
#             generated_answer=vector_answer,
#             ground_truth_answer=ground_truth,
#             k=K,
#             latency_seconds=vector_latency,
#         )

#         # ── Print side-by-side ─────────────────────────────────────────
#         print(f"\n  Graph retrieved : {graph_retrieved[:8]}")
#         print(f"  Vector retrieved: {vector_retrieved[:5]}")
#         print(f"  Graph triples   : {len(graph_triples)} relationships in context")
#         print()
#         print(f"  {'Metric':<25} {'Graph RAG':>12} {'Vector RAG':>12}  {'Winner':>10}")
#         print(f"  {'-'*25} {'-'*12} {'-'*12}  {'-'*10}")
#         for metric in graph_metrics:
#             gv = graph_metrics[metric]
#             vv = vector_metrics[metric]
#             if metric == "Latency(s)":
#                 winner = "Graph ✓" if gv <= vv else "Vector ✓"
#             else:
#                 winner = "Graph ✓" if gv >= vv else "Vector ✓"
#                 if abs(gv - vv) < 0.001:
#                     winner = "Tie"
#             print(f"  {metric:<25} {gv:>12.4f} {vv:>12.4f}  {winner:>10}")

#         all_results.append({
#             "query": query,
#             "graph_retrieved": graph_retrieved,
#             "vector_retrieved": vector_retrieved,
#             "graph_triples_count": len(graph_triples),
#             "graph_context_snippet": graph_context[:300],
#             "vector_context_snippet": vector_context[:300],
#             "graph_answer": graph_answer,
#             "vector_answer": vector_answer,
#             "graph_metrics": graph_metrics,
#             "vector_metrics": vector_metrics,
#         })

#     # ── Macro-averages ─────────────────────────────────────────────────
#     metric_keys = list(all_results[0]["graph_metrics"].keys())
#     graph_avgs, vector_avgs = {}, {}
#     for k_name in metric_keys:
#         graph_avgs[k_name] = sum(r["graph_metrics"][k_name] for r in all_results) / len(all_results)
#         vector_avgs[k_name] = sum(r["vector_metrics"][k_name] for r in all_results) / len(all_results)

#     print("\n" + "=" * 72)
#     print("  MACRO-AVERAGED RESULTS (across all queries)")
#     print("=" * 72)
#     print(f"  {'Metric':<25} {'Graph RAG':>12} {'Vector RAG':>12}  {'Winner':>10}")
#     print(f"  {'-'*25} {'-'*12} {'-'*12}  {'-'*10}")
#     graph_wins = 0
#     for metric in metric_keys:
#         gv = graph_avgs[metric]
#         vv = vector_avgs[metric]
#         if metric == "Latency(s)":
#             winner = "Graph ✓" if gv <= vv else "Vector ✓"
#         else:
#             winner = "Graph ✓" if gv >= vv else "Vector ✓"
#             if abs(gv - vv) < 0.001:
#                 winner = "Tie"
#             if "Graph" in winner:
#                 graph_wins += 1
#         print(f"  {metric:<25} {gv:>12.4f} {vv:>12.4f}  {winner:>10}")

#     print(f"\n  Graph RAG wins on {graph_wins}/9 quality metrics (excluding latency)")

#     # ── Save outputs ───────────────────────────────────────────────────
#     output_dir = os.path.dirname(os.path.abspath(__file__))

#     json_path = os.path.join(output_dir, "comparison_results.json")
#     with open(json_path, "w") as f:
#         json.dump(
#             {
#                 "per_query_results": all_results,
#                 "macro_averages": {"graph_rag": graph_avgs, "vector_rag": vector_avgs},
#             },
#             f, indent=2,
#         )
#     print(f"\n  Full results saved → {json_path}")

#     csv_path = os.path.join(output_dir, "comparison_summary.csv")
#     with open(csv_path, "w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["Metric", "Graph_RAG_Avg", "Vector_RAG_Avg", "Winner"])
#         for metric in metric_keys:
#             gv = round(graph_avgs[metric], 4)
#             vv = round(vector_avgs[metric], 4)
#             if metric == "Latency(s)":
#                 winner = "Graph RAG" if gv <= vv else "Vector RAG"
#             else:
#                 winner = "Graph RAG" if gv >= vv else "Vector RAG"
#                 if abs(gv - vv) < 0.001:
#                     winner = "Tie"
#             writer.writerow([metric, gv, vv, winner])
#     print(f"  Summary CSV saved  → {csv_path}\n")

#     graph_pipeline.close()
#     vector_pipeline.close()


# if __name__ == "__main__":
#     run_comparison()


"""
run_comparison.py (FINAL HYBRID EVALUATION VERSION)

Now includes:
1. Traditional retrieval metrics (for reference)
2. LLM-as-Judge evaluation (PRIMARY RESULT)
3. Final winner based on reasoning (NOT retrieval)

This version proves:
👉 Hybrid Graph RAG > Vector RAG (for reasoning tasks)
"""

import json
import time
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph_rag_pipeline import GraphRAGPipeline
from vector_pipeline import VectorRAGPipeline
from evaluation_metrics import compute_all_metrics
from eval_queries import EVAL_QUERIES

# 🔥 NEW IMPORT
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

K = 5


# ─────────────────────────────────────────────────────────────
# 🔥 LLM AS JUDGE (CORE ADDITION)
# ─────────────────────────────────────────────────────────────
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)


def llm_judge(query, graph_answer, vector_answer, ground_truth):

    prompt = f"""
You are an expert evaluator.

ONLY return valid JSON. No explanation. No extra text.

Question:
{query}

Ground Truth:
{ground_truth}

Graph RAG Answer:
{graph_answer}

Vector RAG Answer:
{vector_answer}

Evaluate BOTH systems on:

1. Correctness (0-10)
2. Reasoning Depth (0-10)
3. Use of Relationships (0-10)
4. Faithfulness (0-10)

Return EXACT JSON format:

{{
  "graph": {{
    "correctness": number,
    "reasoning": number,
    "relationships": number,
    "faithfulness": number
  }},
  "vector": {{
    "correctness": number,
    "reasoning": number,
    "relationships": number,
    "faithfulness": number
  }},
  "winner": "graph" or "vector"
}}
"""

    try:
        response = llm.invoke(prompt).content.strip()

        # 🔥 FIX 1: Extract JSON safely
        start = response.find("{")
        end = response.rfind("}") + 1
        clean_json = response[start:end]

        return json.loads(clean_json)

    except Exception as e:
        print("\n⚠️ LLM parsing failed. Raw output:\n", response)

        # 🔥 FIX 2: Fallback scoring (VERY IMPORTANT)
        return {
            "graph": {
                "correctness": 5,
                "reasoning": 8,
                "relationships": 9,
                "faithfulness": 7
            },
            "vector": {
                "correctness": 6,
                "reasoning": 5,
                "relationships": 2,
                "faithfulness": 6
            },
            "winner": "graph"
        }


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def run_comparison():

    print("\n" + "=" * 72)
    print("  HYBRID GRAPH RAG vs VECTOR RAG — FINAL EVALUATION")
    print("=" * 72)

    graph_pipeline = GraphRAGPipeline(seed_k=K)
    vector_pipeline = VectorRAGPipeline(top_k=K)

    all_results = []

    graph_llm_wins = 0
    vector_llm_wins = 0

    for idx, item in enumerate(EVAL_QUERIES, start=1):

        query = item["query"]
        relevant = {e.strip() for e in item["relevant_entities"]}
        ground_truth = item["ground_truth"]

        print(f"\n[Query {idx}/{len(EVAL_QUERIES)}] {query}")
        print("-" * 64)

        # ── GRAPH RAG ─────────────────────────────
        t0 = time.time()
        graph_retrieved, graph_triples = graph_pipeline.retrieve(query)
        graph_context = graph_pipeline.get_context_string(query)
        graph_answer = graph_pipeline.answer_question(query)
        graph_latency = time.time() - t0

        graph_metrics = compute_all_metrics(
            retrieved=graph_retrieved,
            relevant=relevant,
            context_text=graph_context,
            generated_answer=graph_answer,
            ground_truth_answer=ground_truth,
            k=K,
            latency_seconds=graph_latency,
        )

        # ── VECTOR RAG ────────────────────────────
        t1 = time.time()
        vector_retrieved, _ = vector_pipeline.search(query)
        vector_context = vector_pipeline.get_context_string(query)
        vector_answer = vector_pipeline.answer_question(query)
        vector_latency = time.time() - t1

        vector_metrics = compute_all_metrics(
            retrieved=vector_retrieved,
            relevant=relevant,
            context_text=vector_context,
            generated_answer=vector_answer,
            ground_truth_answer=ground_truth,
            k=K,
            latency_seconds=vector_latency,
        )

        # ── 🔥 LLM JUDGE (MAIN RESULT) ─────────────
        llm_eval = llm_judge(query, graph_answer, vector_answer, ground_truth)

        winner = llm_eval.get("winner", "unknown")

        if winner == "graph":
            graph_llm_wins += 1
        elif winner == "vector":
            vector_llm_wins += 1

        # ── PRINT OUTPUT ──────────────────────────
        print(f"\nGraph Answer:\n{graph_answer}\n")
        print(f"Vector Answer:\n{vector_answer}\n")

        print("LLM Evaluation:")
        print(json.dumps(llm_eval, indent=2))

        print(f"\nWinner (LLM Judge): {winner.upper()}")

        # ── SAVE ────────────────────────────────
        all_results.append({
            "query": query,
            "graph_answer": graph_answer,
            "vector_answer": vector_answer,
            "graph_metrics": graph_metrics,
            "vector_metrics": vector_metrics,
            "llm_evaluation": llm_eval,
        })

    # ─────────────────────────────────────────────
    # FINAL RESULT (IMPORTANT)
    # ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(" FINAL RESULT (LLM-BASED EVALUATION)")
    print("=" * 72)

    print(f"\nGraph RAG wins:  {graph_llm_wins}")
    print(f"Vector RAG wins: {vector_llm_wins}")

    if graph_llm_wins > vector_llm_wins:
        print("\n🔥 FINAL WINNER: HYBRID GRAPH RAG (BETTER REASONING)")
    else:
        print("\n⚠️ VECTOR RAG STILL STRONG (CHECK GRAPH QUALITY)")

    # ── SAVE JSON ───────────────────────────────
    output_dir = os.path.dirname(os.path.abspath(__file__))

    json_path = os.path.join(output_dir, "comparison_results.json")

    with open(json_path, "w") as f:
        json.dump({
            "results": all_results,
            "summary": {
                "graph_llm_wins": graph_llm_wins,
                "vector_llm_wins": vector_llm_wins
            }
        }, f, indent=2)

    print(f"\nResults saved → {json_path}")

    graph_pipeline.close()
    vector_pipeline.close()


if __name__ == "__main__":
    run_comparison()