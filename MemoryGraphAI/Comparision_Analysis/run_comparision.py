"""
run_comparison.py
-----------------
Orchestrates the head-to-head evaluation between:
  - MemoryGraph RAG  (Graph-based retrieval using Neo4j + relationship traversal)
  - Vector RAG       (Pure cosine-similarity retrieval over stored entity embeddings)

Usage
-----
    python run_comparison.py

Output
------
  - Prints a per-query breakdown table to the terminal.
  - Saves full results to `comparison_results.json`.
  - Saves a summary CSV  to `comparison_summary.csv`.

Evaluation Test Set
-------------------
Edit the EVAL_QUERIES list below to add your own questions, ground-truth
relevant entities, and reference answers that are appropriate for YOUR documents.

Each entry is a dict with:
  query           : the natural-language question
  relevant_entities : entities that a correct retriever MUST surface
                      (case-insensitive, title-cased internally)
  ground_truth    : a reference answer used for semantic similarity scoring
"""

import json
import time
import csv
import os
import sys

# ── make sure parent directory modules are importable ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from query_engine import GraphQueryEngine
from vector_pipeline import VectorRAGPipeline
from evaluation_metrics import compute_all_metrics

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION TEST SET  ← customise this for your domain / documents
# ─────────────────────────────────────────────────────────────────────────────

EVAL_QUERIES = [
    {
        "query": "What methods are used for node classification?",
        "relevant_entities": {
            "Graph Neural Networks", "Node Classification",
            "Gcn", "Graph Convolutional Network"
        },
        "ground_truth": (
            "Graph Neural Networks (GNNs), including GCN and GraphSAGE, "
            "are the primary methods used for node classification tasks."
        ),
    },
    {
        "query": "Which datasets are used to evaluate graph learning models?",
        "relevant_entities": {
            "Cora Dataset", "Citeseer", "Pubmed", "Ogbn-Arxiv"
        },
        "ground_truth": (
            "Common benchmark datasets include Cora, Citeseer, PubMed, "
            "and OGB datasets like ogbn-arxiv."
        ),
    },
    {
        "query": "What are the key evaluation metrics for graph models?",
        "relevant_entities": {
            "Accuracy", "F1 Score", "Auc-Roc", "Mean Reciprocal Rank"
        },
        "ground_truth": (
            "Accuracy, F1 score, AUC-ROC, and Mean Reciprocal Rank are "
            "common metrics used to evaluate graph learning models."
        ),
    },
    {
        "query": "How is attention mechanism applied in graph networks?",
        "relevant_entities": {
            "Graph Attention Network", "Gat", "Attention Mechanism",
            "Self-Attention"
        },
        "ground_truth": (
            "Graph Attention Networks (GAT) apply self-attention to assign "
            "learnable weights to neighbouring nodes during aggregation."
        ),
    },
    {
        "query": "What techniques are used for link prediction?",
        "relevant_entities": {
            "Link Prediction", "Knowledge Graph Embedding",
            "TransE", "Graph Autoencoder"
        },
        "ground_truth": (
            "Link prediction uses knowledge graph embeddings such as TransE, "
            "and structural methods like Graph Autoencoders."
        ),
    },
]

K = 5   # Retrieval cut-off rank


# ─────────────────────────────────────────────────────────────────────────────
# Graph retrieval helper: returns a ranked list of entity names
# ─────────────────────────────────────────────────────────────────────────────

def graph_retrieved_entities(engine: GraphQueryEngine, query: str, top_k: int) -> list[str]:
    """
    Re-uses the vector index inside Neo4j to find the top-k seed nodes,
    then expands each seed one hop to include all direct neighbours.
    The returned list is ordered: seeds first, then neighbours (deduped).
    """
    q_emb = engine.model.encode(query).tolist()
    seed_query = """
    CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
    YIELD node, score
    RETURN node.name AS name
    """
    neighbour_query = """
    CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
    YIELD node, score
    MATCH (node)-[r]-(neighbour)
    RETURN DISTINCT neighbour.name AS name
    LIMIT 20
    """
    seeds, neighbours = [], []
    with engine.driver.session() as session:
        for rec in session.run(seed_query, k=top_k, embedding=q_emb):
            seeds.append(rec["name"])
        for rec in session.run(neighbour_query, k=top_k, embedding=q_emb):
            neighbours.append(rec["name"])

    # Deduplicate while preserving order (seeds have priority)
    seen = set()
    ordered = []
    for name in seeds + neighbours:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison loop
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison():
    print("\n" + "=" * 70)
    print("  MemoryGraph RAG  vs  Vector RAG  — Retrieval Evaluation")
    print("=" * 70)

    graph_engine = GraphQueryEngine()
    vector_engine = VectorRAGPipeline(top_k=K)

    all_results = []

    for idx, item in enumerate(EVAL_QUERIES, start=1):
        query = item["query"]
        relevant = {e.title() for e in item["relevant_entities"]}
        ground_truth = item["ground_truth"]

        print(f"\n[Query {idx}/{len(EVAL_QUERIES)}] {query}")
        print("-" * 60)

        # ── Graph RAG ──────────────────────────────────────────────────────
        t0 = time.time()
        graph_retrieved = graph_retrieved_entities(graph_engine, query, top_k=K)
        graph_context = graph_engine.search_graph(query, top_k=K)
        graph_answer = graph_engine.answer_question(query)
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

        # ── Vector RAG ─────────────────────────────────────────────────────
        t1 = time.time()
        vector_retrieved, _ = vector_engine.search(query)
        vector_context = vector_engine.get_context_string(query)
        vector_answer = vector_engine.answer_question(query)
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

        # ── Pretty print ───────────────────────────────────────────────────
        print(f"  {'Metric':<25} {'Graph RAG':>12} {'Vector RAG':>12}  {'Winner':>10}")
        print(f"  {'-'*25} {'-'*12} {'-'*12}  {'-'*10}")
        for metric in graph_metrics:
            gv = graph_metrics[metric]
            vv = vector_metrics[metric]
            # For latency, lower is better; for all others, higher is better
            if metric == "Latency(s)":
                winner = "Graph ✓" if gv <= vv else "Vector ✓"
            else:
                winner = "Graph ✓" if gv >= vv else "Vector ✓"
                if abs(gv - vv) < 0.001:
                    winner = "Tie"
            print(f"  {metric:<25} {gv:>12.4f} {vv:>12.4f}  {winner:>10}")

        all_results.append({
            "query": query,
            "graph_retrieved": graph_retrieved,
            "vector_retrieved": vector_retrieved,
            "graph_answer": graph_answer,
            "vector_answer": vector_answer,
            "graph_metrics": graph_metrics,
            "vector_metrics": vector_metrics,
        })

    # ── Macro-averages ─────────────────────────────────────────────────────
    metric_keys = list(all_results[0]["graph_metrics"].keys())
    graph_avgs, vector_avgs = {}, {}
    for k_name in metric_keys:
        graph_avgs[k_name] = sum(r["graph_metrics"][k_name] for r in all_results) / len(all_results)
        vector_avgs[k_name] = sum(r["vector_metrics"][k_name] for r in all_results) / len(all_results)

    print("\n" + "=" * 70)
    print("  MACRO-AVERAGED RESULTS (across all queries)")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Graph RAG':>12} {'Vector RAG':>12}  {'Winner':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}  {'-'*10}")
    for metric in metric_keys:
        gv = graph_avgs[metric]
        vv = vector_avgs[metric]
        if metric == "Latency(s)":
            winner = "Graph ✓" if gv <= vv else "Vector ✓"
        else:
            winner = "Graph ✓" if gv >= vv else "Vector ✓"
            if abs(gv - vv) < 0.001:
                winner = "Tie"
        print(f"  {metric:<25} {gv:>12.4f} {vv:>12.4f}  {winner:>10}")

    # ── Save outputs ───────────────────────────────────────────────────────
    output_dir = os.path.dirname(os.path.abspath(__file__))

    json_path = os.path.join(output_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "per_query_results": all_results,
                "macro_averages": {
                    "graph_rag": graph_avgs,
                    "vector_rag": vector_avgs,
                },
            },
            f,
            indent=2,
        )
    print(f"\n  Full results saved → {json_path}")

    csv_path = os.path.join(output_dir, "comparison_summary.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Graph_RAG_Avg", "Vector_RAG_Avg", "Winner"])
        for metric in metric_keys:
            gv = graph_avgs[metric]
            vv = vector_avgs[metric]
            if metric == "Latency(s)":
                winner = "Graph RAG" if gv <= vv else "Vector RAG"
            else:
                winner = "Graph RAG" if gv >= vv else "Vector RAG"
                if abs(gv - vv) < 0.001:
                    winner = "Tie"
            writer.writerow([metric, round(gv, 4), round(vv, 4), winner])
    print(f"  Summary CSV saved  → {csv_path}")

    graph_engine.close()
    vector_engine.close()
    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    run_comparison()