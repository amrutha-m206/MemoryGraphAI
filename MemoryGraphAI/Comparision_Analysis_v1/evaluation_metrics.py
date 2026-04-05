"""
evaluation_metrics.py
---------------------
Evaluation metrics for Graph RAG vs Vector RAG comparison.

Metrics
-------
1.  Precision@K         — fraction of retrieved items that are relevant
2.  Recall@K            — fraction of relevant items that were retrieved
3.  F1@K                — harmonic mean of Precision and Recall
4.  Mean Reciprocal Rank (MRR)
5.  Normalized Discounted Cumulative Gain (nDCG@K)
6.  Average Precision (AP) / Mean Average Precision (MAP)
7.  Hit Rate@K (HR@K)
8.  Context Relevance Score (CRS) — semantic similarity between context and ground truth
9.  Answer Faithfulness  — semantic similarity between generated answer and ground truth
10. Query Latency        — wall-clock time in seconds
"""

import math
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# Ranking metrics
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for item in top_k if item in relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for item in top_k if item in relevant) / len(relevant)


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    num_hits, score = 0, 0.0
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            num_hits += 1
            score += num_hits / i
    return score / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    def dcg(items):
        return sum(
            1.0 / math.log2(i + 2)
            for i, item in enumerate(items)
            if item in relevant
        )
    actual_dcg = dcg(retrieved[:k])
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return actual_dcg / ideal_dcg if ideal_dcg else 0.0


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Semantic metrics
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0


def context_relevance_score(context_text: str, ground_truth_answer: str) -> float:
    """
    Semantic similarity between the retrieved context and the reference answer.
    Graph RAG context contains relational triples — these carry more
    topically-specific signal than a flat list of entity names,
    so this score should be noticeably higher for Graph RAG.
    """
    model = _get_model()
    return _cosine_sim(
        model.encode(context_text, convert_to_numpy=True),
        model.encode(ground_truth_answer, convert_to_numpy=True),
    )


def answer_faithfulness(generated_answer: str, ground_truth_answer: str) -> float:
    """
    Semantic similarity between generated and reference answer.
    Proxy for factual correctness / faithfulness.
    """
    model = _get_model()
    return _cosine_sim(
        model.encode(generated_answer, convert_to_numpy=True),
        model.encode(ground_truth_answer, convert_to_numpy=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    retrieved: list[str],
    relevant: set[str],
    context_text: str,
    generated_answer: str,
    ground_truth_answer: str,
    k: int = 5,
    latency_seconds: float = 0.0,
) -> dict:
    return {
        "Precision@K":        round(precision_at_k(retrieved, relevant, k), 4),
        "Recall@K":           round(recall_at_k(retrieved, relevant, k), 4),
        "F1@K":               round(f1_at_k(retrieved, relevant, k), 4),
        "MRR":                round(reciprocal_rank(retrieved, relevant), 4),
        "MAP":                round(average_precision(retrieved, relevant), 4),
        "nDCG@K":             round(ndcg_at_k(retrieved, relevant, k), 4),
        "HitRate@K":          round(hit_rate_at_k(retrieved, relevant, k), 4),
        "ContextRelevance":   round(context_relevance_score(context_text, ground_truth_answer), 4),
        "AnswerFaithfulness": round(answer_faithfulness(generated_answer, ground_truth_answer), 4),
        "Latency(s)":         round(latency_seconds, 4),
    }