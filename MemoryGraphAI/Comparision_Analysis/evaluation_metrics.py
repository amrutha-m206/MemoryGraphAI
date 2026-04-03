"""
evaluation_metrics.py
---------------------
Implements all retrieval evaluation metrics used to compare
Graph-based RAG vs Vector-based RAG.

Metrics
-------
1.  Precision@K          — fraction of retrieved items that are relevant
2.  Recall@K             — fraction of relevant items that were retrieved
3.  F1@K                 — harmonic mean of Precision and Recall
4.  Mean Reciprocal Rank (MRR) — how early the first relevant item appears
5.  Normalized Discounted Cumulative Gain (nDCG@K)
                         — rewards relevant items appearing earlier in the ranked list
6.  Average Precision (AP) / Mean Average Precision (MAP)
                         — area under the precision-recall curve
7.  Hit Rate@K (HR@K)    — binary: did ANY relevant item appear in top-K?
8.  Context Relevance Score (CRS)
                         — semantic cosine similarity between the retrieved context
                           string and the ground-truth answer string
9.  Answer Faithfulness  — cosine similarity between generated answer and
                           ground-truth answer (proxy for faithfulness)
10. Query Latency        — wall-clock time in seconds
"""

import math
import time
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None  # Lazy-loaded once


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# Core ranking metrics (operate on lists of entity names / IDs)
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-K retrieved items that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant items found in the top-K retrieved."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / (rank of first relevant item). Returns 0 if no relevant item found."""
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """
    Area under the Precision-Recall curve (AP).
    AP = sum( Precision@i * rel(i) ) / |relevant|
    """
    if not relevant:
        return 0.0
    num_hits = 0
    score = 0.0
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            num_hits += 1
            score += num_hits / i
    return score / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain.
    Treats every relevant item as having relevance grade = 1.
    """
    def dcg(items):
        return sum(
            1.0 / math.log2(i + 2)
            for i, item in enumerate(items)
            if item in relevant
        )

    top_k = retrieved[:k]
    actual_dcg = dcg(top_k)
    # Ideal: the relevant items placed at the top
    ideal_len = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if at least one relevant item is in the top-K, else 0.0."""
    top_k = set(retrieved[:k])
    return 1.0 if top_k & relevant else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Semantic / generation metrics
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


def context_relevance_score(context_text: str, ground_truth_answer: str) -> float:
    """
    Semantic similarity between the retrieved context string and the
    ground-truth answer.  Range: [-1, 1], higher = more relevant context.

    Interpretation
    --------------
    > 0.6  → context is highly on-topic
    0.4–0.6 → moderately relevant
    < 0.4  → context is mostly off-topic
    """
    model = _get_model()
    emb_ctx = model.encode(context_text, convert_to_numpy=True)
    emb_gt = model.encode(ground_truth_answer, convert_to_numpy=True)
    return _cosine_sim(emb_ctx, emb_gt)


def answer_faithfulness(generated_answer: str, ground_truth_answer: str) -> float:
    """
    Cosine similarity between the generated answer and the ground-truth answer.
    Acts as a proxy for factual faithfulness.

    Interpretation
    --------------
    > 0.8  → very faithful / correct answer
    0.6–0.8 → broadly correct
    0.4–0.6 → partially correct
    < 0.4  → poor / unfaithful answer
    """
    model = _get_model()
    emb_gen = model.encode(generated_answer, convert_to_numpy=True)
    emb_gt = model.encode(ground_truth_answer, convert_to_numpy=True)
    return _cosine_sim(emb_gen, emb_gt)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helper
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
    """
    Convenience wrapper — returns every metric in one dictionary.

    Parameters
    ----------
    retrieved           : ordered list of entity names returned by the retriever
    relevant            : ground-truth set of entity names that are relevant
    context_text        : raw context string sent to the LLM
    generated_answer    : answer produced by the LLM
    ground_truth_answer : reference answer for the query
    k                   : cut-off rank (default 5)
    latency_seconds     : wall-clock time for the full retrieval + generation step
    """
    return {
        "Precision@K":           round(precision_at_k(retrieved, relevant, k), 4),
        "Recall@K":              round(recall_at_k(retrieved, relevant, k), 4),
        "F1@K":                  round(f1_at_k(retrieved, relevant, k), 4),
        "MRR":                   round(reciprocal_rank(retrieved, relevant), 4),
        "MAP":                   round(average_precision(retrieved, relevant), 4),
        "nDCG@K":                round(ndcg_at_k(retrieved, relevant, k), 4),
        "HitRate@K":             round(hit_rate_at_k(retrieved, relevant, k), 4),
        "ContextRelevance":      round(context_relevance_score(context_text, ground_truth_answer), 4),
        "AnswerFaithfulness":    round(answer_faithfulness(generated_answer, ground_truth_answer), 4),
        "Latency(s)":            round(latency_seconds, 4),
    }