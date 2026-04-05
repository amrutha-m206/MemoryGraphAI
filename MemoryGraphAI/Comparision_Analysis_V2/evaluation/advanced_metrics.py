import re

# ─────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────

def count_triples(context: str):
    """
    Counts number of relationship triples like:
    A -> B OR (A)-[REL]->(B)
    """
    return context.count("->")


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def overlap(a: str, b: str):
    """
    Simple token overlap (for grounding / relevance)
    """
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens:
        return 0
    return len(a_tokens & b_tokens) / len(a_tokens)


# ─────────────────────────────────────────────────────────────
# MAIN METRICS FUNCTION
# ─────────────────────────────────────────────────────────────

def compute_advanced_metrics(
    query: str,
    hybrid_context: str,
    vector_context: str,
    hybrid_answer: str,
    vector_answer: str,
    ground_truth: str
):

    # ─────────────────────────────
    # 1. RELATIONSHIP METRICS
    # ─────────────────────────────
    hybrid_relations = count_triples(hybrid_context)
    vector_relations = count_triples(vector_context)

    relation_gain = hybrid_relations - vector_relations

    # ─────────────────────────────
    # 2. CONTEXT RICHNESS
    # ─────────────────────────────
    hybrid_tokens = len(tokenize(hybrid_context))
    vector_tokens = len(tokenize(vector_context))

    richness_gain = hybrid_tokens - vector_tokens

    # ─────────────────────────────
    # 3. GROUNDING SCORE
    # ─────────────────────────────
    hybrid_grounding = overlap(hybrid_answer, hybrid_context)
    vector_grounding = overlap(vector_answer, vector_context)

    grounding_gain = hybrid_grounding - vector_grounding

    # ─────────────────────────────
    # 4. RELEVANCE TO QUERY
    # ─────────────────────────────
    hybrid_query_overlap = overlap(query, hybrid_answer)
    vector_query_overlap = overlap(query, vector_answer)

    query_relevance_gain = hybrid_query_overlap - vector_query_overlap

    # ─────────────────────────────
    # 5. GROUND TRUTH ALIGNMENT
    # ─────────────────────────────
    hybrid_gt_overlap = overlap(ground_truth, hybrid_answer)
    vector_gt_overlap = overlap(ground_truth, vector_answer)

    gt_alignment_gain = hybrid_gt_overlap - vector_gt_overlap

    # ─────────────────────────────
    # 6. MULTI-HOP PROXY
    # ─────────────────────────────
    # More relations = deeper reasoning
    hybrid_multihop_score = hybrid_relations
    vector_multihop_score = vector_relations

    multihop_gain = hybrid_multihop_score - vector_multihop_score

    # ─────────────────────────────
    # FINAL OUTPUT
    # ─────────────────────────────

    return {
        "relationship_metrics": {
            "hybrid_relation_count": hybrid_relations,
            "vector_relation_count": vector_relations,
            "relation_gain": relation_gain
        },

        "context_metrics": {
            "hybrid_token_count": hybrid_tokens,
            "vector_token_count": vector_tokens,
            "richness_gain": richness_gain
        },

        "grounding_metrics": {
            "hybrid_grounding": round(hybrid_grounding, 3),
            "vector_grounding": round(vector_grounding, 3),
            "grounding_gain": round(grounding_gain, 3)
        },

        "query_relevance": {
            "hybrid": round(hybrid_query_overlap, 3),
            "vector": round(vector_query_overlap, 3),
            "gain": round(query_relevance_gain, 3)
        },

        "ground_truth_alignment": {
            "hybrid": round(hybrid_gt_overlap, 3),
            "vector": round(vector_gt_overlap, 3),
            "gain": round(gt_alignment_gain, 3)
        },

        "multihop_reasoning": {
            "hybrid_score": hybrid_multihop_score,
            "vector_score": vector_multihop_score,
            "gain": multihop_gain
        }
    }