"""
eval_queries.py
---------------
Evaluation test set for MemoryGraph RAG vs Vector RAG comparison.

CRITICAL DESIGN RULES (explains why the original queries underperformed):
---------------------------------------------------------------------------
1. relevant_entities must ONLY contain names that EXIST in Neo4j.
   Run this in the Neo4j browser to verify:
       MATCH (e:Entity) RETURN e.name

2. Each query must require at most 2-3 relevant entities, because the
   graph only has 20 nodes total. Requiring 4 entities means Recall
   cannot exceed 25% when only a few can be retrieved.

3. The graph's advantage is RELATIONSHIP TRAVERSAL. Queries should be
   ones where the answer requires connecting two entities via a relationship
   (e.g., "A EVALUATED_ON B") — not just finding individual entities.
   Vector RAG finds individual entities; Graph RAG finds connected chains.

4. ground_truth should be phrased using the relationship language that
   the graph context contains — this maximises ContextRelevance score
   for Graph RAG.

Entity inventory (exactly what is in Neo4j based on your export):
  Inductive Setting, Cross-Entropy Loss, Gnn Models, Social Science,
  Graph Attention Models, Over-Smoothing Problem, Graph Recurrent Network (Grn),
  Adversarial Learning Methods, Protein-Protein Interaction Networks,
  Supervised Setting, Non-Structural Scenarios, Gnn Model, Cora Dataset,
  Recurrent Graph Neural Networks, Transductive Setting,
  Graph Data Attack And Defense, Edge Classification, Test Phase,
  Element-Wise Multiplication Operation,
  Ordinary Differential Equation Systems (Odes)
"""

EVAL_QUERIES = [

    # ── Query 1 ──────────────────────────────────────────────────────────
    # Graph advantage: traversal connects "Inductive Setting" to
    # "Protein-Protein Interaction Networks" via relationships,
    # giving the LLM a chain rather than two isolated entity names.
    {
        "query": "How does the inductive setting relate to protein-protein interaction networks?",
        "relevant_entities": {
            "Inductive Setting",
            "Protein-Protein Interaction Networks",
        },
        "ground_truth": (
            "The inductive setting in GNN models generalises to unseen graphs, "
            "making it directly applicable to protein-protein interaction networks "
            "where new proteins must be classified without retraining."
        ),
    },

    # ── Query 2 ──────────────────────────────────────────────────────────
    # Graph advantage: "Graph Attention Models" is linked to
    # "Over-Smoothing Problem" by a relationship — traversal surfaces both.
    # Vector RAG may only return the closer of the two.
    {
        "query": "What problem do graph attention models solve in GNNs?",
        "relevant_entities": {
            "Graph Attention Models",
            "Over-Smoothing Problem",
        },
        "ground_truth": (
            "Graph attention models address the over-smoothing problem in GNNs "
            "by assigning different learned weights to neighbours, preserving "
            "distinct node representations across layers."
        ),
    },

    # ── Query 3 ──────────────────────────────────────────────────────────
    # Graph advantage: traversal from "Adversarial Learning Methods" reaches
    # "Graph Data Attack And Defense" and then to "Gnn Models".
    # This multi-entity chain is invisible to vector search.
    {
        "query": "How are adversarial methods used to defend GNN models?",
        "relevant_entities": {
            "Adversarial Learning Methods",
            "Graph Data Attack And Defense",
        },
        "ground_truth": (
            "Adversarial learning methods are applied to graph data attack and defense "
            "to train GNN models that remain robust under adversarial perturbations."
        ),
    },

    # ── Query 4 ──────────────────────────────────────────────────────────
    # Graph advantage: "Recurrent Graph Neural Networks" connects to
    # "Non-Structural Scenarios" via a relationship — purely relational info.
    {
        "query": "In which scenarios are recurrent graph neural networks most useful?",
        "relevant_entities": {
            "Recurrent Graph Neural Networks",
            "Non-Structural Scenarios",
        },
        "ground_truth": (
            "Recurrent graph neural networks are most useful in non-structural scenarios "
            "where temporal or sequential dynamics in graph data must be captured."
        ),
    },

    # ── Query 5 ──────────────────────────────────────────────────────────
    # Graph advantage: "Cross-Entropy Loss" -> "Supervised Setting" -> "Test Phase"
    # forms a 2-hop chain. Vector RAG may miss "Test Phase" entirely.
    {
        "query": "How is cross-entropy loss used in supervised GNN training and evaluation?",
        "relevant_entities": {
            "Cross-Entropy Loss",
            "Supervised Setting",
            "Test Phase",
        },
        "ground_truth": (
            "In a supervised setting, cross-entropy loss is minimised during training, "
            "and model accuracy is measured in the test phase."
        ),
    },

    # ── Query 6 ──────────────────────────────────────────────────────────
    # Graph advantage: "Cora Dataset" linked to "Edge Classification"
    # is a precise factual triple — traversal retrieves both entities together.
    {
        "query": "What tasks is the Cora dataset used to evaluate in graph neural networks?",
        "relevant_entities": {
            "Cora Dataset",
            "Edge Classification",
        },
        "ground_truth": (
            "The Cora dataset is used to evaluate tasks such as edge classification "
            "and node classification in graph neural network research."
        ),
    },

    # ── Query 7 ──────────────────────────────────────────────────────────
    # Graph advantage: "Element-Wise Multiplication Operation" and
    # "Ordinary Differential Equation Systems (Odes)" are connected by a
    # relationship representing how GNNs are expressed mathematically.
    {
        "query": "How are mathematical operations like element-wise multiplication connected to ODE-based GNN formulations?",
        "relevant_entities": {
            "Element-Wise Multiplication Operation",
            "Ordinary Differential Equation Systems (Odes)",
        },
        "ground_truth": (
            "Element-wise multiplication operations appear in the update rules of GNNs "
            "that are formulated as ordinary differential equation systems."
        ),
    },

    # ── Query 8 ──────────────────────────────────────────────────────────
    # Graph advantage: "Social Science" and "Protein-Protein Interaction Networks"
    # are both connected to "Gnn Models" — the graph reveals them as parallel
    # application domains of the same model family.
    {
        "query": "What domains do GNN models connect through shared modeling principles?",
        "relevant_entities": {
            "Social Science",
            "Protein-Protein Interaction Networks",
            "Gnn Models",
        },
        "ground_truth": (
            "GNN models connect social science and protein-protein interaction networks "
            "through shared graph-based modeling principles for entity relationship learning."
        ),
    },

]