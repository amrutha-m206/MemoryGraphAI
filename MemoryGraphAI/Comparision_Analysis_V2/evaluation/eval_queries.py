EVAL_QUERIES = [

# ─────────────────────────────
# BASIC (sanity checks)
# ─────────────────────────────
{
    "query": "What problem do graph attention models solve?",
    "ground_truth": "They solve the over-smoothing problem"
},
{
    "query": "What is the Cora dataset used for?",
    "ground_truth": "Node classification"
},

# ─────────────────────────────
# RELATIONSHIP-BASED
# ─────────────────────────────
{
    "query": "Which method is used to defend against graph attacks?",
    "ground_truth": "Adversarial learning methods defend against graph attacks"
},
{
    "query": "Which models are used in dynamic graph scenarios?",
    "ground_truth": "Recurrent graph neural networks are used in dynamic graphs"
},

# ─────────────────────────────
# MULTI-HOP (Graph advantage)
# ─────────────────────────────
{
    "query": "What problem is solved by a method used in graph neural networks?",
    "ground_truth": "Graph attention models solve the over-smoothing problem in graph neural networks"
},
{
    "query": "Which dataset is used for tasks evaluated in graph neural networks?",
    "ground_truth": "The Cora dataset is used for node classification in graph neural networks"
},
{
    "query": "How do adversarial learning methods improve graph neural networks?",
    "ground_truth": "They defend against graph data attacks, improving robustness"
},

# ─────────────────────────────
# CHAIN REASONING (2–3 hops)
# ─────────────────────────────
{
    "query": "What is the relationship between graph attention models and graph neural networks?",
    "ground_truth": "Graph neural networks use graph attention models which solve over-smoothing"
},
{
    "query": "Which problem is indirectly solved by graph neural networks through attention mechanisms?",
    "ground_truth": "Over-smoothing problem"
},
{
    "query": "How are datasets and tasks connected in GNN evaluation?",
    "ground_truth": "Datasets like Cora are used for tasks like node classification"
},

# ─────────────────────────────
# HARD REASONING (Hybrid shines)
# ─────────────────────────────
{
    "query": "Explain how a method, a problem, and a model are connected in graph learning",
    "ground_truth": "Graph attention models solve over-smoothing in graph neural networks"
},
{
    "query": "Which combination of method and dataset supports evaluation of graph models?",
    "ground_truth": "Graph neural networks use datasets like Cora for node classification tasks"
},
{
    "query": "How do defense mechanisms relate to graph learning tasks?",
    "ground_truth": "Adversarial learning methods defend graph neural networks from attacks"
},
{
    "query": "Which concepts together improve robustness in graph neural networks?",
    "ground_truth": "Adversarial learning methods and defense against graph attacks"
},

# ─────────────────────────────
# VERY HARD (3-hop reasoning)
# ─────────────────────────────
{
    "query": "How do graph neural networks use methods to solve problems in graph data?",
    "ground_truth": "Graph neural networks use graph attention models to solve over-smoothing problems"
},
{
    "query": "Trace the connection from dataset to model to task in graph learning",
    "ground_truth": "Cora dataset is used by graph neural networks for node classification"
},
{
    "query": "Explain how models, methods, and problems interact in graph neural networks",
    "ground_truth": "Graph neural networks use methods like graph attention models to solve problems like over-smoothing"
}
]