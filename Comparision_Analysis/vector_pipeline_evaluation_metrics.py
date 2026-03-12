import time
from sentence_transformers import SentenceTransformer
import numpy as np


nodes = [
    "Transformer Models",
    "BERT",
    "RoBERTa",
    "DistilBERT",
    "IMDb Dataset",
    "Text Classification",
    "Attention Mechanism"
]


model = SentenceTransformer("all-MiniLM-L6-v2")


embeddings = {node: model.encode(node) for node in nodes}

query = "How do Transformers perform text classification on datasets?"
query_emb = model.encode(query)


# Cosine similarity function
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Measure retrieval time
start_time = time.time()


# Rank nodes by similarity
ranked_nodes = sorted(
    nodes,
    key=lambda n: cosine_sim(query_emb, embeddings[n]),
    reverse=True
)


k = 5
retrieved_entities = ranked_nodes[:k]


latency = time.time() - start_time

relevant_entities = ["BERT", "RoBERTa", "DistilBERT", "Text Classification"]


retrieved_hits = [e for e in retrieved_entities if e in relevant_entities]


precision = len(retrieved_hits) / len(retrieved_entities)
recall = len(retrieved_hits) / len(relevant_entities)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
mrr = 1 / (retrieved_entities.index(retrieved_hits[0]) + 1) if retrieved_hits else 0
coverage = len(retrieved_hits) / len(relevant_entities)


print("\n--- Pure Vector Retrieval Metrics ---")
print("Query:", query)
print("Retrieved Entities:", retrieved_entities)
print("Relevant Hits:", retrieved_hits)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"MRR: {mrr:.2f}")
print(f"Coverage: {coverage:.2f}")
print(f"Query Latency: {latency:.4f} seconds")
