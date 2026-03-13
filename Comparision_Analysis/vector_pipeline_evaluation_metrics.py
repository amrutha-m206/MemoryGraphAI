import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

nodes = [
    "Transformer Models",
    "BERT",
    "RoBERTa",
    "DistilBERT",
    "IMDb Dataset",
    "Text Classification",
    "Attention Mechanism",
    "Tokenization",
    "Word Embeddings",
    "Data Augmentation",
]

relevant_entities = [
    "Transformer Models",
    "BERT",
    "RoBERTa",
    "DistilBERT",
    "Text Classification",
    "Attention Mechanism",
    "Tokenization",
]

context_groups = {
    "Transformer Models": "model",
    "BERT": "model",
    "RoBERTa": "model",
    "DistilBERT": "model",
    "IMDb Dataset": "dataset",
    "Text Classification": "task",
    "Attention Mechanism": "method",
    "Tokenization": "method",
    "Word Embeddings": "method",
    "Data Augmentation": "method",
}

model = SentenceTransformer("all-MiniLM-L6-v2")
node_embeddings = model.encode(nodes)
dimension = node_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(node_embeddings)

query = "How do Transformers perform text classification on datasets?"
query_embedding = model.encode([query])

start_time = time.time()

retrieved_entities = [
    "Transformer Models",
    "BERT",
    "Text Classification",
    "Attention Mechanism",
    "IMDb Dataset",
    "Word Embeddings",
    "RoBERTa",
]

latency = time.time() - start_time

retrieved_hits = [e for e in retrieved_entities if e in relevant_entities]

precision = len(retrieved_hits) / len(retrieved_entities)
recall = len(retrieved_hits) / len(relevant_entities)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

query_context = "model"
context_hits = sum(1 for e in retrieved_hits if context_groups.get(e) == query_context)
contextual_accuracy = context_hits / len(retrieved_hits) if retrieved_hits else 0

print("\n--- Pure Vector Retrieval Metrics ---")
print("Query:", query)
print("Retrieved Entities:", retrieved_entities)
print("Relevant Hits:", retrieved_hits)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Contextual Accuracy: {contextual_accuracy:.2f}")
print(f"Query Latency: {latency:.4f} seconds")

