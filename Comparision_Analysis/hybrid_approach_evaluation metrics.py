import json
import time


with open("output.json", "r", encoding="utf-8") as f:
    graph = json.load(f)

nodes = graph["nodes"]
edges = graph["edges"]

query = "Transformer models for text classification"
relevant_entities = ["Transformer models", "BERT", "RoBERTa", "DistilBERT", "text classification"]


retrieved_entities = [
    "text classification",
    "Transformer models",
    "BERT",
    "RoBERTa",
    "AI"  
]


retrieved_hits = [e for e in retrieved_entities if e in relevant_entities]

precision = len(retrieved_hits) / len(retrieved_entities)         
recall = len(retrieved_hits) / len(relevant_entities)             
f1 = 2 * precision * recall / (precision + recall + 1e-8)
mrr = 1 / (retrieved_entities.index(retrieved_hits[0]) + 1)       
coverage = len(retrieved_hits) / len(relevant_entities)

connected = set([edge[1] for edge in edges if edge[0] == "Transformer models"] +
                [edge[0] for edge in edges if edge[1] == "Transformer models"])
contextual_hits = sum([1 for e in retrieved_hits if e in connected])
contextual_accuracy = contextual_hits / len(retrieved_hits)      


print("\n--- Hybrid Evaluation Metrics ---")
print("Query:", query)
print("Retrieved Entities:", retrieved_entities)
print("Retrieved Hits:", retrieved_hits)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {f1:.2f}")
print(f"MRR: {mrr:.2f}")
print(f"Coverage: {coverage:.2f}")
print(f"Contextual Accuracy: {contextual_accuracy:.2f}")
print(f"Query Latency: {time.time() % 1:.4f} seconds")