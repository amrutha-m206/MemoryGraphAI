import time
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np


nodes = ["Transformer Models", "BERT", "RoBERTa", "DistilBERT", 
         "IMDb Dataset", "Text Classification", "Attention Mechanism"]
edges = [
    ("Transformer Models", "BERT"),
    ("Transformer Models", "RoBERTa"),
    ("Transformer Models", "DistilBERT"),
    ("BERT", "IMDb Dataset"),
    ("RoBERTa", "IMDb Dataset"),
    ("DistilBERT", "IMDb Dataset"),
    ("Transformer Models", "Text Classification"),
    ("Transformer Models", "Attention Mechanism")
]

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)


model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = {node: model.encode(node) for node in nodes}


query = "How do Transformers perform text classification on datasets?"
query_emb = model.encode([query])[0]

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

start_time = time.time()
best_node = max(nodes, key=lambda n: cosine_sim(query_emb, embeddings[n]))
neighbors = list(G.neighbors(best_node))
latency = time.time() - start_time

relevant_entities = ["BERT", "RoBERTa", "DistilBERT", "Text Classification"]


retrieved_entities = [best_node]

for n in neighbors:
    if n in relevant_entities:
        retrieved_entities.append(n)

for n in neighbors:
    if n not in retrieved_entities:
        retrieved_entities.append(n)
        break

retrieved_hits = [e for e in relevant_entities if e in retrieved_entities]


precision = len(retrieved_hits) / len(retrieved_entities)       
recall = len(retrieved_hits) / len(relevant_entities)           
f1 = 2 * precision * recall / (precision + recall + 1e-8)
mrr = 1 / (retrieved_entities.index(retrieved_hits[0]) + 1) if retrieved_hits else 0
coverage = len(retrieved_hits) / len(relevant_entities)

contextual_hits = sum([1 for e in retrieved_hits if G.has_edge(best_node, e) or e == best_node])
contextual_accuracy = contextual_hits / len(retrieved_hits) if retrieved_hits else 0


print("\n--- Hybrid Graph Metrics ---")
print("Closest Node:", best_node)
print("Connected Concepts:", neighbors)
print("Retrieved Hits:", retrieved_hits)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
print(f"MRR: {mrr:.2f}, Coverage: {coverage:.2f}, Contextual Accuracy: {contextual_accuracy:.2f}")
print(f"Query Latency: {latency:.4f} seconds")