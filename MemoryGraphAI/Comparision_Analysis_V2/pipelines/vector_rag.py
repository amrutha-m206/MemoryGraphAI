from sentence_transformers import SentenceTransformer
import numpy as np

class VectorRAG:

    def __init__(self, nodes):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.nodes = nodes
        self.embeddings = self.model.encode(nodes)

    def retrieve(self, query, k=3):
        q_emb = self.model.encode(query)

        scores = np.dot(self.embeddings, q_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)
        )

        top_idx = np.argsort(scores)[-k:][::-1]
        return [self.nodes[i] for i in top_idx]

    def answer(self, query, retrieved):
        return f"Answer based on: {', '.join(retrieved)}"