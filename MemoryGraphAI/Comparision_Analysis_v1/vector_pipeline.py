"""
vector_pipeline.py
------------------
Pure vector-based RAG pipeline (no graph traversal, no relationships).

Retrieves entities by cosine similarity only, builds a flat context string
(entity names + scores), and passes that to the LLM.

The context deliberately contains NO relationship information — this is the
baseline that Graph RAG is compared against.
"""

import os
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


class VectorRAGPipeline:
    def __init__(self, top_k: int = 5):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.top_k = top_k
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        self._all_names: list[str] = []
        self._all_embeddings: np.ndarray | None = None
        self._load_embeddings()

    def _load_embeddings(self):
        query = "MATCH (e:Entity) WHERE e.embedding IS NOT NULL RETURN e.name AS name, e.embedding AS emb"
        names, vecs = [], []
        with self.driver.session() as session:
            for rec in session.run(query):
                names.append(rec["name"])
                vecs.append(rec["emb"])
        self._all_names = names
        self._all_embeddings = np.array(vecs, dtype=np.float32) if vecs else np.empty((0, 384))

    def _cosine_similarity(self, query_vec: np.ndarray) -> np.ndarray:
        if self._all_embeddings.shape[0] == 0:
            return np.array([])
        q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        norms = np.linalg.norm(self._all_embeddings, axis=1, keepdims=True) + 1e-10
        normed = self._all_embeddings / norms
        return normed @ q

    def search(self, user_query: str) -> tuple[list[str], list[float]]:
        q_vec = self.model.encode(user_query, convert_to_numpy=True)
        scores = self._cosine_similarity(q_vec)
        if scores.size == 0:
            return [], []
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        return [self._all_names[i] for i in top_idx], [float(scores[i]) for i in top_idx]

    def get_context_string(self, user_query: str) -> str:
        """Flat list of top-K entities — no relationship information."""
        names, scores = self.search(user_query)
        if not names:
            return ""
        return "\n".join(
            f"Entity: {n}  (similarity={s:.4f})" for n, s in zip(names, scores)
        )

    def answer_question(self, question: str) -> str:
        context = self.get_context_string(question)
        if not context.strip():
            return "No relevant information found via vector search."
        prompt = (
            "You are given a list of entities retrieved by vector similarity search.\n"
            "Use these entities to answer the question.\n\n"
            f"Retrieved Entities:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer directly:"
        )
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

    def close(self):
        self.driver.close()