"""
vector_pipeline.py
------------------
A pure vector-based RAG retrieval pipeline.

How it works:
  1. Load all entities from Neo4j (same data the graph uses).
  2. Embed the user query with the same sentence-transformer model.
  3. Compute cosine similarity between the query and every stored entity embedding.
  4. Return the top-k entities ranked purely by cosine distance — no graph
     traversal, no relationship awareness.
  5. Feed those entities as flat context to the LLM for answer generation.

This deliberately mirrors the GraphQueryEngine interface so both pipelines
can be benchmarked with identical inputs and the same answer-generation prompt.
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
        # Cache all node embeddings once on init for speed
        self._all_names: list[str] = []
        self._all_embeddings: np.ndarray | None = None
        self._load_embeddings()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_embeddings(self):
        """Pull every Entity name + embedding from Neo4j into RAM."""
        query = "MATCH (e:Entity) WHERE e.embedding IS NOT NULL RETURN e.name AS name, e.embedding AS emb"
        names, vecs = [], []
        with self.driver.session() as session:
            for rec in session.run(query):
                names.append(rec["name"])
                vecs.append(rec["emb"])
        self._all_names = names
        self._all_embeddings = np.array(vecs, dtype=np.float32) if vecs else np.empty((0, 384))

    def _cosine_similarity(self, query_vec: np.ndarray) -> np.ndarray:
        """Batch cosine similarity between one query vector and the corpus."""
        if self._all_embeddings.shape[0] == 0:
            return np.array([])
        q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        norms = np.linalg.norm(self._all_embeddings, axis=1, keepdims=True) + 1e-10
        normed = self._all_embeddings / norms
        return normed @ q  # shape: (N,)

    # ------------------------------------------------------------------
    # Public interface (mirrors GraphQueryEngine)
    # ------------------------------------------------------------------

    def search(self, user_query: str) -> tuple[list[str], list[float]]:
        """
        Returns (entity_names, similarity_scores) for the top-k hits,
        ranked purely by cosine similarity — no graph edges used.
        """
        q_vec = self.model.encode(user_query, convert_to_numpy=True)
        scores = self._cosine_similarity(q_vec)
        if scores.size == 0:
            return [], []
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        names = [self._all_names[i] for i in top_idx]
        sims = [float(scores[i]) for i in top_idx]
        return names, sims

    def get_context_string(self, user_query: str) -> str:
        """Returns a flat text context (no relationships) for LLM consumption."""
        names, scores = self.search(user_query)
        if not names:
            return ""
        lines = [f"Entity: {n}  (similarity={s:.4f})" for n, s in zip(names, scores)]
        return "\n".join(lines)

    def answer_question(self, question: str) -> str:
        """Generates an answer using the top-k entities as context."""
        context = self.get_context_string(question)
        if not context.strip():
            return "No relevant information found via vector search."
        prompt = f"Context (retrieved by vector similarity):\n{context}\n\nQuestion: {question}\n\nAnswer directly:"
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

    def close(self):
        self.driver.close()