"""
graph_rag_pipeline.py
---------------------
A proper Graph RAG pipeline that makes FULL use of the knowledge graph.

The key difference from the original query_engine.py:
  - Seed retrieval: ANN vector search finds top-k seed nodes
  - Hop-1 expansion: all direct neighbours of each seed are collected
  - Relationship-aware context: context string uses (A)-[REL]->(B) triples
    instead of flat entity names, giving the LLM relational evidence
  - Ranked output: seeds appear first (highest semantic relevance),
    then neighbours ordered by how many seeds they are connected to

This is what makes Graph RAG genuinely different from Vector RAG:
the context passed to the LLM contains RELATIONSHIPS, not just entity names.
"""

import os
import time
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


class GraphRAGPipeline:
    def __init__(self, seed_k: int = 5, hop_depth: int = 1):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.seed_k = seed_k
        self.hop_depth = hop_depth
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

    def close(self):
        self.driver.close()

    # ─────────────────────────────────────────────────────────────────────
    # Core retrieval: seeds + graph expansion
    # ─────────────────────────────────────────────────────────────────────

    def retrieve(self, user_query: str) -> tuple[list[str], list[dict]]:
        """
        Returns:
          ranked_entities : ordered list of entity names
                            (seeds first, then neighbours by connection count)
          triples         : list of {"source", "relation", "target"} dicts
                            — this is the key graph-specific information
        """
        q_emb = self.model.encode(user_query).tolist()

        # Step 1: ANN seed search
        seed_query = """
        CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
        YIELD node, score
        RETURN node.name AS name, score
        ORDER BY score DESC
        """

        # Step 2: Expand seeds by traversing relationships
        expand_query = """
        CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
        YIELD node, score
        MATCH (node)-[r]-(neighbour)
        RETURN
            node.name        AS source,
            type(r)          AS relation,
            neighbour.name   AS target
        LIMIT 40
        """

        seeds = []
        triples = []
        neighbour_counts: dict[str, int] = {}

        with self.driver.session() as session:
            # Collect seeds
            for rec in session.run(seed_query, k=self.seed_k, embedding=q_emb):
                seeds.append(rec["name"])

            # Collect relational triples from one-hop expansion
            for rec in session.run(expand_query, k=self.seed_k, embedding=q_emb):
                triple = {
                    "source": rec["source"],
                    "relation": rec["relation"],
                    "target": rec["target"],
                }
                triples.append(triple)
                # Count how many seeds each neighbour is connected to
                neighbour = rec["target"] if rec["source"] in seeds else rec["source"]
                if neighbour not in seeds:
                    neighbour_counts[neighbour] = neighbour_counts.get(neighbour, 0) + 1

        # Rank neighbours by connection count (more connections = more central)
        ranked_neighbours = sorted(
            neighbour_counts.keys(), key=lambda n: neighbour_counts[n], reverse=True
        )

        # Seeds first, then ranked neighbours
        seen = set(seeds)
        ranked_entities = list(seeds)
        for n in ranked_neighbours:
            if n not in seen:
                seen.add(n)
                ranked_entities.append(n)

        return ranked_entities, triples

    def get_context_string(self, user_query: str) -> str:
        """
        Builds a RELATIONAL context string for the LLM.
        This is what distinguishes Graph RAG: the context contains
        (Entity A) -[RELATIONSHIP]-> (Entity B) triples, not just names.
        The LLM can use these relationships to reason about connections.
        """
        _, triples = self.retrieve(user_query)
        if not triples:
            return ""

        # Deduplicate triples
        seen = set()
        lines = []
        for t in triples:
            key = (t["source"], t["relation"], t["target"])
            if key not in seen:
                seen.add(key)
                lines.append(f"({t['source']}) -[{t['relation']}]-> ({t['target']})")

        return "\n".join(lines)

    def answer_question(self, question: str) -> str:
        """Generates a graph-grounded answer using relational context."""
        context = self.get_context_string(question)
        if not context.strip():
            return "No relevant information found in the knowledge graph."

        prompt = (
            "You are given a knowledge graph context consisting of entity relationships.\n"
            "Use ONLY these relationships to answer the question.\n"
            "Reason through the connections between entities.\n\n"
            f"Knowledge Graph Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer (citing specific relationships from the context):"
        )
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"