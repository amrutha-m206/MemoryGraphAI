from pipelines.vector_rag import VectorRAG
from pipelines.graph_rag import GraphRAG

class HybridRAG:

    def __init__(self, graph):
        self.vector = VectorRAG(graph["nodes"])
        self.graph = GraphRAG(graph)

    def answer(self, query):
        seeds = self.vector.retrieve(query)
        triples = self.graph.expand(seeds)

        context = "\n".join([f"{s} -[{r}]-> {t}" for s, r, t in triples])

        return f"""
Hybrid Reasoning:

Seeds: {seeds}

Graph Context:
{context}
"""