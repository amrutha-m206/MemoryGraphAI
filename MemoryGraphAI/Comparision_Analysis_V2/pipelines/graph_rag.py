class GraphRAG:

    def __init__(self, graph):
        self.nodes = graph["nodes"]
        self.edges = graph["edges"]

    def expand(self, seeds):
        triples = []
        for s, r, t in self.edges:
            if s in seeds:
                triples.append((s, r, t))
        return triples

    def answer(self, query, triples):
        context = "\n".join([f"{s} -[{r}]-> {t}" for s, r, t in triples])
        return f"Using relationships:\n{context}"