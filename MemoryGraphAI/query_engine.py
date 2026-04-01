import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class GraphQueryEngine:
    def __init__(self):
        # 1. Connect to Neo4j
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # 2. Load the Embedding Model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. Initialize LLM (Using 8b-instant for higher rate limits and speed)
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def close(self):
        self.driver.close()

    def search_graph(self, user_query: str, top_k=5):
        """Finds relevant nodes and their neighbors in the graph using bi-directional search."""
        query_embedding = self.model.encode(user_query).tolist()
        
        # IMPROVED CYPHER: Looks for relationships in BOTH directions
        search_query = """
        CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
        YIELD node, score
        OPTIONAL MATCH (node)-[r]-(neighbor)
        RETURN node.name AS source, type(r) AS relationship, neighbor.name AS target, score
        LIMIT 15
        """
        
        context_parts = []
        with self.driver.session() as session:
            result = session.run(search_query, k=top_k, embedding=query_embedding)
            for record in result:
                if record['relationship']: 
                    context_parts.append(
                        f"({record['source']}) -[{record['relationship']}]-> ({record['target']})"
                    )
                else: 
                    context_parts.append(f"Entity found: {record['source']}")

    def get_visualization_data(self, user_query: str, top_k=5):
        """Fetches nodes and edges specifically for the visual graph."""
        query_embedding = self.model.encode(user_query).tolist()
        search_query = """
        CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
        YIELD node, score
        MATCH (node)-[r]-(neighbor)
        RETURN node.name AS source, type(r) AS rel, neighbor.name AS target
        LIMIT 25
        """
        nodes = set()
        edges = []
        with self.driver.session() as session:
            result = session.run(search_query, k=top_k, embedding=query_embedding)
            for record in result:
                src, tgt, rel = record['source'], record['target'], record['rel']
                nodes.add(src)
                nodes.add(tgt)
                edges.append({"source": src, "target": tgt, "label": rel})
        return list(nodes), edges

    def get_graph_analytics(self):
        """Runs Graph Data Science metrics to find influential nodes and themes."""
        # Query 1: Hub Entities (Most connected)
        centrality_query = """
        MATCH (e:Entity)
        RETURN e.name AS name, count{(e)--()} AS connections
        ORDER BY connections DESC
        LIMIT 10
        """
        # Query 2: Theme Clusters (Entities sharing common neighbors)
        community_query = """
        MATCH (e1:Entity)-[:RELATED]-(common)-[:RELATED]-(e2:Entity)
        WHERE e1.name < e2.name
        RETURN e1.name + " & " + e2.name AS pair, count(common) AS strength
        ORDER BY strength DESC
        LIMIT 10
        """
        hubs = []
        communities = []
        with self.driver.session() as session:
            hub_result = session.run(centrality_query)
            for record in hub_result:
                hubs.append({"Entity": record["name"], "Connections": record["connections"]})
            comm_result = session.run(community_query)
            for record in comm_result:
                communities.append({"Cluster": record["pair"], "Strength": record["strength"]})
        return hubs, communities
        
        
        return "\n".join(list(set(context_parts)))

    def answer_question(self, question: str):
        """Generalized answering engine for any document type."""
        print(f"\nThinking about: {question}...")
        
        # 1. Retrieve Graph Evidence
        graph_context = self.search_graph(question)
        
        # SAFETY CHECK: If no graph data is found at all
        if not graph_context.strip():
            return "I couldn't find any specific information in the documents to answer that accurately."

        # 2. Generalized Intelligence Prompt (No longer restricted to research)
        prompt = f"""
        You are MemoryGraph AI, a sophisticated knowledge assistant. 
        Your task is to provide a clear and direct answer to the user's question using the provided context.

        Guidelines:
        - Use the Knowledge Graph Context below as your only source of facts.
        - Be direct. If the context contains the answer, state it clearly.
        - If the context doesn't contain a direct answer but has related information, explain those relationships.
        - Use bullet points for multiple facts.
        - Do NOT use academic headers like 'Research Summary', 'Hypothesis', or 'Future Directions' unless specifically asked.
        - Avoid making up connections that don't exist in the context.

        Knowledge Graph Context (Evidence):
        {graph_context}

        User Question: {question}

        Final Answer:
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            if "429" in str(e):
                return "⚠️ Groq API Rate Limit Reached. Please wait a moment."
            return f"An error occurred: {str(e)}"

# --- TEST INTERFACE ---
if __name__ == "__main__":
    engine = GraphQueryEngine()
    print("--- MemoryGraph AI Query Engine ---")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nAsk a question: ")
        if user_input.lower() == 'exit':
            break
            
        answer = engine.answer_question(user_input)
        print(f"\nAI Answer:\n{answer}")
        
    engine.close()
