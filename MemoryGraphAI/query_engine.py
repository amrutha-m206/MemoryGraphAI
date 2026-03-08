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




# import os
# from neo4j import GraphDatabase
# from sentence_transformers import SentenceTransformer
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# load_dotenv()

# class GraphQueryEngine:
#     def __init__(self):
#         # 1. Connect to Neo4j
#         uri = os.getenv("NEO4J_URI")
#         user = os.getenv("NEO4J_USERNAME")
#         password = os.getenv("NEO4J_PASSWORD")
#         self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
#         # 2. Load the same Embedding Model as Phase 4
#         self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
#         # 3. Initialize LLM (Groq)
#         # self.llm = ChatGroq(
#         #     temperature=0,
#         #     model_name="llama-3.3-70b-versatile",
#         #     groq_api_key=os.getenv("GROQ_API_KEY")
#         # )

#         self.llm = ChatGroq(
#             temperature=0,
#             model_name="llama-3.1-8b-instant", # <--- Change here too
#             groq_api_key=os.getenv("GROQ_API_KEY")
#         )

#     def close(self):
#         self.driver.close()


#     def search_graph(self, user_query: str, top_k=5):
#         """Finds relevant nodes and their neighbors in the graph."""
#         query_embedding = self.model.encode(user_query).tolist()
        
#         # IMPROVED CYPHER: 
#         # 1. Look for relationships in BOTH directions
#         # 2. Use OPTIONAL MATCH so we don't lose the node if it has no links
#         search_query = """
#         CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
#         YIELD node, score
#         OPTIONAL MATCH (node)-[r]-(neighbor)
#         RETURN node.name AS source, type(r) AS relationship, neighbor.name AS target, score
#         LIMIT 15
#         """
        
#         context_parts = []
#         with self.driver.session() as session:
#             result = session.run(search_query, k=top_k, embedding=query_embedding)
#             for record in result:
#                 if record['relationship']: # If it has a connection
#                     context_parts.append(
#                         f"({record['source']}) -[{record['relationship']}]-> ({record['target']})"
#                     )
#                 else: # If it's a standalone node found by vector search
#                     context_parts.append(f"Entity found: {record['source']}")
        
#         # Remove duplicates from the context list
#         return "\n".join(list(set(context_parts)))

#     def answer_question(self, question: str):
#         print(f"\nThinking about: {question}...")
        
#         # 1. Retrieve Graph Evidence
#         graph_context = self.search_graph(question)
        
#         # SAFETY CHECK: If no graph data is found
#         if not graph_context.strip():
#             return "I couldn't find any specific information in the current memory graph to answer that accurately."

#         # 2. Improved "Researcher" Prompt
#         prompt = f"""
#         You are MemoryGraph AI, a professional research assistant. 
#         Your task is to provide a detailed, well-formulated answer to the user's question using the provided Knowledge Graph context.

#         Guidelines:
#         - Use the Graph Context as your "Ground Truth" or evidence.
#         - Explain the relationships logically (e.g., if Paper A uses Method B, mention that).
#         - Structure your answer like a formal research summary.
#         - If multiple papers or authors are mentioned, synthesize them together.

#         Knowledge Graph Context (Evidence):
#         {graph_context}

#         User Question: {question}

#         Helpful Research Answer:
#         """
        
#         try:
#             response = self.llm.invoke(prompt)
#             return response.content
#         except Exception as e:
#             # return f"Error connecting to LLM: {str(e)}"
#             if "429" in str(e):
#                 return "⚠️ Groq API Rate Limit Reached. Please wait a few minutes or switch to a higher-limit model."
#             return f"An error occurred: {str(e)}"


#     def answer_question(self, question: str):
#         print(f"\nThinking about: {question}...")
        
#         # 1. Retrieve Graph Evidence
#         graph_context = self.search_graph(question)
        
#         # 2. Improved "Researcher" Prompt
#         prompt = f"""
#         You are MemoryGraph AI, a professional research assistant. 
#         Your task is to provide a detailed, well-formulated answer to the user's question using the provided Knowledge Graph context.

#         Guidelines:
#         - Use the Graph Context as your "Ground Truth" or evidence.
#         - Use your internal knowledge to explain the relationships found in the graph (e.g., if you see a USES relationship, explain WHY it is used).
#         - Structure your answer like a research summary.
#         - If the graph context mentions specific papers or authors, include them in your answer.

#         Knowledge Graph Context (Evidence):
#         {graph_context}

#         User Question: {question}

#         Helpful Research Answer:
#         """
        
#         response = self.llm.invoke(prompt)
#         return response.content

# # --- TEST INTERFACE ---
# if __name__ == "__main__":
#     engine = GraphQueryEngine()
    
#     print("--- MemoryGraph AI Query Engine ---")
#     print("Type 'exit' to quit.")
    
#     while True:
#         user_input = input("\nAsk a research question: ")
#         if user_input.lower() == 'exit':
#             break
            
#         answer = engine.answer_question(user_input)
#         print(f"\nAI Answer:\n{answer}")
        
#     engine.close()



# def search_graph(self, user_query: str, top_k=5):
    #     """Finds relevant nodes and their neighbors in the graph."""
    #     # Convert user question into a vector
    #     query_embedding = self.model.encode(user_query).tolist()
        
    #     # Cypher: Vector search for nodes, then find their immediate relationships
    #     search_query = """
    #     CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
    #     YIELD node, score
    #     MATCH (node)-[r]->(neighbor)
    #     RETURN node.name AS source, type(r) AS relationship, neighbor.name AS target, score
    #     LIMIT 10
    #     """
        
    #     context_parts = []
    #     with self.driver.session() as session:
    #         result = session.run(search_query, k=top_k, embedding=query_embedding)
    #         for record in result:
    #             context_parts.append(
    #                 f"({record['source']}) -[{record['relationship']}]-> ({record['target']})"
    #             )
        
    #     return "\n".join(context_parts)

    # def answer_question(self, question: str):
    #     """Main pipeline: Question -> Graph Context -> AI Answer."""
    #     print(f"\nThinking about: {question}...")
        
    #     # 1. Retrieve Graph Evidence
    #     graph_context = self.search_graph(question)
        
    #     if not graph_context:
    #         return "I couldn't find any relevant connections in my memory graph."

    #     # 2. Create the Prompt
    #     prompt = f"""
    #     You are MemoryGraph AI. Use the following Knowledge Graph excerpts to answer the user's question.
    #     The excerpts are in the format: (Source) -[RELATIONSHIP]-> (Target)

    #     Knowledge Graph Context:
    #     {graph_context}

    #     User Question: {question}

    #     Answer based ONLY on the graph context provided. If you can't find the answer, say you don't know.
    #     """
        
    #     # 3. Get AI Response
    #     response = self.llm.invoke(prompt)
    #     return response.content