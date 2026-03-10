# MemoryGraph AI: A Dynamic, Context-Aware Hybrid Knowledge Graph Framework for Documents with Comparative Vector Database Analysis

**MemoryGraph AI** is an advanced AI system that transcends traditional Vector-based RAG (Retrieval-Augmented Generation) by building a **Structured Knowledge Graph** from unstructured documents (PDFs, Text, Notes). 

While standard AI assistants search for text snippets, MemoryGraph AI builds an "Associative Memory" using **Neo4j**, enabling the AI to perform **multi-hop reasoning**—connecting ideas across different pages or even different documents.

---

## Key Features
- **Automated Knowledge Extraction:** Converts raw text into Entities (Nodes) and Relationships (Edges).
- **GraphRAG Architecture:** Combines the power of Large Language Models (LLMs) with the factual precision of Graph Databases.
- **Semantic Vector Indexing:** Uses mathematical embeddings to find concepts even if the wording is different.
- **Reasoning Engine:** Traverses the graph to answer complex questions that require "connecting the dots."
- **Performance Analytics:** Real-time tracking of ingestion latency, extraction speed, and answer accuracy.

---

## Technical Architecture

The system operates in a 6-phase pipeline:
1. **Ingestion Layer:** Parses PDFs/Docs and performs Recursive Character Splitting to maintain context.
2. **Extraction Layer (Groq/Llama 3):** Uses LLMs and Pydantic schemas to extract structured JSON triples (Subject -> Relation -> Object).
3. **Graph Construction (Neo4j):** Uses Cypher `MERGE` logic to build a non-redundant, interconnected knowledge web.
4. **Embedding Layer:** Generates 384-dimension vectors for every node using `Sentence-Transformers` for semantic search.
5. **Reasoning Engine:** Converts user queries into vector searches, retrieves the local sub-graph, and synthesizes a grounded answer.
6. **UI Layer (Streamlit):** A professional dashboard for document management and interactive discovery.

---

## Technology Stack
- **LLM:** Groq (Llama-3.1-8b-instant)
- **Database:** Neo4j AuraDB (Cloud)
- **NLP/Embeddings:** LangChain, Sentence-Transformers (all-MiniLM-L6-v2)
- **Frontend:** Streamlit
- **Language:** Python 3.10+

---

## Prerequisites

Before running the app, you need:
1. **Neo4j AuraDB:** Create a free instance at [neo4j.com/cloud/aura/](https://neo4j.com/cloud/aura/). Download your `credentials.txt`.
2. **Groq API Key:** Get a free API key at [console.groq.com](https://console.groq.com/).
3. **Python:** Ensure Python 3.10 or higher is installed.

---

## Installation

1. **Clone the project folder and navigate into it:**
   ```bash
   cd MemoryGraphAI
   ```

2. **Install the required dependencies:**
   ```bash
   pip install streamlit neo4j sentence-transformers langchain-groq python-dotenv PyPDF2 python-docx tqdm pandas
   ```

3. **Configure Environment Variables:**
   Create a file named `.env` in the root directory and add your credentials:
   ```env
   GROQ_API_KEY=your_groq_key_here
   NEO4J_URI=neo4j+s://your_instance_id.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password_here
   ```

---

## How to Run

The entire system is integrated into a single Streamlit dashboard. You do not need to run the individual phase scripts manually.

1. **Launch the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

2. **Using the Interface:**
   - **Step 1 (Sidebar):** Upload your PDF documents.
   - **Step 2 (Sidebar):** Click **"Build & Index Memory Graph"**. 
     - *Note: This will clean the temporary data folder, extract text, build nodes in Neo4j, and generate vector embeddings.*
   - **Step 3 (Main Screen):** Once the "Graph Ready" message appears, type your question in the search bar.
   - **Step 4 (Evaluation):** Review the **AI Answer** side-by-side with the **Graph Evidence** to ensure the answer is grounded in facts.

---

## Evaluation & Metrics

The app includes a built-in **Performance & Accuracy Dashboard** that tracks:
- **Ingestion Latency:** Time taken to parse documents.
- **Extraction Speed:** Time taken by the LLM to identify relationships.
- **Node Density:** Total count of unique entities stored in your "Memory."
- **Fact Verification:** A dedicated "Evidence" window showing the exact graph paths used to generate each answer.

---

## Project Structure
- `app.py`: The main Streamlit interface and orchestration logic.
- `ingestion.py`: PDF parsing and text cleaning.
- `extraction.py`: LLM-based entity and relationship extraction.
- `graph_builder.py`: Neo4j connection and Cypher query execution.
- `graph_embeddings.py`: Vector generation and indexing.
- `query_engine.py`: The GraphRAG reasoning and answering logic.

---

## Important Notes
- **Rate Limits:** This project uses the Groq Free Tier. If you process very large documents, you may hit the "Tokens Per Day" limit. If this happens, wait a few minutes or switch to the `llama-3.1-8b-instant` model for higher limits.
- **Data Privacy:** Documents are processed locally and then stored in your private Neo4j cloud instance. No data is used to train public models.

---
*Created as part of the MemoryGraph AI Research Initiative.*
