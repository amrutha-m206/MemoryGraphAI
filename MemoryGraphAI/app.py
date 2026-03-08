# import streamlit as st
# from query_engine import GraphQueryEngine
# from ingestion import DocumentIngestion
# from extraction import InformationExtractor
# from graph_builder import MemoryGraphBuilder
# import os

# # Set up page
# st.set_page_config(page_title="MemoryGraph AI", layout="wide")
# st.title("🧠 MemoryGraph AI")
# st.markdown("### AI System that Builds Memory Graphs from Research Papers")

# # Initialize classes (Cached so they don't reload every time)
# @st.cache_resource
# def load_engine():
#     return GraphQueryEngine()

# engine = load_engine()

# # --- SIDEBAR: Upload & Process ---
# with st.sidebar:
#     st.header("1. Data Ingestion")
#     uploaded_files = st.file_uploader("Upload Research Papers (PDF)", accept_multiple_files=True)
    
#     if st.button("Build/Update Graph"):
#         if uploaded_files:
#             with st.status("Processing Documents...", expanded=True) as status:
#                 # 1. Ingestion
#                 st.write("Extracting text...")
#                 # (Logic to save uploaded files to /data folder)
#                 for f in uploaded_files:
#                     with open(os.path.join("data", f.name), "wb") as buffer:
#                         buffer.write(f.read())
                
#                 # 2. Extraction & Building
#                 ingestor = DocumentIngestion()
#                 docs = ingestor.process_folder("./data")
#                 extractor = InformationExtractor()
                
#                 builder = MemoryGraphBuilder()
                
#                 for doc in docs:
#                     st.write(f"Analyzing {doc['filename']}...")
#                     # Increase max_chunks for better knowledge
#                     graph_data = extractor.extract(doc['content'], max_chunks=15)
#                     builder.build_graph(graph_data)
                
#                 builder.close()
#                 status.update(label="Graph Built Successfully!", state="complete")
#         else:
#             st.error("Please upload files first.")

# # --- MAIN AREA: Query & Reasoning ---
# st.header("2. Graph Reasoning Query")
# query = st.text_input("Ask a question about your research library:", placeholder="e.g. What datasets are used with GNNs?")

# if query:
#     with st.spinner("Traversing Memory Graph..."):
#         answer = engine.answer_question(query)
        
#         st.markdown("#### 🤖 AI Answer")
#         st.info(answer)
        
#         with st.expander("View Graph Evidence"):
#             evidence = engine.search_graph(query)
#             st.code(evidence)

# # --- VISUALIZATION ---
# st.header("3. Graph Explorer")
# st.write("Go to your [Neo4j Aura Console](https://console.neo4j.io) to see the full 3D interactive graph.")





# import streamlit as st
# import os
# import json

# # Import all your custom modules
# from ingestion import DocumentIngestion
# from extraction import InformationExtractor
# from graph_builder import MemoryGraphBuilder
# from graph_embeddings import MemoryGraphEmbedder  # <--- Added this
# from query_engine import GraphQueryEngine

# # Set up page
# st.set_page_config(page_title="MemoryGraph AI", layout="wide", page_icon="🧠")
# st.title("🧠 MemoryGraph AI")
# st.markdown("### End-to-End Graph-Based Research Assistant")

# # Initialize classes (Cached for performance)
# @st.cache_resource
# def load_tools():
#     return {
#         "ingestor": DocumentIngestion(),
#         "extractor": InformationExtractor(),
#         "query_engine": GraphQueryEngine()
#     }

# tools = load_tools()

# # --- SIDEBAR: Data Processing ---
# with st.sidebar:
#     st.header("📁 Step 1: Ingestion")
#     uploaded_files = st.file_uploader("Upload PDF Papers", accept_multiple_files=True, type=['pdf'])
    
#     process_btn = st.button("Build & Index Memory Graph")
    
#     if process_btn:
#         if uploaded_files:
#             with st.status("Processing Pipeline...", expanded=True) as status:
                
#                 # 1. SAVE FILES
#                 st.write("💾 Saving files to local storage...")
#                 os.makedirs("data", exist_ok=True)
#                 for f in uploaded_files:
#                     with open(os.path.join("data", f.name), "wb") as buffer:
#                         buffer.write(f.read())

#                 # 2. INGESTION (Text Extraction)
#                 st.write("📄 Extracting text from documents...")
#                 docs = tools["ingestor"].process_folder("./data")
                
#                 # 3. EXTRACTION & GRAPH BUILDING
#                 builder = MemoryGraphBuilder()
#                 for doc in docs:
#                     st.write(f"🔍 Extracting Entities from: {doc['filename']}...")
#                     # We process chunks and push to Neo4j
#                     graph_data = tools["extractor"].extract(doc['content'], max_chunks=10)
                    
#                     st.write(f"🕸️ Writing to Neo4j Database...")
#                     builder.build_graph(graph_data)
#                 builder.close()

#                 # 4. GRAPH EMBEDDINGS (CRITICAL MISSING STEP FIXED)
#                 st.write("🔢 Generating Vector Embeddings for Graph Nodes...")
#                 embedder = MemoryGraphEmbedder()
#                 embedder.generate_and_store_embeddings()
#                 embedder.close()
                
#                 status.update(label="✅ Graph Fully Indexed & Ready!", state="complete")
#                 st.success("Your Memory Graph is now mathematically searchable.")
#         else:
#             st.error("Please upload at least one PDF.")

# # --- MAIN AREA: Search & Discovery ---
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.header("🔍 Step 2: Graph-Based Reasoning")
#     query = st.text_input("Ask a question about your research library:", 
#                          placeholder="e.g. Which methods are used with the Cora dataset?")

#     if query:
#         with st.spinner("Searching Knowledge Graph..."):
#             answer = tools["query_engine"].answer_question(query)
#             st.markdown("#### 🤖 AI Research Answer")
#             st.info(answer)

# with col2:
#     st.header("📊 Step 3: Evidence")
#     if query:
#         st.write("Paths found in Graph:")
#         evidence = tools["query_engine"].search_graph(query)
#         if evidence:
#             st.code(evidence, language="text")
#         else:
#             st.write("No direct paths found.")

# # --- FOOTER: Visualization ---
# st.divider()
# st.subheader("🌐 Visual Explorer")
# st.write("To see the full interactive 3D graph, log in to your [Neo4j Aura Console](https://console.neo4j.io/).")
# if st.button("Explain Graph Structure"):
#     st.write("""
#     - **Nodes (Circles):** Represent Entities (Papers, Datasets, Methods).
#     - **Edges (Lines):** Represent Relationships (USES, EVALUATED_ON, PROPOSED_BY).
#     - **Vectors:** Hidden numbers on each node that allow the AI to find 'GNN' when you ask about 'Graph Models'.
#     """)