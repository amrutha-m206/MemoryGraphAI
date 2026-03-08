# import streamlit as st
# import os
# import shutil
# import json

# # Import all your custom modules
# from ingestion import DocumentIngestion
# from extraction import InformationExtractor
# from graph_builder import MemoryGraphBuilder
# from graph_embeddings import MemoryGraphEmbedder 
# from query_engine import GraphQueryEngine

# # Set up page
# st.set_page_config(page_title="MemoryGraph AI", layout="wide", page_icon="🧠")
# st.title("🧠 MemoryGraph AI")
# st.markdown("### End-to-End Graph-Based Research Assistant")

# # Initialize classes
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
    
#     # NEW: Option to clear the existing graph
#     clear_db = st.checkbox("Clear existing graph before building?")
    
#     uploaded_files = st.file_uploader("Upload PDF Papers", accept_multiple_files=True, type=['pdf'])
    
#     process_btn = st.button("Build & Index Memory Graph")
    
#     if process_btn:
#         if uploaded_files:
#             with st.status("Processing Pipeline...", expanded=True) as status:
                
#                 # --- FIX: CLEAR PREVIOUS FILES ---
#                 st.write("🧹 Cleaning temporary storage...")
#                 if os.path.exists("data"):
#                     shutil.rmtree("data") # Delete the whole folder
#                 os.makedirs("data", exist_ok=True) # Recreate empty folder

#                 # 1. SAVE ONLY CURRENTLY UPLOADED FILES
#                 st.write("💾 Saving new files...")
#                 for f in uploaded_files:
#                     with open(os.path.join("data", f.name), "wb") as buffer:
#                         buffer.write(f.read())

#                 # 2. INGESTION (Text Extraction)
#                 st.write("📄 Extracting text...")
#                 docs = tools["ingestor"].process_folder("./data")
                
#                 # 3. EXTRACTION & GRAPH BUILDING
#                 builder = MemoryGraphBuilder()
                
#                 # If user checked "Clear existing graph", wipe Neo4j first
#                 if clear_db:
#                     st.warning("🗑️ Clearing Neo4j Database...")
#                     builder.clear_database()
                
#                 for doc in docs:
#                     st.write(f"🔍 Analyzing: {doc['filename']}...")
#                     # Process chunks (Reduced max_chunks to 5 to save Groq tokens)
#                     graph_data = tools["extractor"].extract(doc['content'], max_chunks=5)
                    
#                     st.write(f"🕸️ Adding to Knowledge Graph...")
#                     builder.build_graph(graph_data)
#                 builder.close()

#                 # 4. GRAPH EMBEDDINGS
#                 st.write("🔢 Indexing nodes for search...")
#                 embedder = MemoryGraphEmbedder()
#                 embedder.generate_and_store_embeddings()
#                 embedder.close()
                
#                 status.update(label="✅ Success: Graph Built!", state="complete")
#                 st.success(f"Processed {len(uploaded_files)} files.")
#         else:
#             st.error("Please upload at least one PDF.")

# # --- MAIN AREA ---
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.header("🔍 Step 2: Reasoning")
#     query = st.text_input("Ask a question:", placeholder="What is discussed in the uploaded papers?")

#     if query:
#         with st.spinner("Thinking..."):
#             answer = tools["query_engine"].answer_question(query)
#             st.markdown("#### 🤖 AI Answer")
#             st.info(answer)

# with col2:
#     st.header("📊 Evidence")
#     if query:
#         evidence = tools["query_engine"].search_graph(query)
#         if evidence:
#             st.code(evidence, language="text")
#         else:
#             st.write("No paths found for this query.")

# # --- FOOTER ---
# st.divider()
# st.write("🔗 [Neo4j Aura Console](https://console.neo4j.io/) | Model: llama-3.1-8b-instant")

