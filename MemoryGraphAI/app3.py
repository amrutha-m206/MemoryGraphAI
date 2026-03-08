import streamlit as st
import os
import shutil
import time
import pandas as pd # For clean metrics table

# Import all your custom modules
from ingestion import DocumentIngestion
from extraction import InformationExtractor
from graph_builder import MemoryGraphBuilder
from graph_embeddings import MemoryGraphEmbedder 
from query_engine import GraphQueryEngine

# Set up page
st.set_page_config(page_title="MemoryGraph AI Eval", layout="wide", page_icon="🧠")
st.title("MemoryGraph AI: A Dynamic, Context-Aware Hybrid Knowledge Graph Framework for Documents")

# Initialize session state for metrics
if 'metrics' not in st.session_state:
    st.session_state.metrics = []

# Initialize classes
@st.cache_resource
def load_tools():
    return {
        "ingestor": DocumentIngestion(),
        "extractor": InformationExtractor(),
        "query_engine": GraphQueryEngine()
    }

tools = load_tools()

# --- SIDEBAR: Processing & Controls ---
with st.sidebar:
    st.header("System Controls")
    clear_db = st.checkbox("Wipe Neo4j (Fresh Start)")
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
    process_btn = st.button("Build & Index Graph")
    
    if process_btn and uploaded_files:
        overall_start = time.time()
        
        with st.status("Pipeline Execution...", expanded=True) as status:
            # 0. CLEANUP
            if os.path.exists("data"): shutil.rmtree("data")
            os.makedirs("data", exist_ok=True)
            for f in uploaded_files:
                with open(os.path.join("data", f.name), "wb") as b: b.write(f.read())

            # 1. INGESTION TIMER
            t0 = time.time()
            docs = tools["ingestor"].process_folder("./data")
            ingest_time = time.time() - t0
            st.write(f"Ingestion: {ingest_time:.2f}s")

            # 2. EXTRACTION & BUILDING TIMER
            t1 = time.time()
            builder = MemoryGraphBuilder()
            if clear_db: builder.clear_database()
            
            total_entities = 0
            for doc in docs:
                st.write(f"Extracting: {doc['filename']}...")
                graph_data = tools["extractor"].extract(doc['content'], max_chunks=5)
                total_entities += len(graph_data.get('entities', []))
                builder.build_graph(graph_data)
            builder.close()
            extract_time = time.time() - t1
            st.write(f"Extraction/Graph: {extract_time:.2f}s")

            # 3. EMBEDDING TIMER
            t2 = time.time()
            embedder = MemoryGraphEmbedder()
            embedder.generate_and_store_embeddings()
            embedder.close()
            embed_time = time.time() - t2
            st.write(f"Vector Indexing: {embed_time:.2f}s")

            # Store metrics for evaluation
            st.session_state.metrics.append({
                "Timestamp": time.strftime("%H:%M:%S"),
                "Files": len(uploaded_files),
                # "Entities": total_entities,
                "Ingest(s)": round(ingest_time, 2),
                "Extract(s)": round(extract_time, 2),
                "Embed(s)": round(embed_time, 2),
                "Total(s)": round(time.time() - overall_start, 2)
            })
            status.update(label="Graph Ready!", state="complete")

# --- MAIN AREA: REASONING & ACCURACY EVALUATION ---
# st.header("Graph Reasoning & Accuracy")
query = st.text_input("Ask a question:", placeholder="e.g. Compare the methods used in these papers.")

if query:
    # Query Latency Timer
    q_start = time.time()
    
    with st.spinner("Traversing Graph..."):
        # 1. Get the Evidence (The "Why")
        evidence = tools["query_engine"].search_graph(query)
        
        # 2. Get the Answer (The "What")
        answer = tools["query_engine"].answer_question(query)
        
    q_end = time.time()
    q_latency = q_end - q_start

    # LAYOUT FOR ACCURACY EVALUATION
    col_ans, col_evid = st.columns([3, 2])
    
    with col_ans:
        st.subheader("Answer")
        st.info(answer)
        st.caption(f"Latency: {q_latency:.2f} seconds")

    with col_evid:
        st.subheader("Graph Evidence")
        st.write("Does the answer match these extracted facts?")
        if evidence:
            st.code(evidence, language="text")
            # Quality Check tool
            st.checkbox("Mark Answer as Accurate", key=f"acc_{time.time()}")
        else:
            st.error("No evidence found in graph for this query.")

# --- EVALUATION DASHBOARD ---
st.divider()
st.subheader("Performance & Accuracy Metrics")
if st.session_state.metrics:
    df = pd.DataFrame(st.session_state.metrics)
    
    # KPIs in columns
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Avg Ingest Time", f"{df['Ingest(s)'].mean():.2f}s")
    kpi2.metric("Avg Extract Time", f"{df['Extract(s)'].mean():.2f}s")
    # kpi3.metric("Total Nodes Extracted", f"{df['Entities'].sum()}")
    kpi4.metric("Last Query Speed", f"{q_latency:.2f}s" if 'q_latency' in locals() else "N/A")
    
    st.dataframe(df, use_container_width=True)
else:
    st.info("Upload and process files to see performance analytics.")

# # --- FOOTER ---
st.divider()
st.write("🔗 [Neo4j Aura Console](https://console.neo4j.io/)")