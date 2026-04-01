import streamlit as st
import os
import shutil
import time
import pandas as pd

# Import custom modules
from ingestion import DocumentIngestion
from extraction import InformationExtractor
from graph_builder import MemoryGraphBuilder
from graph_embeddings import MemoryGraphEmbedder 
from query_engine import GraphQueryEngine
from streamlit_agraph import agraph, Node, Edge, Config

# Page Configuration
st.set_page_config(page_title="MemoryGraph AI", layout="wide", page_icon="🧠")
st.title("MemoryGraph AI: Context-Aware Hybrid Knowledge Graph")

# Initialize session state for metrics
if 'metrics' not in st.session_state:
    st.session_state.metrics = []

@st.cache_resource
def load_tools():
    return {
        "ingestor": DocumentIngestion(),
        "extractor": InformationExtractor(),
        "query_engine": GraphQueryEngine()
    }

tools = load_tools()

# --- SIDEBAR: Processing ---
with st.sidebar:
    st.header("⚙️ System Controls")
    clear_db = st.checkbox("Wipe Neo4j (Fresh Start)")
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
    process_btn = st.button("Build & Index Graph")
    
    if process_btn and uploaded_files:
        overall_start = time.time()
        with st.status("Pipeline Processing...", expanded=True) as status:
            if os.path.exists("data"): shutil.rmtree("data")
            os.makedirs("data", exist_ok=True)
            for f in uploaded_files:
                with open(os.path.join("data", f.name), "wb") as b: b.write(f.read())

            t0 = time.time()
            docs = tools["ingestor"].process_folder("./data")
            ingest_time = time.time() - t0

            t1 = time.time()
            builder = MemoryGraphBuilder()
            if clear_db: builder.clear_database()
            for doc in docs:
                st.write(f"Extracting: {doc['filename']}...")
                graph_data = tools["extractor"].extract(doc['content'], max_chunks=5)
                builder.build_graph(graph_data)
            builder.close()
            extract_time = time.time() - t1

            t2 = time.time()
            embedder = MemoryGraphEmbedder()
            embedder.generate_and_store_embeddings()
            embedder.close()
            embed_time = time.time() - t2

            st.session_state.metrics.append({
                "Timestamp": time.strftime("%H:%M:%S"),
                "Files": len(uploaded_files),
                "Ingest(s)": round(ingest_time, 2),
                "Extract(s)": round(extract_time, 2),
                "Embed(s)": round(embed_time, 2),
                "Total(s)": round(time.time() - overall_start, 2)
            })
            status.update(label="Graph Ready!", state="complete")

# --- MAIN AREA TABS ---
tab1, tab2 = st.tabs(["🔍 Search & Reasoning", "📊 Knowledge Discovery"])

with tab1:
    query = st.text_input("Ask a question:", placeholder="e.g. What are the key findings?")
    if query:
        q_start = time.time()
        with st.spinner("Analyzing Memory Graph..."):
            answer = tools["query_engine"].answer_question(query)
            node_names, edge_data = tools["query_engine"].get_visualization_data(query)
            evidence = tools["query_engine"].search_graph(query)
        q_latency = time.time() - q_start

        col_ans, col_vis = st.columns([1, 1])
        with col_ans:
            st.subheader("🤖 AI Answer")
            st.info(answer)
            st.caption(f"Latency: {q_latency:.2f}s")
            with st.expander("📄 View Knowledge Context"):
                st.code(evidence)
        with col_vis:
            st.subheader("🌐 Visual View")
            if node_names:
                nodes = [Node(id=n, label=n, size=20, color="#007bff") for n in node_names]
                edges = [Edge(source=e['source'], target=e['target'], label=e['label']) for e in edge_data]
                config = Config(width=600, height=450, directed=True, physics=True)
                agraph(nodes=nodes, edges=edges, config=config)

with tab2:
    st.header("📈 Graph Data Science Insights")
    st.write("Analyze the global structure of your Knowledge Graph to find hidden patterns.")
    
    if st.button("Run Global Analytics"):
        with st.spinner("Calculating Graph Metrics..."):
            hubs, communities = tools["query_engine"].get_graph_analytics()
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🔝 Hub Entities")
                st.table(pd.DataFrame(hubs))
            with c2:
                st.subheader("🤝 Concept Clusters")
                st.table(pd.DataFrame(communities))

# --- PERFORMANCE ---
st.divider()
st.subheader("📊 Performance History")
if st.session_state.metrics:
    st.dataframe(pd.DataFrame(st.session_state.metrics), use_container_width=True)
