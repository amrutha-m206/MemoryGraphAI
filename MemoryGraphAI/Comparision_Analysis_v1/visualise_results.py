"""
visualise_results.py
--------------------
Streamlit dashboard for comparison_results.json.

Run after run_comparison.py:
    streamlit run visualise_results.py
"""

import json
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="MemoryGraph RAG vs Vector RAG — Evaluation v2",
    layout="wide",
    page_icon="📊",
)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "comparison_results.json")


@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


data = load_results()

st.title("📊 MemoryGraph RAG vs Vector RAG — Retrieval Evaluation (v2)")

if data is None:
    st.error("**comparison_results.json not found.** Run `python run_comparison.py` first.")
    st.stop()

macro = data["macro_averages"]
per_query = data["per_query_results"]

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Macro-average summary
# ─────────────────────────────────────────────────────────────────────────────
st.header("1 — Macro-Average Summary")

rows = []
for metric in macro["graph_rag"]:
    gv = macro["graph_rag"][metric]
    vv = macro["vector_rag"][metric]
    if metric == "Latency(s)":
        winner = "🟢 Graph RAG" if gv <= vv else "🟠 Vector RAG"
    else:
        winner = "🟢 Graph RAG" if gv >= vv else "🟠 Vector RAG"
        if abs(gv - vv) < 0.001:
            winner = "🔵 Tie"
    rows.append({"Metric": metric, "Graph RAG": round(gv, 4), "Vector RAG": round(vv, 4), "Winner": winner})

df_summary = pd.DataFrame(rows)
st.dataframe(df_summary, use_container_width=True, hide_index=True)

graph_wins = sum(
    1 for r in rows
    if "Graph RAG" in r["Winner"] and r["Metric"] != "Latency(s)"
)
st.metric("Graph RAG wins on quality metrics", f"{graph_wins} / 9")

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Radar chart
# ─────────────────────────────────────────────────────────────────────────────
st.header("2 — Radar Chart (All Retrieval Metrics)")

radar_metrics = [
    "Precision@K", "Recall@K", "F1@K", "MRR",
    "MAP", "nDCG@K", "HitRate@K",
    "ContextRelevance", "AnswerFaithfulness",
]
graph_vals = [macro["graph_rag"].get(m, 0) for m in radar_metrics]
vector_vals = [macro["vector_rag"].get(m, 0) for m in radar_metrics]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=graph_vals + [graph_vals[0]],
    theta=radar_metrics + [radar_metrics[0]],
    fill="toself", name="Graph RAG",
    line_color="#007bff", fillcolor="rgba(0,123,255,0.15)",
))
fig_radar.add_trace(go.Scatterpolar(
    r=vector_vals + [vector_vals[0]],
    theta=radar_metrics + [radar_metrics[0]],
    fill="toself", name="Vector RAG",
    line_color="#ff7f0e", fillcolor="rgba(255,127,14,0.15)",
))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True, height=500,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────
st.header("3 — Grouped Bar Chart")

df_bar = pd.DataFrame({
    "Metric": radar_metrics * 2,
    "Score": graph_vals + vector_vals,
    "System": ["Graph RAG"] * len(radar_metrics) + ["Vector RAG"] * len(radar_metrics),
})
fig_bar = px.bar(
    df_bar, x="Metric", y="Score", color="System", barmode="group",
    color_discrete_map={"Graph RAG": "#007bff", "Vector RAG": "#ff7f0e"},
    range_y=[0, 1],
)
fig_bar.update_layout(height=420)
st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Context comparison (KEY DIFFERENTIATOR)
# ─────────────────────────────────────────────────────────────────────────────
st.header("4 — Context Quality: What Each System Passes to the LLM")
st.markdown(
    """
    This is the most important section. It shows the **actual context** each system
    sends to the LLM for the same query.  
    - **Vector RAG**: flat list of entity names  
    - **Graph RAG**: structured triples like `(A) -[RELATIONSHIP]-> (B)`  
    
    The relational structure is what enables the LLM to reason about *how* entities
    are connected — this is the core advantage of the graph approach.
    """
)

query_labels = [f"Q{i+1}: {r['query'][:60]}…" for i, r in enumerate(per_query)]
selected_q = st.selectbox("Select a query", query_labels)
q_idx = query_labels.index(selected_q)
q_data = per_query[q_idx]

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("🔵 Graph RAG Context (Relational Triples)")
    st.code(q_data.get("graph_context_snippet", ""), language="text")
    st.caption(f"Triples in full context: {q_data.get('graph_triples_count', 'N/A')}")
    st.subheader("Graph RAG Answer")
    st.info(q_data["graph_answer"])

with col_b:
    st.subheader("🟠 Vector RAG Context (Flat Entity Names)")
    st.code(q_data.get("vector_context_snippet", ""), language="text")
    st.caption("No relationship information — just names and cosine scores")
    st.subheader("Vector RAG Answer")
    st.info(q_data["vector_answer"])

# Per-query metric table
metric_rows = [
    {"Metric": m, "Graph RAG": q_data["graph_metrics"][m], "Vector RAG": q_data["vector_metrics"][m]}
    for m in q_data["graph_metrics"]
]
st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Context Relevance & Answer Faithfulness — the key semantic scores
# ─────────────────────────────────────────────────────────────────────────────
st.header("5 — Semantic Scores Across All Queries")

fig_sem = go.Figure()
queries_short = [f"Q{i+1}" for i in range(len(per_query))]

for system, color in [("graph_metrics", "#007bff"), ("vector_metrics", "#ff7f0e")]:
    label = "Graph RAG" if "graph" in system else "Vector RAG"
    fig_sem.add_trace(go.Bar(
        name=f"{label} — ContextRelevance",
        x=queries_short,
        y=[r[system]["ContextRelevance"] for r in per_query],
        marker_color=color, opacity=0.9,
        legendgroup=label,
    ))
fig_sem.update_layout(barmode="group", height=350, yaxis_range=[0, 1])
st.plotly_chart(fig_sem, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Latency
# ─────────────────────────────────────────────────────────────────────────────
st.header("6 — Query Latency Comparison")
lat_df = pd.DataFrame({
    "Query": queries_short,
    "Graph RAG (s)": [r["graph_metrics"]["Latency(s)"] for r in per_query],
    "Vector RAG (s)": [r["vector_metrics"]["Latency(s)"] for r in per_query],
})
fig_lat = px.line(
    lat_df.melt(id_vars="Query", var_name="System", value_name="Latency (s)"),
    x="Query", y="Latency (s)", color="System", markers=True,
    color_discrete_map={"Graph RAG (s)": "#007bff", "Vector RAG (s)": "#ff7f0e"},
)
fig_lat.update_layout(height=320)
st.plotly_chart(fig_lat, use_container_width=True)
st.caption("Higher latency for Graph RAG is the expected trade-off for richer relational context.")

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Interpretation guide
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("7 — Interpretation Guide")
st.markdown("""
| Metric | What it measures | Why Graph RAG should win |
|---|---|---|
| **Precision@K** | Fraction of retrieved entities that are relevant | Graph traversal adds topically-connected neighbours, not random entities |
| **Recall@K** | Fraction of relevant entities retrieved | Expanding seeds to neighbours surfaces entities that are too far from the query in embedding space |
| **F1@K** | Balance of both | Graph expansion finds more relevant entities without destroying precision |
| **MRR** | How early a relevant entity appears | Semantic seeds are already relevant; graph places connected relevant nodes right after |
| **MAP** | Full area under P/R curve | Relationship chains enable finding all relevant entities at multiple hops |
| **nDCG@K** | Rewards early relevant hits | Seeds (most semantically relevant) come first; neighbours follow |
| **HitRate@K** | Did we find anything at all? | Baseline; both systems should tie here |
| **ContextRelevance** | ⭐ Is the context semantically on-topic? | Relational triples are more specific than entity names alone — the single most important metric |
| **AnswerFaithfulness** | Is the generated answer correct? | Structured triples give the LLM relationship evidence, reducing hallucination |
| **Latency** | Speed | Vector RAG wins here — single ANN scan is faster than traversal |

### Key insight: the Context Relevance gap
If Graph RAG's `ContextRelevance` is ≥ 0.05 higher than Vector RAG's, this directly
proves that structured graph context is more informative for answering questions — 
even when the ranking metrics are similar.

### Why the original run showed Vector RAG winning
The original `run_comparison.py` used `GraphQueryEngine` which did the same ANN
search as Vector RAG but did **not** pass relational triples to the LLM — it passed
the same format of entity names. The two systems were functionally identical in terms
of context content, so the graph added latency with no benefit. This version fixes
that by using `(A)-[REL]->(B)` triples as the graph context.
""")