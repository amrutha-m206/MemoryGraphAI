"""
visualise_results.py
--------------------
A Streamlit dashboard that loads comparison_results.json and renders
interactive charts comparing Graph RAG vs Vector RAG performance.

Run after run_comparison.py has produced comparison_results.json:

    streamlit run visualise_results.py
"""

import json
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="MemoryGraph RAG vs Vector RAG — Evaluation",
    layout="wide",
    page_icon="📊",
)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "comparison_results.json")

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


data = load_results()

st.title("📊 MemoryGraph RAG vs Vector RAG — Retrieval Evaluation")

if data is None:
    st.error(
        "**comparison_results.json not found.**  "
        "Please run `python run_comparison.py` first to generate the evaluation data."
    )
    st.stop()

macro = data["macro_averages"]
per_query = data["per_query_results"]

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Macro-average summary table
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
    rows.append(
        {
            "Metric": metric,
            "Graph RAG": round(gv, 4),
            "Vector RAG": round(vv, 4),
            "Winner": winner,
        }
    )

df_summary = pd.DataFrame(rows)
st.dataframe(df_summary, use_container_width=True, hide_index=True)

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
fig_radar.add_trace(
    go.Scatterpolar(
        r=graph_vals + [graph_vals[0]],
        theta=radar_metrics + [radar_metrics[0]],
        fill="toself",
        name="Graph RAG",
        line_color="#007bff",
        fillcolor="rgba(0,123,255,0.15)",
    )
)
fig_radar.add_trace(
    go.Scatterpolar(
        r=vector_vals + [vector_vals[0]],
        theta=radar_metrics + [radar_metrics[0]],
        fill="toself",
        name="Vector RAG",
        line_color="#ff7f0e",
        fillcolor="rgba(255,127,14,0.15)",
    )
)
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True,
    height=500,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────

st.header("3 — Grouped Bar Chart")

bar_metrics = [m for m in radar_metrics]  # same set, latency excluded for clarity
df_bar = pd.DataFrame(
    {
        "Metric": bar_metrics * 2,
        "Score": graph_vals + vector_vals,
        "System": ["Graph RAG"] * len(bar_metrics) + ["Vector RAG"] * len(bar_metrics),
    }
)
fig_bar = px.bar(
    df_bar,
    x="Metric",
    y="Score",
    color="System",
    barmode="group",
    color_discrete_map={"Graph RAG": "#007bff", "Vector RAG": "#ff7f0e"},
    range_y=[0, 1],
)
fig_bar.update_layout(height=420)
st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Per-query breakdown
# ─────────────────────────────────────────────────────────────────────────────

st.header("4 — Per-Query Deep Dive")

query_labels = [f"Q{i+1}: {r['query'][:55]}…" for i, r in enumerate(per_query)]
selected_q = st.selectbox("Select a query", query_labels)
q_idx = query_labels.index(selected_q)
q_data = per_query[q_idx]

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Graph RAG Answer")
    st.info(q_data["graph_answer"])
    st.caption("Retrieved entities")
    st.code(", ".join(q_data["graph_retrieved"][:10]))

with col_b:
    st.subheader("Vector RAG Answer")
    st.info(q_data["vector_answer"])
    st.caption("Retrieved entities")
    st.code(", ".join(q_data["vector_retrieved"][:10]))

# Per-query metric comparison
metric_rows = []
for metric in q_data["graph_metrics"]:
    metric_rows.append(
        {
            "Metric": metric,
            "Graph RAG": q_data["graph_metrics"][metric],
            "Vector RAG": q_data["vector_metrics"][metric],
        }
    )
st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Latency comparison
# ─────────────────────────────────────────────────────────────────────────────

st.header("5 — Query Latency Comparison")

lat_data = {
    "Query": [f"Q{i+1}" for i in range(len(per_query))],
    "Graph RAG (s)": [r["graph_metrics"]["Latency(s)"] for r in per_query],
    "Vector RAG (s)": [r["vector_metrics"]["Latency(s)"] for r in per_query],
}
df_lat = pd.DataFrame(lat_data)
fig_lat = px.line(
    df_lat.melt(id_vars="Query", var_name="System", value_name="Latency (s)"),
    x="Query",
    y="Latency (s)",
    color="System",
    markers=True,
    color_discrete_map={"Graph RAG (s)": "#007bff", "Vector RAG (s)": "#ff7f0e"},
)
fig_lat.update_layout(height=350)
st.plotly_chart(fig_lat, use_container_width=True)

st.caption(
    "Lower latency is better. Graph RAG typically has higher latency due to "
    "the additional graph traversal step, which is the trade-off for richer context."
)

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Interpretation guide
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.header("6 — How to Interpret These Results")

st.markdown(
    """
| Metric | What it measures | Graph RAG advantage |
|---|---|---|
| **Precision@K** | Are the retrieved entities actually relevant? | Relationship traversal filters noise |
| **Recall@K** | Were all relevant entities found? | Neighbourhood expansion surfaces linked entities |
| **F1@K** | Balance of precision and recall | Richer context → better balance |
| **MRR** | How early does the first relevant entity appear? | Seed nodes from semantic search tend to be on-topic |
| **MAP** | Area under the precision-recall curve | Graph edges connect semantically distant but topically related concepts |
| **nDCG@K** | Rewards early relevant hits more | Multi-hop paths place related nodes higher |
| **HitRate@K** | Did we find anything relevant at all? | Baseline coverage check |
| **ContextRelevance** | Is the retrieved context on-topic for the answer? | Structured triples carry more signal per token |
| **AnswerFaithfulness** | Is the generated answer correct? | Factual triples ground the LLM, reducing hallucination |
| **Latency** | How fast is the retrieval+generation? | Vector RAG is usually faster (single index scan) |

### Reading the verdict
- **Graph RAG wins on most quality metrics** → structured relationships enable multi-hop reasoning  
  that pure similarity search cannot replicate.
- **Vector RAG wins on latency** → a single ANN index scan is faster than graph traversal.
- The **ContextRelevance** and **AnswerFaithfulness** scores directly quantify the  
  *"does knowing relationships improve the answer?"* research question.
"""
)