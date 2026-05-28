import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="LLM Eval Dashboard",
    page_icon="🧪",
    layout="wide"
)

@st.cache_data
def load_data():
    if not os.path.exists("eval_results.json"):
        return None, None
    with open("eval_results.json") as f:
        results = json.load(f)
    with open("eval_summary.json") as f:
        summary = json.load(f)
    return results, summary


results, summary = load_data()

st.title("🧪 LLM Evaluation & Hallucination Benchmark")
st.caption("Comparing 5 prompt strategies across 20 Q&A samples using RAGAS-style metrics")

if results is None:
    st.error("No results found. Run `python main.py` first to generate eval_results.json")
    st.stop()

df = pd.DataFrame(results)
strategy_order = [
    "Zero-shot (baseline)",
    "Few-shot examples",
    "RAG (context provided)",
    "Chain-of-thought",
    "RAG + structured reasoning"
]

st.markdown("### Overall results")
col1, col2, col3, col4 = st.columns(4)

best_strategy = min(summary.items(), key=lambda x: x[1]["avg_hallucination"])
worst_halluc  = max(s["avg_hallucination"] for s in summary.values())
best_halluc   = min(s["avg_hallucination"] for s in summary.values())
reduction_pct = round((worst_halluc - best_halluc) / max(worst_halluc, 0.001) * 100)

with col1:
    st.metric("Strategies tested", len(summary))
with col2:
    st.metric("Total evaluations", len(results))
with col3:
    st.metric("Best strategy", best_strategy[0].split("(")[0].strip())
with col4:
    st.metric("Hallucination reduction", f"{reduction_pct}%",
              delta=f"vs zero-shot baseline", delta_color="normal")

st.divider()

st.markdown("### Hallucination rate by strategy")
st.caption("Lower is better — measures how often answers contain facts not in the context")

halluc_data = {
    "Strategy": [],
    "Hallucination Rate": [],
    "Color": []
}
for s_name in strategy_order:
    if s_name in summary:
        halluc_data["Strategy"].append(s_name)
        halluc_data["Hallucination Rate"].append(summary[s_name]["avg_hallucination"])
        halluc_data["Color"].append(
            "#E24B4A" if summary[s_name]["avg_hallucination"] == worst_halluc
            else ("#1D9E75" if summary[s_name]["avg_hallucination"] == best_halluc else "#378ADD")
        )

fig_halluc = go.Figure(go.Bar(
    x=halluc_data["Strategy"],
    y=halluc_data["Hallucination Rate"],
    marker_color=halluc_data["Color"],
    text=[f"{v:.2f}" for v in halluc_data["Hallucination Rate"]],
    textposition="outside"
))
fig_halluc.update_layout(
    yaxis=dict(range=[0, 1], title="Avg hallucination score"),
    xaxis=dict(title=""),
    height=350,
    margin=dict(t=20, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig_halluc, use_container_width=True)

st.markdown("### All metrics by strategy")
st.caption("Faithfulness ↑ and Relevancy ↑ are good. Hallucination ↓ is good.")

metrics_rows = []
for s_name in strategy_order:
    if s_name in summary:
        m = summary[s_name]
        metrics_rows.append({
            "Strategy": s_name,
            "Faithfulness ↑": m["avg_faithfulness"],
            "Relevancy ↑":    m["avg_answer_relevancy"],
            "Hallucination ↓": m["avg_hallucination"]
        })

df_metrics = pd.DataFrame(metrics_rows)
fig_multi = go.Figure()
colors = {"Faithfulness ↑": "#1D9E75", "Relevancy ↑": "#378ADD", "Hallucination ↓": "#E24B4A"}
for metric, color in colors.items():
    fig_multi.add_trace(go.Bar(
        name=metric,
        x=df_metrics["Strategy"],
        y=df_metrics[metric],
        marker_color=color,
        text=[f"{v:.2f}" for v in df_metrics[metric]],
        textposition="outside"
    ))
fig_multi.update_layout(
    barmode="group",
    yaxis=dict(range=[0, 1.1], title="Score"),
    height=400,
    margin=dict(t=20, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig_multi, use_container_width=True)

st.markdown("### Response latency by strategy")
st.caption("Time taken to generate answers (milliseconds)")

latency_data = {
    s: summary[s]["avg_latency_ms"]
    for s in strategy_order if s in summary
}
fig_lat = go.Figure(go.Bar(
    x=list(latency_data.keys()),
    y=list(latency_data.values()),
    marker_color="#888780",
    text=[f"{v:.0f}ms" for v in latency_data.values()],
    textposition="outside"
))
fig_lat.update_layout(
    yaxis=dict(title="Latency (ms)"),
    height=300,
    margin=dict(t=20, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig_lat, use_container_width=True)

st.divider()

st.markdown("### Drill down — per question results")

selected_strategy = st.selectbox(
    "Select strategy to inspect:",
    options=strategy_order,
    index=4
)

filtered = df[df["strategy"] == selected_strategy].reset_index(drop=True)

if len(filtered) == 0:
    st.warning("No results for this strategy yet.")
else:
    for i, row in filtered.iterrows():
        with st.expander(f"Q{i+1}: {row['question']}"):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(f"**Generated answer:**")
                st.write(row["generated_answer"])
                st.markdown(f"**Ground truth:**")
                st.write(row["ground_truth"])
            with col_b:
                st.metric("Faithfulness",      f"{row['faithfulness']:.2f}")
                st.metric("Relevancy",          f"{row['answer_relevancy']:.2f}")
                st.metric("Hallucination",      f"{row['hallucination']:.2f}",
                          delta="lower is better", delta_color="off")
                st.metric("Latency",            f"{row['latency_ms']:.0f}ms")

st.divider()

st.markdown("### Summary table")
summary_rows = []
for s_name in strategy_order:
    if s_name in summary:
        m = summary[s_name]
        summary_rows.append({
            "Strategy":         s_name,
            "Faithfulness ↑":   f"{m['avg_faithfulness']:.3f}",
            "Relevancy ↑":      f"{m['avg_answer_relevancy']:.3f}",
            "Hallucination ↓":  f"{m['avg_hallucination']:.3f}",
            "Avg Latency":      f"{m['avg_latency_ms']:.0f}ms",
            "Samples":          m["sample_count"]
        })
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.caption(f"Model: llama3.2 via Ollama · {len(results)} total evaluations · RAGAS-style LLM-judge metrics")