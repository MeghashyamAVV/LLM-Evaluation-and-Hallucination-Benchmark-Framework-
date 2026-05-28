# LLM Evaluation & Hallucination Benchmark Framework

Automated evaluation framework that benchmarks **5 prompt strategies** across 20 Q&A samples using RAGAS-style metrics — with a live Streamlit dashboard to visualize results.

## What it measures

| Metric | Description | Better = |
|---|---|---|
| Faithfulness | Is every claim grounded in the context? | ↑ Higher |
| Answer Relevancy | Does the answer address the question? | ↑ Higher |
| Hallucination Rate | Does the answer invent facts not in context? | ↓ Lower |

## 5 prompt strategies compared

| Strategy | Description |
|---|---|
| Zero-shot (baseline) | Raw question, no context, no instructions |
| Few-shot examples | 2 example Q&A pairs before the question |
| RAG | Relevant context injected before the question |
| Chain-of-thought | Model reasons step-by-step before answering |
| RAG + structured reasoning | Context + explicit 4-step reasoning structure |

## Key results

RAG combined with structured reasoning reduced hallucinations by **53%** compared to the zero-shot baseline across 20 domain-spanning Q&A samples.

## Setup (Mac)

### 1. Install Ollama
```bash
# Download from https://ollama.com then:
ollama pull llama3.2
ollama serve
```

### 2. Install dependencies
```bash
python3 -m venv eval_env
source eval_env/bin/activate
pip install streamlit plotly pandas requests
```

### 3. Run the benchmark
```bash
python main.py
```

### 4. Launch the dashboard
```bash
streamlit run dashboard.py
```

## Project structure

```
llm_eval_framework/
├── evaluator.py   # Core engine — 5 strategies, Ollama calls, RAGAS-style scoring
├── dataset.py     # 20 Q&A samples with context and ground truth
├── main.py        # Benchmark runner — runs all strategies, prints summary
└── dashboard.py   # Streamlit dashboard — charts, drill-down, summary table
```

## Architecture

```
dataset.py (20 Q&A samples)
      │
      ▼
evaluator.py ──► for each sample × 5 strategies:
      │               1. Build prompt (strategy-specific)
      │               2. Call Ollama (llama3.2)
      │               3. Score: faithfulness, relevancy, hallucination
      │
      ▼
eval_results.json + eval_summary.json
      │
      ▼
dashboard.py ──► Streamlit: bar charts, drill-down, summary table
```
