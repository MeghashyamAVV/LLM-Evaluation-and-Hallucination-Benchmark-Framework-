import json
import re
import time
import math
from dataclasses import dataclass
from collections import Counter
from typing import Optional
import requests

OLLAMA_BASE  = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","of","and","or","but",
    "for","with","as","by","from","be","are","was","were","has","have",
    "had","not","do","does","did","this","that","these","those","i","you",
    "he","she","we","they","what","which","who","when","where","how","all",
    "its","their","there","than","then","so","if","can","will","would",
    "could","should","may","might","also","been","being","about","more"
}

@dataclass
class EvalSample:
    question: str
    context: str
    ground_truth: str


@dataclass
class StrategyResult:
    strategy_name: str
    question: str
    context: str
    ground_truth: str
    generated_answer: str
    faithfulness_score: float
    answer_relevancy_score: float
    hallucination_score: float
    latency_ms: float
    raw_prompt: str = ""

STRATEGIES = {
    "zero_shot": {
        "name": "Zero-shot (baseline)",
        "description": "No context, no instructions. Raw question to model.",
        "build_prompt": lambda q, ctx: f"Answer this question concisely:\n\n{q}"
    },
    "few_shot": {
        "name": "Few-shot examples",
        "description": "2 examples of Q&A before the real question.",
        "build_prompt": lambda q, ctx: (
            "Here are examples of good concise answers:\n\n"
            "Q: What is photosynthesis?\n"
            "A: Photosynthesis is the process by which plants convert sunlight "
            "into food using carbon dioxide and water.\n\n"
            "Q: What causes thunder?\n"
            "A: Thunder is caused by the rapid expansion of air heated by lightning.\n\n"
            f"Q: {q}\nA:"
        )
    },
    "rag": {
        "name": "RAG (context provided)",
        "description": "Relevant context injected before the question.",
        "build_prompt": lambda q, ctx: (
            f"Use ONLY the following context to answer. "
            f"Do not add outside knowledge.\n\n"
            f"Context:\n{ctx}\n\n"
            f"Question: {q}\n\nAnswer:"
        )
    },
    "chain_of_thought": {
        "name": "Chain-of-thought",
        "description": "Model asked to reason step-by-step before answering.",
        "build_prompt": lambda q, ctx: (
            f"Context:\n{ctx}\n\n"
            f"Question: {q}\n\n"
            f"Think step by step, then give a concise final answer.\n"
            f"Reasoning:"
        )
    },
    "rag_structured": {
        "name": "RAG + structured reasoning",
        "description": "Context + explicit 4-step reasoning structure.",
        "build_prompt": lambda q, ctx: (
            f"You are a precise Q&A assistant. Follow these steps exactly:\n"
            f"1. Read the context\n"
            f"2. Identify what the question asks\n"
            f"3. Find the answer ONLY within the context\n"
            f"4. If not in context, say 'Not found in context'\n\n"
            f"Context:\n{ctx}\n\n"
            f"Question: {q}\n\n"
            f"Final answer (1-2 sentences only):"
        )
    }
}

def tokenize(text: str) -> list[str]:
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def keyword_overlap(text_a: str, text_b: str) -> float:
    """Jaccard similarity between keyword sets of two texts."""
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)

def tfidf_cosine(text_a: str, text_b: str, corpus: list[str]) -> float:
    """
    Simple TF-IDF cosine similarity between two texts given a corpus.
    Used for faithfulness scoring.
    """
    def tf(tokens):
        count = Counter(tokens)
        total = max(len(tokens), 1)
        return {w: c / total for w, c in count.items()}

    def idf(word, docs):
        n_containing = sum(1 for d in docs if word in d)
        return math.log((len(docs) + 1) / (n_containing + 1)) + 1

    all_docs = [tokenize(t) for t in corpus + [text_a, text_b]]
    vocab = set(w for d in all_docs for w in d)

    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    tf_a = tf(tokens_a)
    tf_b = tf(tokens_b)

    vec_a, vec_b = [], []
    for word in vocab:
        idf_val = idf(word, all_docs)
        vec_a.append(tf_a.get(word, 0) * idf_val)
        vec_b.append(tf_b.get(word, 0) * idf_val)

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a**2 for a in vec_a))
    mag_b = math.sqrt(sum(b**2 for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def contains_hedge_phrases(answer: str) -> bool:
    """Check if answer admits it doesn't know — sign of low hallucination."""
    hedges = [
        "not found in context", "not mentioned", "not provided",
        "context does not", "cannot determine", "no information",
        "not stated", "does not mention", "i don't know", "unclear"
    ]
    lower = answer.lower()
    return any(h in lower for h in hedges)


def extract_final_answer(answer: str, strategy_name: str = "") -> str:
    """
    For chain-of-thought: extract only the conclusion, not reasoning steps.
    Reasoning steps contain transition words not in context that unfairly
    inflate the hallucination score.
    """
    if "chain" in strategy_name.lower() or "thought" in strategy_name.lower():
        for marker in ["therefore", "conclusion:", "final answer:", "answer:",
                       "finally,", "thus,", "in conclusion", "so,"]:
            idx = answer.lower().rfind(marker)
            if idx != -1:
                extracted = answer[idx:].strip()
                if len(extracted.split()) > 3:
                    return extracted
        # Fallback: last 2 sentences
        sentences = [s.strip() for s in re.split(r'[.!?]', answer)
                     if len(s.strip()) > 10]
        if sentences:
            return ". ".join(sentences[-2:])
    return answer


def score_faithfulness(answer: str, context: str) -> float:
    """
    Faithfulness: how much of the answer's content overlaps with the context?
    High overlap = answer stays within context = faithful.
    """
    if not answer or answer.startswith("ERROR"):
        return 0.0
    # Reward hedging (model admits it doesn't know) as perfectly faithful
    if contains_hedge_phrases(answer):
        return 1.0
    overlap = keyword_overlap(answer, context)
    # Scale: overlap of 0.3+ is very faithful for Q&A
    return min(overlap * 2.8, 1.0)


def score_answer_relevancy(answer: str, question: str) -> float:
    """
    Relevancy: how well does the answer address the question?
    Measured by keyword overlap between answer and question.
    """
    if not answer or answer.startswith("ERROR"):
        return 0.0
    # Short non-answers score low
    if len(answer.split()) < 3:
        return 0.1
    overlap = keyword_overlap(answer, question)
    return min(overlap * 3.5, 1.0)


def score_hallucination(answer: str, context: str, ground_truth: str) -> float:
    """
    Hallucination: how much of the answer is NOT grounded in context or ground truth?
    Lower is better.
    """
    if not answer or answer.startswith("ERROR"):
        return 1.0
    if contains_hedge_phrases(answer):
        return 0.0

    answer_tokens  = set(tokenize(answer))
    context_tokens = set(tokenize(context))
    truth_tokens   = set(tokenize(ground_truth))
    grounded_tokens = context_tokens | truth_tokens

    if not answer_tokens:
        return 0.5

    # What fraction of answer words appear nowhere in context or ground truth
    ungrounded = answer_tokens - grounded_tokens
    ungrounded_ratio = len(ungrounded) / len(answer_tokens)

    # Penalize extra: long answers with many unknown words hallucinate more
    length_penalty = min(len(answer.split()) / 120, 0.25)
    return min(ungrounded_ratio * 1.4 + length_penalty, 1.0)

def call_ollama(prompt: str, temperature: float = 0.1) -> tuple[str, float]:
    start = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": 200}
            },
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.time() - start) * 1000
        return data.get("response", "").strip(), latency_ms
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        return f"ERROR: {str(e)}", latency_ms

def run_evaluation(samples: list[EvalSample]) -> list[StrategyResult]:
    all_results = []
    total = len(samples) * len(STRATEGIES)
    done  = 0

    # Corpus for TF-IDF (all contexts)
    corpus = [s.context for s in samples]

    for strategy_key, strategy in STRATEGIES.items():
        print(f"\n{'='*55}")
        print(f"  Strategy: {strategy['name']}")
        print(f"{'='*55}")

        for i, sample in enumerate(samples):
            done += 1
            print(f"  [{done}/{total}] Q{i+1}: {sample.question[:50]}...")

            prompt = strategy["build_prompt"](sample.question, sample.context)
            answer, latency = call_ollama(prompt)
            print(f"          → {answer[:75]}...")

            # For CoT: score only the final answer, not the reasoning steps
            answer_for_scoring = extract_final_answer(answer, strategy["name"])

            faith  = score_faithfulness(answer_for_scoring, sample.context)
            relev  = score_answer_relevancy(answer_for_scoring, sample.question)
            halluc = score_hallucination(answer_for_scoring, sample.context, sample.ground_truth)
            print(f"          faith={faith:.2f}  relev={relev:.2f}  halluc={halluc:.2f}  {latency:.0f}ms")

            all_results.append(StrategyResult(
                strategy_name=strategy["name"],
                question=sample.question,
                context=sample.context,
                ground_truth=sample.ground_truth,
                generated_answer=answer,
                faithfulness_score=faith,
                answer_relevancy_score=relev,
                hallucination_score=halluc,
                latency_ms=latency,
                raw_prompt=prompt
            ))

    return all_results

def aggregate_results(results: list[StrategyResult]) -> dict:
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in results:
        buckets[r.strategy_name].append(r)

    summary = {}
    for name, items in buckets.items():
        n = len(items)
        summary[name] = {
            "avg_faithfulness":     round(sum(r.faithfulness_score     for r in items) / n, 3),
            "avg_answer_relevancy": round(sum(r.answer_relevancy_score for r in items) / n, 3),
            "avg_hallucination":    round(sum(r.hallucination_score    for r in items) / n, 3),
            "avg_latency_ms":       round(sum(r.latency_ms             for r in items) / n, 1),
            "sample_count":         n
        }
    return summary


def save_results(results: list[StrategyResult], path: str = "eval_results.json"):
    data = [
        {
            "strategy":         r.strategy_name,
            "question":         r.question,
            "ground_truth":     r.ground_truth,
            "generated_answer": r.generated_answer,
            "faithfulness":     r.faithfulness_score,
            "answer_relevancy": r.answer_relevancy_score,
            "hallucination":    r.hallucination_score,
            "latency_ms":       r.latency_ms,
        }
        for r in results
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to {path}")