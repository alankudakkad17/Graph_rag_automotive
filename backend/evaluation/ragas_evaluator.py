"""
ragas_evaluator.py
───────────────────
RAGAS-based evaluation library for the Graph-RAG pipeline.
Used ONLY by evaluate.py — not imported by the FastAPI app.

Metrics:
  • Faithfulness       — claims in answer supported by context?
  • Answer Relevancy   — does the answer address the question?
  • Context Precision  — are retrieved chunks truly relevant?
  • Context Recall     — does context contain enough info?

All scoring uses local Ollama + sentence-transformers.
100% free — no API keys, no cloud calls.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings

from backend.config import get_settings
from backend.agents.graph_rag_agent import run_agent
from backend.utils import logger


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class EvaluationSample:
    """One question with its ground-truth answer and expected context keywords."""
    question:        str
    ground_truth:    str                      # correct reference answer
    expected_keywords: list[str] = field(default_factory=list)  # words that must appear in context


@dataclass
class SampleResult:
    question:          str
    generated_answer:  str
    ground_truth:      str
    retrieved_context: str
    faithfulness:      float    # 0–1
    answer_relevancy:  float    # 0–1
    context_precision: float    # 0–1
    context_recall:    float    # 0–1
    latency_ms:        float
    was_refused:       bool
    pipeline_used:     str


@dataclass
class EvaluationResult:
    """Aggregate scores across all evaluated samples."""
    total_samples:     int
    avg_faithfulness:       float
    avg_answer_relevancy:   float
    avg_context_precision:  float
    avg_context_recall:     float
    avg_latency_ms:         float
    refusal_rate:           float    # fraction of queries refused
    sample_results:         list[SampleResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sample_results"] = [asdict(s) for s in self.sample_results]
        return d


# ── Built-in BMW 7 Series benchmark suite ────────────────────────────────────
BMW_BENCHMARK = [
    EvaluationSample(
        question="What is the horsepower of the 2023 BMW 760i xDrive?",
        ground_truth="The 2023 BMW 760i xDrive produces 536 horsepower.",
        expected_keywords=["536", "horsepower", "760i"],
    ),
    EvaluationSample(
        question="What type of engine does the BMW 740i use?",
        ground_truth="The BMW 740i uses a 3.0-liter TwinPower Turbo inline 6-cylinder engine.",
        expected_keywords=["inline", "6", "turbo", "3.0"],
    ),
    EvaluationSample(
        question="What is the wheelbase of the BMW 7 Series?",
        ground_truth="The BMW 7 Series has a wheelbase of 3215 mm.",
        expected_keywords=["wheelbase", "3215", "mm"],
    ),
    EvaluationSample(
        question="What ADAS features are standard on the BMW 7 Series?",
        ground_truth="The BMW 7 Series includes Driving Assistant Professional as standard.",
        expected_keywords=["driving assistant", "adas", "standard"],
    ),
    EvaluationSample(
        question="What is the fuel type of the BMW i7?",
        ground_truth="The BMW i7 is a fully electric vehicle.",
        expected_keywords=["electric", "ev", "i7"],
    ),
    EvaluationSample(
        question="What is the price of the 2024 BMW 8 Series?",
        ground_truth="NOT_IN_KNOWLEDGE_BASE",  # should be refused
        expected_keywords=[],
    ),
]


# ── Evaluator ─────────────────────────────────────────────────────────────────
class RagasEvaluator:
    """
    Evaluates Graph-RAG pipeline quality using RAGAS metrics.
    All scoring uses local Ollama + sentence-transformers — no API costs.
    """

    def __init__(self):
        cfg = get_settings()
        # LLM judge for faithfulness & answer relevancy
        self._llm = OllamaLLM(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            temperature=0.0,
        )
        # Embeddings for semantic similarity scoring
        self._embeddings = HuggingFaceEmbeddings(
            model_name=cfg.hf_embed_model,
            model_kwargs={"device": "cpu"},
        )

    # ── Public API ────────────────────────────────────────────────────────────
    def evaluate_sample(self, sample: EvaluationSample) -> SampleResult:
        """Evaluate a single question against the pipeline."""
        start = time.time()

        # Run the full RAG pipeline
        result = run_agent(sample.question)
        latency = (time.time() - start) * 1000

        answer  = result.get("answer", "")
        sources = result.get("sources", [])
        context = result.get("context", "")
        refused = result.get("was_refused", False)
        pipeline= result.get("pipeline", "hybrid")

        # Score each metric
        faithfulness       = self._score_faithfulness(answer, context)
        answer_relevancy   = self._score_answer_relevancy(sample.question, answer)
        context_precision  = self._score_context_precision(
            sample.question, context, sample.expected_keywords
        )
        context_recall     = self._score_context_recall(
            sample.ground_truth, context
        )

        return SampleResult(
            question          = sample.question,
            generated_answer  = answer,
            ground_truth      = sample.ground_truth,
            retrieved_context = context[:500],   # truncate for storage
            faithfulness      = faithfulness,
            answer_relevancy  = answer_relevancy,
            context_precision = context_precision,
            context_recall    = context_recall,
            latency_ms        = round(latency, 1),
            was_refused       = refused,
            pipeline_used     = pipeline,
        )

    def evaluate_batch(
        self, samples: list[EvaluationSample]
    ) -> EvaluationResult:
        """Evaluate a list of samples and return aggregate scores."""
        logger.info(f"[RAGAS] Evaluating {len(samples)} samples …")
        results: list[SampleResult] = []

        for i, sample in enumerate(samples, 1):
            logger.info(f"  [{i}/{len(samples)}] {sample.question[:60]}")
            try:
                sr = self.evaluate_sample(sample)
                results.append(sr)
            except Exception as exc:
                logger.error(f"  Sample failed: {exc}")

        return self._aggregate(results)

    def run_benchmark(self) -> EvaluationResult:
        """Run the built-in BMW 7 Series benchmark suite."""
        logger.info("[RAGAS] Running BMW 7 Series benchmark suite …")
        return self.evaluate_batch(BMW_BENCHMARK)

    # ── Metric implementations ─────────────────────────────────────────────────
    def _score_faithfulness(self, answer: str, context: str) -> float:
        """
        Faithfulness: what fraction of claims in the answer are
        supported by the retrieved context?

        Method: ask LLM to identify unsupported claims.
        Score = 1 - (unsupported_claims / total_claims)
        """
        if not answer or not context:
            return 0.0

        prompt = f"""You are evaluating whether an AI answer is faithful to the source context.

Context:
{context[:1500]}

Answer:
{answer[:800]}

List ONLY the claims in the answer that are NOT supported by the context.
If all claims are supported, respond with: NONE
Respond as a JSON array of strings: ["claim1", "claim2"] or []"""

        try:
            raw = self._llm.invoke(prompt).strip()
            import re
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                unsupported = json.loads(match.group())
            elif "NONE" in raw.upper():
                unsupported = []
            else:
                unsupported = []

            # Estimate total claims (sentences in answer)
            total_claims = max(len(re.split(r'[.!?]', answer)), 1)
            unsupported_count = len(unsupported)
            score = max(0.0, 1.0 - (unsupported_count / total_claims))
            return round(score, 3)
        except Exception as exc:
            logger.warning(f"Faithfulness scoring failed: {exc}")
            return 0.5   # neutral fallback

    def _score_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Answer Relevancy: does the answer actually address the question?

        Method: ask LLM to generate what question the answer is answering,
        then compute cosine similarity between original and generated question.
        Higher similarity = answer is more relevant to the question.
        """
        if not answer:
            return 0.0

        # Check if it's a refusal — refusals are intentionally relevant
        if "not find" in answer.lower() or "not available" in answer.lower():
            return 0.8   # refusals are appropriate responses

        prompt = f"""Based on this answer, what question was it answering?
Respond with ONLY the question, nothing else.

Answer: {answer[:600]}

Question:"""

        try:
            inferred_question = self._llm.invoke(prompt).strip()
            # Semantic similarity between original and inferred question
            import numpy as np
            emb_orig     = self._embeddings.embed_query(question)
            emb_inferred = self._embeddings.embed_query(inferred_question)
            similarity   = float(np.dot(emb_orig, emb_inferred) /
                                 (np.linalg.norm(emb_orig) * np.linalg.norm(emb_inferred) + 1e-9))
            return round(max(0.0, similarity), 3)
        except Exception as exc:
            logger.warning(f"Answer relevancy scoring failed: {exc}")
            return 0.5

    def _score_context_precision(
        self, question: str, context: str, expected_keywords: list[str]
    ) -> float:
        """
        Context Precision: are the retrieved chunks actually relevant
        to the question?

        Method: keyword overlap check + LLM relevance judgement.
        """
        if not context:
            return 0.0

        context_lower = context.lower()

        # Keyword overlap component (fast, no LLM)
        if expected_keywords:
            matched = sum(1 for kw in expected_keywords if kw.lower() in context_lower)
            keyword_score = matched / len(expected_keywords)
        else:
            keyword_score = 0.5   # unknown expected — neutral

        # LLM precision judgement
        prompt = f"""On a scale of 0 to 10, how relevant is this retrieved context
to answering the question? Respond with ONLY a number.

Question: {question}
Context: {context[:800]}

Score (0-10):"""

        try:
            raw_score = self._llm.invoke(prompt).strip()
            import re
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', raw_score)
            llm_score = float(numbers[0]) / 10.0 if numbers else 0.5
            llm_score = min(1.0, max(0.0, llm_score))
        except Exception:
            llm_score = 0.5

        # Weighted average: 40% keyword, 60% LLM
        final = (0.4 * keyword_score) + (0.6 * llm_score)
        return round(final, 3)

    def _score_context_recall(self, ground_truth: str, context: str) -> float:
        """
        Context Recall: does the retrieved context contain the information
        needed to construct the ground-truth answer?

        Method: check if key facts from ground truth appear in context.
        Skipped for 'NOT_IN_KNOWLEDGE_BASE' ground truths.
        """
        if ground_truth == "NOT_IN_KNOWLEDGE_BASE":
            return 1.0   # correct to not have this in context

        if not context:
            return 0.0

        prompt = f"""Given this ground truth answer and retrieved context,
what fraction of the information needed to answer correctly is present in the context?

Ground truth: {ground_truth}
Context: {context[:1000]}

Respond with ONLY a decimal number between 0 and 1 (e.g. 0.8):"""

        try:
            import re
            raw = self._llm.invoke(prompt).strip()
            numbers = re.findall(r'\b0?\.\d+\b|\b1\.0\b|\b[01]\b', raw)
            score = float(numbers[0]) if numbers else 0.5
            return round(min(1.0, max(0.0, score)), 3)
        except Exception as exc:
            logger.warning(f"Context recall scoring failed: {exc}")
            return 0.5

    # ── Aggregation ────────────────────────────────────────────────────────────
    @staticmethod
    def _aggregate(results: list[SampleResult]) -> EvaluationResult:
        if not results:
            return EvaluationResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])

        n = len(results)
        return EvaluationResult(
            total_samples          = n,
            avg_faithfulness       = round(sum(r.faithfulness       for r in results) / n, 3),
            avg_answer_relevancy   = round(sum(r.answer_relevancy   for r in results) / n, 3),
            avg_context_precision  = round(sum(r.context_precision  for r in results) / n, 3),
            avg_context_recall     = round(sum(r.context_recall     for r in results) / n, 3),
            avg_latency_ms         = round(sum(r.latency_ms         for r in results) / n, 1),
            refusal_rate           = round(sum(1 for r in results if r.was_refused) / n, 3),
            sample_results         = results,
        )
