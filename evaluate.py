"""
evaluate.py — Standalone RAGAS Test Script
═══════════════════════════════════════════
Run this once after ingestion to measure pipeline quality.
Does NOT require the FastAPI server or Gradio to be running.

Usage:
  python evaluate.py                    # run full BMW benchmark (6 questions)
  python evaluate.py --mode single      # evaluate one custom question
  python evaluate.py --mode benchmark   # explicit benchmark mode
  python evaluate.py --save results.json  # save results to JSON file

Requirements:
  • Neo4j running      (docker-compose up)
  • Ollama running     (ollama serve + ollama pull llama3)
  • BMW PDF ingested   (python ingest_bmw.py)
"""
from __future__ import annotations

import sys
import json
import argparse
import time
from pathlib import Path
from dataclasses import asdict

# ── Project root on path ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from backend.evaluation.ragas_evaluator import (
    RagasEvaluator,
    EvaluationSample,
    EvaluationResult,
    BMW_BENCHMARK,
)
from backend.utils import logger


# ── Console output helpers ────────────────────────────────────────────────────
def _divider(char: str = "─", width: int = 65) -> str:
    return char * width


def _print_result(result: EvaluationResult) -> None:
    print()
    print(_divider("═"))
    print("  RAGAS EVALUATION RESULTS")
    print(_divider("═"))
    print(f"  Total samples  : {result.total_samples}")
    print(f"  Refusal rate   : {result.refusal_rate:.1%}")
    print(f"  Avg latency    : {result.avg_latency_ms:.0f} ms")
    print(_divider())
    print(f"  {'Metric':<25} {'Score':>8}  {'Bar'}")
    print(_divider())

    metrics = {
        "Faithfulness":      result.avg_faithfulness,
        "Answer Relevancy":  result.avg_answer_relevancy,
        "Context Precision": result.avg_context_precision,
        "Context Recall":    result.avg_context_recall,
    }
    for name, score in metrics.items():
        bar   = "█" * int(score * 20)
        color = "✅" if score >= 0.7 else ("⚠️ " if score >= 0.4 else "❌")
        print(f"  {name:<25} {score:>7.3f}  {color} {bar}")

    print(_divider())
    print()
    print("  PER-QUESTION BREAKDOWN")
    print(_divider())

    for i, sr in enumerate(result.sample_results, 1):
        refused = " [REFUSED]" if sr.was_refused else ""
        print(f"\n  [{i}] {sr.question}{refused}")
        print(f"       Pipeline : {sr.pipeline_used}")
        print(f"       Latency  : {sr.latency_ms:.0f} ms")
        print(f"       Faithfulness     : {sr.faithfulness:.3f}")
        print(f"       Answer Relevancy : {sr.answer_relevancy:.3f}")
        print(f"       Context Precision: {sr.context_precision:.3f}")
        print(f"       Context Recall   : {sr.context_recall:.3f}")
        if sr.generated_answer:
            preview = sr.generated_answer[:120].replace("\n", " ")
            print(f"       Answer preview  : {preview}…")

    print()
    print(_divider("═"))


def _print_sample_result(sr) -> None:
    print()
    print(_divider("═"))
    print("  SINGLE EVALUATION RESULT")
    print(_divider("═"))
    print(f"  Question  : {sr.question}")
    print(f"  Pipeline  : {sr.pipeline_used}")
    print(f"  Latency   : {sr.latency_ms:.0f} ms")
    print(f"  Refused   : {sr.was_refused}")
    print(_divider())
    print(f"  Generated answer:")
    print(f"  {sr.generated_answer[:300]}")
    print(_divider())
    print(f"  Ground truth:")
    print(f"  {sr.ground_truth}")
    print(_divider())
    print(f"  {'Metric':<25} {'Score':>8}")
    print(_divider())
    print(f"  {'Faithfulness':<25} {sr.faithfulness:>8.3f}")
    print(f"  {'Answer Relevancy':<25} {sr.answer_relevancy:>8.3f}")
    print(f"  {'Context Precision':<25} {sr.context_precision:>8.3f}")
    print(f"  {'Context Recall':<25} {sr.context_recall:>8.3f}")
    print(_divider("═"))


# ── Modes ─────────────────────────────────────────────────────────────────────
def run_benchmark(save_path: str | None = None) -> None:
    """Run the full BMW 7 Series benchmark suite."""
    print()
    print(_divider("═"))
    print("  BMW 7 SERIES — RAGAS BENCHMARK")
    print(_divider("═"))
    print(f"  Questions : {len(BMW_BENCHMARK)}")
    print(f"  LLM Judge : Ollama (local, free)")
    print(f"  Embeddings: sentence-transformers (local, free)")
    print(_divider("═"))
    print()

    evaluator = RagasEvaluator()
    result    = evaluator.run_benchmark()
    _print_result(result)

    if save_path:
        Path(save_path).write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"  Results saved → {save_path}")
        print()


def run_custom(
    question:  str,
    truth:     str,
    keywords:  list[str],
    save_path: str | None = None,
) -> None:
    """Evaluate a single custom question."""
    evaluator = RagasEvaluator()
    sample    = EvaluationSample(
        question           = question,
        ground_truth       = truth,
        expected_keywords  = keywords,
    )
    sr = evaluator.evaluate_sample(sample)
    _print_sample_result(sr)

    if save_path:
        Path(save_path).write_text(
            json.dumps(asdict(sr), indent=2), encoding="utf-8"
        )
        print(f"\n  Result saved → {save_path}")


def run_custom_batch(questions_file: str, save_path: str | None = None) -> None:
    """
    Evaluate questions from a JSON file.

    JSON format:
    [
      {
        "question":          "What is the wheelbase of the 760i?",
        "ground_truth":      "The wheelbase is 3215 mm.",
        "expected_keywords": ["3215", "wheelbase"]
      },
      ...
    ]
    """
    data    = json.loads(Path(questions_file).read_text(encoding="utf-8"))
    samples = [
        EvaluationSample(
            question          = d["question"],
            ground_truth      = d.get("ground_truth", ""),
            expected_keywords = d.get("expected_keywords", []),
        )
        for d in data
    ]

    evaluator = RagasEvaluator()
    result    = evaluator.evaluate_batch(samples)
    _print_result(result)

    if save_path:
        Path(save_path).write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"  Results saved → {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation script for the Automotive Graph RAG system",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", "single", "batch"],
        default="benchmark",
        help=(
            "benchmark : run built-in BMW 7 Series test suite (default)\n"
            "single    : evaluate one custom question (interactive)\n"
            "batch     : evaluate questions from a JSON file"
        ),
    )
    parser.add_argument(
        "--questions-file",
        default="eval_questions.json",
        help="JSON file with questions for batch mode",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="FILE",
        help="Save results to a JSON file (e.g. results.json)",
    )
    args = parser.parse_args()

    if args.mode == "benchmark":
        run_benchmark(save_path=args.save)

    elif args.mode == "single":
        print("\n  SINGLE QUESTION EVALUATION")
        print(_divider())
        question = input("  Question       : ").strip()
        truth    = input("  Ground truth   : ").strip()
        kw_raw   = input("  Keywords (csv) : ").strip()
        keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]
        run_custom(question, truth, keywords, save_path=args.save)

    elif args.mode == "batch":
        if not Path(args.questions_file).exists():
            # Create a sample file if none exists
            sample = [
                {
                    "question":          "What is the horsepower of the BMW 760i?",
                    "ground_truth":      "The BMW 760i produces 536 horsepower.",
                    "expected_keywords": ["536", "horsepower", "760i"],
                },
                {
                    "question":          "What type of engine does the 740i have?",
                    "ground_truth":      "The 740i uses a 3.0L TwinPower Turbo inline-6.",
                    "expected_keywords": ["inline", "6", "turbo", "3.0"],
                },
            ]
            Path(args.questions_file).write_text(
                json.dumps(sample, indent=2), encoding="utf-8"
            )
            print(f"\n  Created sample questions file: {args.questions_file}")
            print("  Edit it and re-run: python evaluate.py --mode batch")
            return
        run_custom_batch(args.questions_file, save_path=args.save)


if __name__ == "__main__":
    main()
