"""Command-line entry point for the AutoCorp RAG evaluation pipeline."""

import argparse
import json
import sys

from kb_evaluation import evaluate_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate the AutoCorp hybrid RAG pipeline")
    parser.add_argument("--dataset", required=True, help="Path to the golden JSONL dataset")
    parser.add_argument(
        "--output", default="evals/latest_report.json", help="Evaluation report path"
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Add Gemini correctness, faithfulness, and relevance judging",
    )
    args = parser.parse_args()

    report = evaluate_dataset(
        args.dataset, output_path=args.output, use_llm_judge=args.llm_judge
    )
    print(json.dumps(report["aggregate_metrics"], indent=2))
    print("Quality gates:", "PASSED" if report["passed"] else "FAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
