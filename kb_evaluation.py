"""Offline evaluation pipeline and quality metrics for AutoCorp hybrid RAG."""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from kb_config import get_gemini_model
from kb_query_engine import run_query_pipeline


DEFAULT_QUALITY_GATES = {
    "retrieval_recall_at_10": 0.80,
    "retrieval_ndcg_at_10": 0.70,
    "answer_token_f1": 0.70,
    "citation_precision": 0.95,
}


def _unique(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(item) for item in items if item))


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = list(retrieved)[:k]
    return sum(item in relevant for item in top) / k


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 1.0 if not list(retrieved)[:k] else 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str]) -> float:
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    gains = [1.0 if item in relevant else 0.0 for item in list(retrieved)[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_count = min(len(relevant), k)
    if ideal_count == 0:
        return 1.0 if not gains else 0.0
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal_dcg


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def token_f1(actual: str, expected: str) -> float:
    actual_tokens = _tokens(actual)
    expected_tokens = _tokens(expected)
    if not actual_tokens and not expected_tokens:
        return 1.0
    if not actual_tokens or not expected_tokens:
        return 0.0
    actual_counts = defaultdict(int)
    expected_counts = defaultdict(int)
    for token in actual_tokens:
        actual_counts[token] += 1
    for token in expected_tokens:
        expected_counts[token] += 1
    overlap = sum(
        min(count, expected_counts[token]) for token, count in actual_counts.items()
    )
    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _relationship_key(relationship: Any) -> Tuple[str, str, str]:
    if isinstance(relationship, str):
        parts = [part.strip().lower() for part in relationship.split("|")]
        return tuple((parts + ["", "", ""])[:3])
    return (
        str(relationship.get("source", "")).strip().lower(),
        str(relationship.get("relationship", relationship.get("type", "")))
        .strip()
        .lower(),
        str(relationship.get("target", "")).strip().lower(),
    )


def graph_relationship_metrics(
    actual: Iterable[Any], expected: Iterable[Any]
) -> Dict[str, float]:
    actual_set = {_relationship_key(item) for item in actual}
    expected_set = {_relationship_key(item) for item in expected}
    if not expected_set:
        return {
            "graph_relationship_precision": 1.0 if not actual_set else 0.0,
            "graph_relationship_recall": 1.0,
            "graph_relationship_f1": 1.0 if not actual_set else 0.0,
        }
    true_positives = len(actual_set & expected_set)
    precision = true_positives / len(actual_set) if actual_set else 0.0
    recall = true_positives / len(expected_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "graph_relationship_precision": precision,
        "graph_relationship_recall": recall,
        "graph_relationship_f1": f1,
    }


def _citation_document_metrics(
    result: Dict[str, Any], relevant_documents: Set[str]
) -> Dict[str, float]:
    source_map = {
        source.get("source_id"): source.get("metadata", {}).get("source")
        for source in result.get("sources", [])
    }
    cited_documents = {
        source_map[citation]
        for citation in result.get("citations", [])
        if citation in source_map and source_map[citation]
    }
    if not result.get("citations"):
        precision = 0.0
    elif relevant_documents:
        precision = len(cited_documents & relevant_documents) / len(
            result["citations"]
        )
    else:
        precision = float(result.get("citation_precision", 0.0))
    recall = (
        len(cited_documents & relevant_documents) / len(relevant_documents)
        if relevant_documents
        else 1.0
    )
    return {"citation_precision": precision, "citation_recall": recall}


def judge_answer_quality(
    question: str, expected_answer: str, answer: str, context: str
) -> Dict[str, float]:
    """Optional LLM judge; deterministic metrics remain the evaluation baseline."""
    model = get_gemini_model()
    prompt = f"""Evaluate a RAG answer. Return JSON only with scores from 0 to 1.

Question: {question}
Reference answer: {expected_answer}
Generated answer: {answer}
Retrieved context: {context}

{{"answer_correctness": 0.0, "faithfulness": 0.0, "answer_relevance": 0.0}}"""
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json", "temperature": 0},
    )
    parsed = json.loads(response.text)
    return {
        key: max(0.0, min(1.0, float(parsed.get(key, 0.0))))
        for key in ("answer_correctness", "faithfulness", "answer_relevance")
    }


def evaluate_case(
    case: Dict[str, Any], result: Dict[str, Any], use_llm_judge: bool = False
) -> Dict[str, Any]:
    relevant_source_ids = set(case.get("relevant_source_ids") or [])
    relevant_documents = set(case.get("relevant_documents") or [])
    if relevant_source_ids:
        retrieved = _unique(result.get("retrieved_source_ids", []))
        relevant = relevant_source_ids
    else:
        retrieved = _unique(result.get("retrieved_documents", []))
        relevant = relevant_documents

    should_abstain = bool(case.get("should_abstain", False))
    correct_abstention = float(
        bool(result.get("abstained")) == should_abstain
    )
    metrics = {
        "retrieval_precision_at_5": precision_at_k(retrieved, relevant, 5),
        "retrieval_recall_at_5": recall_at_k(retrieved, relevant, 5),
        "retrieval_recall_at_10": recall_at_k(retrieved, relevant, 10),
        "retrieval_mrr": reciprocal_rank(retrieved, relevant),
        "retrieval_ndcg_at_10": ndcg_at_k(retrieved, relevant, 10),
        "answer_token_f1": correct_abstention
        if should_abstain
        else token_f1(result.get("answer", ""), case.get("expected_answer", "")),
        "correct_abstention": correct_abstention,
        "end_to_end_latency_ms": float(
            sum(result.get("latencies_ms", {}).values())
        ),
    }
    metrics.update(_citation_document_metrics(result, relevant_documents))
    if should_abstain and not result.get("citations"):
        metrics["citation_precision"] = 1.0
        metrics["citation_recall"] = 1.0
    if "expected_relationships" in case:
        metrics.update(
            graph_relationship_metrics(
                result.get("graph_triples", []), case.get("expected_relationships", [])
            )
        )
    if use_llm_judge:
        metrics.update(
            judge_answer_quality(
                case["question"],
                case.get("expected_answer", ""),
                result.get("answer", ""),
                result.get("context", ""),
            )
        )
    return {
        "id": case.get("id", case["question"][:40]),
        "question": case["question"],
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "abstained": result.get("abstained", False),
        "metrics": metrics,
    }


def load_jsonl_dataset(path: str) -> List[Dict[str, Any]]:
    cases = []
    with open(path, "r", encoding="utf-8") as dataset:
        for line_number, line in enumerate(dataset, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            if "question" not in case:
                raise ValueError(f"Dataset line {line_number} has no question")
            cases.append(case)
    if not cases:
        raise ValueError("The evaluation dataset is empty")
    return cases


def _aggregate(case_results: List[Dict[str, Any]]) -> Dict[str, float]:
    values = defaultdict(list)
    for case in case_results:
        for name, value in case["metrics"].items():
            values[name].append(float(value))
    return {
        name: sum(metric_values) / len(metric_values)
        for name, metric_values in values.items()
        if metric_values
    }


def evaluate_dataset(
    dataset_path: str,
    output_path: str | None = None,
    use_llm_judge: bool = False,
    quality_gates: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    cases = load_jsonl_dataset(dataset_path)
    case_results = []
    for case in cases:
        try:
            result = run_query_pipeline(
                case["question"], access_context=case.get("access_context")
            )
            case_results.append(evaluate_case(case, result, use_llm_judge))
        except Exception as exc:
            failed_result = {
                "answer": "",
                "retrieved_documents": [],
                "retrieved_source_ids": [],
                "citations": [],
                "sources": [],
                "graph_triples": [],
                "abstained": True,
                "latencies_ms": {},
            }
            evaluated = evaluate_case(case, failed_result, use_llm_judge=False)
            evaluated["error"] = str(exc)
            case_results.append(evaluated)

    aggregate = _aggregate(case_results)
    gates = quality_gates or DEFAULT_QUALITY_GATES
    gate_results = {
        metric: {
            "actual": aggregate.get(metric),
            "required": threshold,
            "passed": aggregate.get(metric) is not None
            and aggregate[metric] >= threshold,
        }
        for metric, threshold in gates.items()
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": os.path.abspath(dataset_path),
        "case_count": len(case_results),
        "aggregate_metrics": aggregate,
        "quality_gates": gate_results,
        "passed": all(gate["passed"] for gate in gate_results.values()),
        "cases": case_results,
    }
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
    return report
