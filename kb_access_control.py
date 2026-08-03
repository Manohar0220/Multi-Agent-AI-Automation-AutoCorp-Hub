"""Document-level access-policy helpers for RAG retrieval."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


CLASSIFICATION_LEVELS = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}

DEFAULT_ACCESS_CONTEXT = {
    "department": "general",
    "clearance": "internal",
    "allowed_document_ids": [],
    "allow_pii": False,
    "is_admin": False,
}


def normalize_access_context(context: Dict[str, Any] | None) -> Dict[str, Any]:
    normalized = dict(DEFAULT_ACCESS_CONTEXT)
    if context:
        normalized.update(context)
    normalized["department"] = str(normalized.get("department") or "general").lower()
    normalized["clearance"] = str(normalized.get("clearance") or "public").lower()
    if normalized["clearance"] not in CLASSIFICATION_LEVELS:
        normalized["clearance"] = "public"
    normalized["allowed_document_ids"] = list(
        normalized.get("allowed_document_ids") or []
    )
    normalized["allow_pii"] = bool(normalized.get("allow_pii", False))
    normalized["is_admin"] = bool(normalized.get("is_admin", False))
    return normalized


def is_authorized(metadata: Dict[str, Any], context: Dict[str, Any] | None) -> bool:
    access = normalize_access_context(context)
    if access["is_admin"]:
        return True

    document_id = str(metadata.get("document_id") or "")
    explicit_ids = set(access["allowed_document_ids"])
    if explicit_ids and document_id not in explicit_ids:
        return False

    classification = str(metadata.get("classification") or "public").lower()
    required_level = CLASSIFICATION_LEVELS.get(classification, 3)
    granted_level = CLASSIFICATION_LEVELS[access["clearance"]]
    if required_level > granted_level:
        return False

    document_department = str(metadata.get("department") or "general").lower()
    return document_department in {"general", "all", access["department"]}


def filter_authorized_results(
    results: Iterable[Dict[str, Any]], context: Dict[str, Any] | None
) -> List[Dict[str, Any]]:
    return [
        result
        for result in results
        if is_authorized(result.get("metadata", {}), context)
    ]
