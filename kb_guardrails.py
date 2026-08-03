"""Input, retrieved-context, citation, and PII guardrails for the RAG pipeline."""

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from kb_config import MAX_QUERY_CHARS, MAX_UPLOAD_BYTES


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".md"}

_PROMPT_INJECTION_PATTERNS = {
    "instruction_override": re.compile(
        r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|system|developer)\b.{0,30}\b(instruction|prompt|message)s?\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "secret_exfiltration": re.compile(
        r"\b(reveal|print|return|expose|show)\b.{0,40}\b(system prompt|developer message|api key|secret|credential)s?\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "tool_manipulation": re.compile(
        r"\b(call|invoke|execute|run)\b.{0,30}\b(tool|function|shell|command)s?\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "role_impersonation": re.compile(
        r"\b(you are now|act as|pretend to be)\b.{0,60}\b(system|administrator|developer|root)\b",
        re.IGNORECASE | re.DOTALL,
    ),
}

_PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)"),
    "phone": re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"),
}


class GuardrailViolation(ValueError):
    """Raised when an input violates a blocking RAG guardrail."""


@dataclass(frozen=True)
class CitationValidation:
    valid: List[str]
    invalid: List[str]

    @property
    def precision(self) -> float:
        total = len(self.valid) + len(self.invalid)
        return len(self.valid) / total if total else 0.0


def detect_prompt_injection(text: str) -> List[str]:
    return [name for name, pattern in _PROMPT_INJECTION_PATTERNS.items() if pattern.search(text or "")]


def validate_query(query: str) -> str:
    query = (query or "").strip()
    if not query:
        raise GuardrailViolation("The question cannot be empty.")
    if len(query) > MAX_QUERY_CHARS:
        raise GuardrailViolation(
            f"The question exceeds the {MAX_QUERY_CHARS}-character limit."
        )
    risks = detect_prompt_injection(query)
    if risks:
        raise GuardrailViolation(
            "The question was blocked by the prompt-injection guardrail: "
            + ", ".join(risks)
        )
    return query


def validate_upload(filename: str, file_size: int, text: str) -> List[str]:
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    violations = []
    if extension not in ALLOWED_EXTENSIONS:
        violations.append(f"unsupported file type: {extension or 'none'}")
    if file_size > MAX_UPLOAD_BYTES:
        violations.append(f"file exceeds {MAX_UPLOAD_BYTES} bytes")
    if not (text or "").strip():
        violations.append("document contains no extractable text")
    risks = detect_prompt_injection(text)
    if risks:
        violations.append("possible prompt injection: " + ", ".join(risks))
    return violations


def contains_prompt_injection(text: str) -> bool:
    return bool(detect_prompt_injection(text))


def redact_pii(text: str) -> Tuple[str, List[str]]:
    redacted = text or ""
    detected = []
    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(redacted):
            detected.append(pii_type)
            redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)
    return redacted, detected


def validate_citations(
    citations: Iterable[str], allowed_source_ids: Iterable[str]
) -> CitationValidation:
    allowed = set(allowed_source_ids)
    valid = []
    invalid = []
    seen = set()
    for citation in citations or []:
        citation = str(citation).strip()
        if not citation or citation in seen:
            continue
        seen.add(citation)
        (valid if citation in allowed else invalid).append(citation)
    return CitationValidation(valid=valid, invalid=invalid)
