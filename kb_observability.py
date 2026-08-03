"""Structured query tracing for the RAG pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict

from kb_config import RAG_TRACE_FILE


_trace_lock = threading.Lock()


def hash_query(query: str) -> str:
    return hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:16]


def write_rag_trace(trace: Dict[str, Any], trace_file: str | None = None):
    path = trace_file or RAG_TRACE_FILE
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **trace,
    }
    with _trace_lock, open(path, "a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
