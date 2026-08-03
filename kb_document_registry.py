"""SQLite-backed document registry for provenance and lifecycle metadata."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from kb_config import DOCUMENT_REGISTRY_PATH, EMBEDDING_MODEL


_registry_lock = threading.Lock()


def _path(db_path=None):
    return db_path or DOCUMENT_REGISTRY_PATH


def _connect(db_path=None):
    registry_path = _path(db_path)
    os.makedirs(os.path.dirname(os.path.abspath(registry_path)), exist_ok=True)
    connection = sqlite3.connect(registry_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            version INTEGER NOT NULL,
            owner TEXT NOT NULL,
            department TEXT NOT NULL,
            classification TEXT NOT NULL,
            status TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(filename, version)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)"
    )
    connection.commit()
    return connection


def _row_to_dict(row) -> Dict[str, Any] | None:
    return dict(row) if row else None


def register_document(
    filename: str,
    text: str,
    owner: str = "unknown",
    department: str = "general",
    classification: str = "internal",
    db_path=None,
) -> Dict[str, Any]:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc).isoformat()

    with _registry_lock, _connect(db_path) as connection:
        existing = connection.execute(
            "SELECT * FROM documents WHERE filename = ? AND content_hash = ?",
            (filename, content_hash),
        ).fetchone()
        if existing:
            connection.execute(
                """
                UPDATE documents
                SET owner = ?, department = ?, classification = ?,
                    status = 'processing', updated_at = ?
                WHERE document_id = ?
                """,
                (owner, department, classification, now, existing["document_id"]),
            )
            connection.commit()
            return get_document(existing["document_id"], db_path=db_path)

        latest = connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM documents WHERE filename = ?",
            (filename,),
        ).fetchone()
        version = int(latest["version"]) + 1
        document_id = "doc_" + hashlib.sha256(
            f"{filename}:{content_hash}".encode("utf-8")
        ).hexdigest()[:20]

        connection.execute(
            """
            INSERT INTO documents (
                document_id, filename, content_hash, version, owner, department,
                classification, status, embedding_model, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'processing', ?, ?, ?)
            """,
            (
                document_id,
                filename,
                content_hash,
                version,
                owner or "unknown",
                department or "general",
                classification or "internal",
                EMBEDDING_MODEL,
                now,
                now,
            ),
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return _row_to_dict(row)


def update_document_status(document_id: str, status: str, db_path=None):
    now = datetime.now(timezone.utc).isoformat()
    with _registry_lock, _connect(db_path) as connection:
        connection.execute(
            "UPDATE documents SET status = ?, updated_at = ? WHERE document_id = ?",
            (status, now, document_id),
        )
        connection.commit()


def get_document(document_id: str, db_path=None) -> Dict[str, Any] | None:
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return _row_to_dict(row)


def get_latest_document_by_filename(filename: str, db_path=None) -> Dict[str, Any] | None:
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT * FROM documents
            WHERE filename = ? AND status != 'deleted'
            ORDER BY version DESC LIMIT 1
            """,
            (filename,),
        ).fetchone()
        return _row_to_dict(row)


def get_documents_by_filenames(
    filenames: Iterable[str], db_path=None
) -> Dict[str, Dict[str, Any]]:
    documents = {}
    for filename in set(filter(None, filenames)):
        record = get_latest_document_by_filename(filename, db_path=db_path)
        if record:
            documents[filename] = record
    return documents


def list_documents(db_path=None) -> List[Dict[str, Any]]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM documents ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]
