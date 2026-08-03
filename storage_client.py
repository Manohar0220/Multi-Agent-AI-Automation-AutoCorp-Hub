import os
import threading
from google.cloud import storage

_storage_client = None
_bucket = None
_client_lock = threading.Lock()


def _get_storage():
    """Initialize GCS lazily so unrelated agents do not require GCP credentials."""
    global _storage_client, _bucket
    if _storage_client is None:
        with _client_lock:
            if _storage_client is None:
                project = os.getenv("GCP_PROJECT_ID")
                bucket_name = os.getenv("GCS_BUCKET")
                if not bucket_name:
                    raise RuntimeError("GCS_BUCKET is not configured")
                _storage_client = storage.Client(project=project)
                _bucket = _storage_client.bucket(bucket_name)
    return _storage_client, _bucket

def fetch_employee_file(employee_id: str, file_name: str):
    """
    Looks for GCS object at <employee_id>/<file_name>.
    Returns (bytes, content_type) or None if not found.
    """
    _, bucket = _get_storage()
    blob = bucket.blob(f"{employee_id}/{file_name}")
    if not blob.exists():
        return None
    data = blob.download_as_bytes()
    return data, (blob.content_type or "application/octet-stream")

def list_employee_files(employee_id: str, limit: int = 10):
    """
    Returns up to `limit` file names under the employee's folder (no prefixes).
    Helpful when exact file doesn't exist.
    """
    prefix = f"{employee_id}/"
    storage_client, bucket = _get_storage()
    names = []
    for b in storage_client.list_blobs(bucket.name, prefix=prefix):
        names.append(b.name[len(prefix):])
        if len(names) >= limit:
            break
    return names
