import os
from google.cloud import storage

_project = os.getenv("GCP_PROJECT_ID")
_bucket_name = os.getenv("GCS_BUCKET")

_storage_client = storage.Client(project=_project)
_bucket = _storage_client.bucket(_bucket_name)

def fetch_employee_file(employee_id: str, file_name: str):
    """
    Looks for GCS object at <employee_id>/<file_name>.
    Returns (bytes, content_type) or None if not found.
    """
    blob = _bucket.blob(f"{employee_id}/{file_name}")
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
    names = []
    for b in _storage_client.list_blobs(_bucket_name, prefix=prefix):
        names.append(b.name[len(prefix):])
        if len(names) >= limit:
            break
    return names
