import os
import psycopg2
from contextlib import contextmanager

@contextmanager
def _conn():
    conn = psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
    )
    try:
        yield conn
    finally:
        conn.close()

def get_employee_id_by_email(email: str) -> str:
    """
    Returns employee_id for given email (case-insensitive), or None if not found.
    """
    sql = """
        SELECT employee_id
        FROM employees
        WHERE LOWER(email) = LOWER(%s)
        LIMIT 1
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (email,))
            row = cur.fetchone()
            return row[0] if row else None
