import os
from dotenv import load_dotenv
from db import get_employee_id_by_email
import psycopg2

load_dotenv()

def print_env():
    print("PGHOST=", os.getenv("PGHOST"))
    print("PGPORT=", os.getenv("PGPORT"))
    print("PGDATABASE=", os.getenv("PGDATABASE"))
    print("PGUSER=", os.getenv("PGUSER"))

def simple_select():
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
    )
    with conn, conn.cursor() as cur:
        cur.execute("SELECT employee_id, full_name, email FROM employees ORDER BY employee_id;")
        rows = cur.fetchall()
        print("\nAll employees:")
        for r in rows:
            print(r)
    conn.close()

def lookup_tests():
    print("\nLookup tests:")
    for email in [
        "korikanamanohar2@gmail.com",
        "harshitha.attanti1@gmail.com",
    ]:
        eid = get_employee_id_by_email(email)
        print(f"{email} -> {eid}")

if __name__ == "__main__":
    print_env()
    simple_select()
    lookup_tests()
