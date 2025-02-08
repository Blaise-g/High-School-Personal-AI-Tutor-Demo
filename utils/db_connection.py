
import os
import psycopg2

def get_conn():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise Exception("DATABASE_URL not found in environment variables")
    return psycopg2.connect(db_url)

def close_conn(conn):
    conn.close()
