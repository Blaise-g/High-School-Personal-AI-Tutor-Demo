# usage_db.py
import streamlit as st
import psycopg2
from datetime import datetime
from .db_connection import get_conn, close_conn

def init_usage_db():
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model TEXT NOT NULL,
                total_tokens INTEGER NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost REAL NOT NULL
            )
        """)
        conn.commit()
    except psycopg2.Error as e:
        st.error(f"Error initializing usage database: {e}")
    finally:
        cur.close()
        close_conn(conn)

def log_usage(username, model, usage, cost):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO usage (username, model, total_tokens, input_tokens, output_tokens, cost)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            username,
            model,
            usage.get('total_tokens', 0),
            usage.get('prompt_tokens', 0),
            usage.get('completion_tokens', 0),
            cost
        ))
        conn.commit()
    except psycopg2.Error as e:
        st.error(f"Error logging usage: {e}")
    finally:
        cur.close()
        close_conn(conn)
        
def get_user_usage(username):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT username, timestamp, model, total_tokens, input_tokens, output_tokens, cost
            FROM usage
            WHERE username = %s
            ORDER BY timestamp DESC
        """, (username,))
        usage_data = cur.fetchall()
        return usage_data
    except psycopg2.Error as e:
        st.error(f"Error retrieving user usage: {e}")
        return []
    finally:
        cur.close()
        close_conn(conn)

def get_usage_by_user():
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT username, timestamp, model, total_tokens, input_tokens, output_tokens, cost
            FROM usage
            ORDER BY timestamp DESC
        """)
        usage_data = cur.fetchall()
        return usage_data
    except psycopg2.Error as e:
        st.error(f"Error retrieving usage data: {e}")
        return []
    finally:
        cur.close()
        close_conn(conn)

def get_total_usage():
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                SUM(total_tokens) AS total_tokens, 
                SUM(input_tokens) AS input_tokens, 
                SUM(output_tokens) AS output_tokens, 
                SUM(cost) AS total_cost 
            FROM usage
        """)
        result = cur.fetchone()
        return result if result else (0, 0, 0, 0)
    except psycopg2.Error as e:
        st.error(f"Error retrieving total usage: {e}")
        return (0, 0, 0, 0)
    finally:
        cur.close()
        close_conn(conn)