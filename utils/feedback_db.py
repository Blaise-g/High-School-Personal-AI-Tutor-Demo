# feedback_db.py
import streamlit as st
from .db_connection import get_conn, close_conn
import psycopg2

def init_feedback_db():
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feature TEXT NOT NULL,
                feature_id TEXT,
                prompt TEXT,
                response TEXT,
                rating INTEGER,
                comments TEXT,
                model TEXT,
                full_context TEXT
            )
        ''')
        conn.commit()
        print("Feedback database initialized successfully")
    except psycopg2.Error as e:
        print(f"Error initializing feedback database: {e}")
        st.error(f"Error initializing feedback database: {e}")
    finally:
        cur.close()
        close_conn(conn)

def log_feedback(username, feature, feature_id, prompt, response, rating, comments, model, full_context):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute('''
            INSERT INTO feedback (username, feature, feature_id, prompt, response, rating, comments, model, full_context)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (username, feature, feature_id, prompt, response, rating, comments, model, full_context))
        conn.commit()
        print("Feedback logged successfully")
    except psycopg2.Error as e:
        print(f"Error logging feedback: {e}")
        st.error(f"Error logging feedback: {e}")
    finally:
        cur.close()
        close_conn(conn)

def get_feedback_summary():
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute('''
            SELECT feature, COUNT(*) as total_feedback, 
            SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as positive_feedback,
            SUM(CASE WHEN rating = 0 THEN 1 ELSE 0 END) as negative_feedback
            FROM feedback
            GROUP BY feature
        ''')
        feedback_data = cur.fetchall()
        return feedback_data
    except psycopg2.Error as e:
        print(f"Error retrieving feedback summary: {e}")
        st.error(f"Error retrieving feedback summary: {e}")
        return []
    finally:
        cur.close()
        close_conn(conn)

def get_user_feedback(username):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute('''
            SELECT timestamp, feature, rating, comments
            FROM feedback
            WHERE username = %s
            ORDER BY timestamp DESC
        ''', (username,))
        feedback_data = cur.fetchall()
        return feedback_data
    except psycopg2.Error as e:
        st.error(f"Error retrieving user feedback: {e}")
        return []
    finally:
        cur.close()
        close_conn(conn)