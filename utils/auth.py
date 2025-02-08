
# utils/auth.py

import streamlit as st
import hashlib
from datetime import datetime
from .db_connection import get_conn, close_conn
import psycopg2
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture detailed logs


# utils/auth.py

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    try:
        # Create users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT,
                created_at TIMESTAMP,
                is_admin BOOLEAN
            )
        """)
        logger.info("Table 'users' created successfully.")

        # Create user_progress table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_progress (
                username TEXT PRIMARY KEY,
                documents_processed INTEGER DEFAULT 0,
                flashcards_generated INTEGER DEFAULT 0,
                quizzes_taken INTEGER DEFAULT 0,
                hints_requested INTEGER DEFAULT 0,
                FOREIGN KEY (username) REFERENCES users(username)
            )
        """)
        logger.info("Table 'user_progress' created successfully.")

        # Create user_preferences table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                username TEXT PRIMARY KEY,
                class_year INTEGER,
                language TEXT,
                difficulty TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users(username)
            )
        """)
        logger.info("Table 'user_preferences' created successfully.")

        # Create subjects table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS subjects (
                subject_id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Table 'subjects' created successfully.")

        # Create topics table with importance field
        cur.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                topic_id SERIAL PRIMARY KEY,
                subject_id INTEGER REFERENCES subjects(subject_id),
                name VARCHAR(100) NOT NULL,
                emoji VARCHAR(10),
                start_date DATE,
                end_date DATE,
                importance INTEGER DEFAULT 2,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Table 'topics' created successfully.")

        # Create files table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS files (
                file_id SERIAL PRIMARY KEY,
                username TEXT REFERENCES users(username),
                topic_id INTEGER REFERENCES topics(topic_id),
                filename TEXT NOT NULL,
                file_type VARCHAR(50),
                upload_date DATE DEFAULT CURRENT_DATE,
                metadata JSONB,
                file_path TEXT NOT NULL,
                processed_content TEXT,
                embeddings JSONB,
                embedding_model TEXT,
                version INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Table 'files' created successfully.")

        # Create attivita table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attivita (
                attivita_id SERIAL PRIMARY KEY,
                username TEXT REFERENCES users(username),
                tipo VARCHAR(50) NOT NULL, 
                descrizione TEXT NOT NULL,
                data DATE NOT NULL, 
                materia VARCHAR(100) NOT NULL,
                argomenti TEXT[] NOT NULL, 
                data_scadenza DATE,
                dettagli TEXT, 
                priorita INTEGER DEFAULT 1, 
                stato BOOLEAN DEFAULT FALSE, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Table 'attivita' created successfully.")

        # Create quizzes table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS quizzes (
                quiz_id SERIAL PRIMARY KEY,
                username TEXT REFERENCES users(username),
                topic_id INTEGER REFERENCES topics(topic_id),
                quiz_data JSONB NOT NULL,
                status VARCHAR(20) DEFAULT 'ready',
                priority INTEGER DEFAULT 2,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                last_attempt TIMESTAMP WITH TIME ZONE,
                completion_rate FLOAT DEFAULT 0,
                average_score FLOAT DEFAULT 0,
                total_questions INTEGER DEFAULT 0,
                time_spent INTEGER DEFAULT 0
            )
        """)
        logger.info("Table 'quizzes' created successfully.")

        # Create quiz_questions table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS quiz_questions (
                question_id SERIAL PRIMARY KEY,
                quiz_id INTEGER REFERENCES quizzes(quiz_id),
                question_number INTEGER NOT NULL,
                question TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                options JSONB,
                question_type VARCHAR(50) DEFAULT 'open',
                created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                times_answered INTEGER DEFAULT 0,
                times_correct INTEGER DEFAULT 0
            )
        """)
        logger.info("Table 'quiz_questions' created successfully.")

        # Create quiz_attempts table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                attempt_id SERIAL PRIMARY KEY,
                quiz_id INTEGER REFERENCES quizzes(quiz_id),
                username TEXT REFERENCES users(username),
                score FLOAT NOT NULL,
                start_time TIMESTAMP WITH TIME ZONE NOT NULL,
                end_time TIMESTAMP WITH TIME ZONE NOT NULL,
                duration INTEGER,  -- in seconds
                answers JSONB,    -- store user answers
                feedback JSONB,   -- store any feedback given
                created_at TIMESTAMP WITH TIME ZONE NOT NULL
            )
        """)
        logger.info("Table 'quiz_attempts' created successfully.")

        # Add quiz_analytics table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS quiz_analytics (
                analytics_id SERIAL PRIMARY KEY,
                username TEXT REFERENCES users(username),
                topic_id INTEGER REFERENCES topics(topic_id),
                total_quizzes INTEGER DEFAULT 0,
                total_questions INTEGER DEFAULT 0,
                correct_answers INTEGER DEFAULT 0,
                avg_time_spent FLOAT DEFAULT 0.0,
                avg_accuracy FLOAT DEFAULT 0.0,
                last_quiz_date TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create flashcards table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS flashcards (
                flashcard_id SERIAL PRIMARY KEY,
                username TEXT REFERENCES users(username),
                topic_id INTEGER REFERENCES topics(topic_id),
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                state INTEGER DEFAULT 0,
                difficulty FLOAT DEFAULT 5.0,
                stability FLOAT DEFAULT 1.0,
                reps INTEGER DEFAULT 0,
                lapses INTEGER DEFAULT 0,
                last_review TIMESTAMP WITH TIME ZONE,
                next_review TIMESTAMP WITH TIME ZONE,
                reviewed_today BOOLEAN DEFAULT FALSE,
                priority INTEGER DEFAULT 2,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL
            )
        """)
        logger.info("Table 'flashcards' created successfully.")

        # Create flashcard_metadata table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS flashcard_metadata (
                metadata_id SERIAL PRIMARY KEY,
                flashcard_id INTEGER REFERENCES flashcards(flashcard_id),
                metadata JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL
            )
        """)
        logger.info("Table 'flashcard_metadata' created successfully.")

        # Create flashcard_reviews table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS flashcard_reviews (
                review_id SERIAL PRIMARY KEY,
                flashcard_id INTEGER REFERENCES flashcards(flashcard_id),
                username TEXT REFERENCES users(username),
                rating INTEGER NOT NULL,
                review_time TIMESTAMP WITH TIME ZONE NOT NULL,
                response_time INTEGER,
                review_state INTEGER,
                is_initial_review BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL
            )
        """)
        logger.info("Table 'flashcard_reviews' created successfully.")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS fsrs_user_parameters (
                username TEXT PRIMARY KEY REFERENCES users(username),
                parameters JSONB NOT NULL,
                last_optimization TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for better performance
        cur.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_next_review ON flashcards(next_review)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_username ON flashcards(username)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_topic ON flashcards(topic_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_quizzes_status ON quizzes(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_quizzes_topic_user ON quizzes(topic_id, username)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_quiz_attempts_dates ON quiz_attempts(start_time, end_time)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_quiz_analytics_user ON quiz_analytics(username)")
        logger.info("Indexes created successfully.")

        conn.commit()
        logger.info("All tables and indexes created successfully.")
    except psycopg2.Error as e:
        logger.error(f"Database initialization error: {e}")
        st.error(f"Database initialization error: {e}")
    finally:
        cur.close()
        close_conn(conn)



def hash_password(password):
    """
    Hashes the password using SHA-256.
    """
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, is_admin=False):
    hashed_password = hash_password(password)
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO users (username, password, created_at, is_admin)
            VALUES (%s, %s, %s, %s)
        """, (username, hashed_password, datetime.now(), is_admin))

        cur.execute("""
            INSERT INTO user_progress (username)
            VALUES (%s)
        """, (username,))

        conn.commit()
        return True
    except psycopg2.IntegrityError:
        conn.rollback()
        return False
    except psycopg2.Error as e:
        st.error(f"Errore nel database durante la registrazione: {e}")
        return False
    finally:
        cur.close()
        close_conn(conn)

def verify_user(username, password):
    hashed_password = hash_password(password)
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT password, is_admin FROM users WHERE username = %s
        """, (username,))
        result = cur.fetchone()
        if result and result[0] == hashed_password:
            return True, result[1]
        return False, False
    except psycopg2.Error as e:
        st.error(f"Errore nel database durante la verifica: {e}")
        return False, False
    finally:
        cur.close()
        close_conn(conn)

def get_user_progress(username):
    """
    Retrieves user progress metrics.
    """
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT documents_processed, flashcards_generated, quizzes_taken, hints_requested
            FROM user_progress WHERE username = %s
        """, (username,))
        result = cur.fetchone()
        if result:
            return {
                'documents_processed': result[0],
                'flashcards_generated': result[1],
                'quizzes_taken': result[2],
                'hints_requested': result[3]
            }
        return None
    except psycopg2.Error as e:
        st.error(f"Errore nel database recuperando i progressi dell'utente: {e}")
        return None
    finally:
        cur.close()
        close_conn(conn)

def update_user_progress(username, progress):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE user_progress 
            SET documents_processed = documents_processed + %s, 
                flashcards_generated = flashcards_generated + %s, 
                quizzes_taken = quizzes_taken + %s, 
                hints_requested = hints_requested + %s
            WHERE username = %s
        """, (
            progress.get('documents_processed', 0),
            progress.get('flashcards_generated', 0),
            progress.get('quizzes_taken', 0),
            progress.get('hints_requested', 0),
            username
        ))
        conn.commit()
    except psycopg2.Error as e:
        st.error(f"Errore nel database aggiornando i progressi dell'utente: {e}")
    finally:
        cur.close()
        close_conn(conn)

def init_session_state():
    """
    Initializes session state variables for user authentication.
    """
    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

def login():
    """
    Renders the login form and handles authentication.
    """
    st.title("🔐 Login")
    username = st.text_input("Nome Utente ")
    password = st.text_input("Password ", type="password")
    if st.button("Accedi"):
        if not username or not password:
            st.error("Per favore, inserisci sia il nome utente che la password.")
            return
        is_valid, is_admin = verify_user(username, password)
        if is_valid:
            st.session_state.username = username
            st.session_state.is_admin = is_admin
            st.session_state.logged_in = True
            st.success("Accesso effettuato con successo!")
            st.rerun()
        else:
            st.error("Nome utente o password non validi.")

def logout():
    """
    Logs out the current user.
    """
    st.session_state.username = ''
    st.session_state.is_admin = False
    st.session_state.logged_in = False
    st.success("Sei stato disconnesso con successo!")
    st.rerun()

def is_logged_in():
    """
    Checks if a user is logged in.
    """
    return st.session_state.get('logged_in', False)

def get_all_users():
    """
    Retrieves all usernames.
    """
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT username FROM users")
        users = [row[0] for row in cur.fetchall()]
        return users
    except psycopg2.Error as e:
        st.error(f"Errore nel database recuperando gli utenti: {e}")
        return []
    finally:
        cur.close()
        close_conn(conn)
