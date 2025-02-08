# utils/fsrs.py

from fsrs import FSRS, Card, Rating, State
from datetime import datetime, timezone, timedelta
import pandas as pd
from utils.db_connection import get_conn, close_conn
import streamlit as st
import subprocess
import json
import os
import logging
import psycopg2

logger = logging.getLogger(__name__)

class FSRSManager:
    def __init__(self):
        self.base_params = {
            "w": [0.40255, 1.18385, 3.173, 15.69105, 7.1949, 0.5345, 1.4604, 0.0046, 1.54575, 0.1192, 1.01925, 1.9395, 0.11, 0.29605, 2.2698, 0.2315, 2.9898, 0.51655, 0.6621],
            "request_retention": 0.875,
            "maximum_interval": 36500
        }
        self.fsrs = FSRS(**self.base_params)
        # Change from daily to weekly check
        self.check_and_run_weekly_optimization()

    def load_user_params(self, username):
        """Load user-specific FSRS parameters if they exist"""
        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT parameters, last_optimization
                FROM fsrs_user_parameters
                WHERE username = %s
            """, (username,))
            result = cur.fetchone()

            if result:
                params, last_opt = result
                return params, last_opt
            return None, None
        except Exception as e:
            logger.error(f"Error loading user parameters: {str(e)}")
            return None, None
        finally:
            cur.close()
            close_conn(conn)

    def save_user_params(self, username, parameters):
        """Save user-specific FSRS parameters"""
        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO fsrs_user_parameters (
                    username, parameters, last_optimization, updated_at
                ) VALUES (%s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (username) 
                DO UPDATE SET 
                    parameters = EXCLUDED.parameters,
                    last_optimization = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
            """, (username, json.dumps(parameters)))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving user parameters: {str(e)}")
            return False
        finally:
            cur.close()
            close_conn(conn)

    def load_user_params(self, username):
        """Load user-specific FSRS parameters if they exist and are less than 7 days old"""
        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT parameters, last_optimization
                FROM fsrs_user_parameters
                WHERE username = %s
                AND last_optimization > CURRENT_TIMESTAMP - INTERVAL '7 days'
            """, (username,))
            result = cur.fetchone()

            if result:
                params, last_opt = result
                return params, last_opt
            return None, None
        except Exception as e:
            logger.error(f"Error loading user parameters: {str(e)}")
            return None, None
        finally:
            cur.close()
            close_conn(conn)

    def check_and_run_weekly_optimization(self):
        """Check and run optimization for users who need it (weekly)"""
        conn = get_conn()
        cur = conn.cursor()
        try:
            # Get all users with reviews who haven't been optimized in the last 7 days
            cur.execute("""
                WITH user_reviews AS (
                    SELECT DISTINCT fr.username 
                    FROM flashcard_reviews fr
                    WHERE fr.rating > 0
                )
                SELECT ur.username 
                FROM user_reviews ur
                LEFT JOIN fsrs_user_parameters fup 
                    ON ur.username = fup.username
                WHERE fup.username IS NULL 
                   OR fup.last_optimization < CURRENT_TIMESTAMP - INTERVAL '7 days'
            """)
            users_to_optimize = [row[0] for row in cur.fetchall()]

            for username in users_to_optimize:
                logger.info(f"Running weekly optimization for user: {username}")
                self.run_optimization_for_user(username)

        except Exception as e:
            logger.error(f"Error in weekly optimization check: {str(e)}")
        finally:
            cur.close()
            close_conn(conn)

    def generate_review_logs(self, username):
        """Generate review logs file for a specific user"""
        conn = get_conn()
        cur = conn.cursor()
        try:
            # First check if we have enough reviews
            cur.execute("""
                SELECT COUNT(*) 
                FROM flashcard_reviews 
                WHERE username = %s AND rating > 0
            """, (username,))
            review_count = cur.fetchone()[0]

            if review_count < 10:  # Minimum reviews needed for optimization
                return False

            # Get review data
            cur.execute("""
                SELECT 
                    flashcard_id as card_id,
                    EXTRACT(EPOCH FROM review_time) * 1000 as review_time,
                    rating as review_rating,
                    review_state,
                    COALESCE(response_time, 0) as review_duration
                FROM flashcard_reviews
                WHERE username = %s AND rating > 0
                ORDER BY review_time
            """, (username,))

            # Convert to DataFrame manually
            columns = ['card_id', 'review_time', 'review_rating', 'review_state', 'review_duration']
            data = cur.fetchall()
            df = pd.DataFrame(data, columns=columns)

            filename = f"revlog_{username}.csv"
            df.to_csv(filename, index=False)
            return filename
        except Exception as e:
            logger.error(f"Error generating review logs for {username}: {str(e)}")
            return False
        finally:
            cur.close()
            close_conn(conn)

    def run_optimization_for_user(self, username):
        """Run optimization for a specific user"""
        try:
            # Generate user-specific review logs
            log_file = self.generate_review_logs(username)
            if not log_file:
                logger.info(f"Not enough review data for user {username}")
                return False

            # Run the optimizer
            result = subprocess.run(
                ["python", "-m", "fsrs_optimizer", log_file],
                capture_output=True,
                text=True
            )

            # Clean up log file
            try:
                os.remove(log_file)
            except:
                pass

            if result.returncode == 0:
                # Parse optimizer output to get parameters
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if '"w":' in line:
                        try:
                            params = json.loads("{" + line.strip().rstrip(',') + "}")
                            params.update({
                                "request_retention": self.base_params["request_retention"],
                                "maximum_interval": self.base_params["maximum_interval"]
                            })
                            # Save optimized parameters for user
                            if self.save_user_params(username, params):
                                logger.info(f"Successfully optimized parameters for user {username}")
                                return True
                        except json.JSONDecodeError:
                            logger.error(f"Error parsing optimizer output for user {username}")

            logger.error(f"Optimization failed for user {username}: {result.stderr}")
            return False

        except Exception as e:
            logger.error(f"Error in optimization process for user {username}: {str(e)}")
            return False

    def get_user_fsrs(self, username):
        """Get FSRS instance with user-specific parameters"""
        params, last_opt = self.load_user_params(username)
        if params:
            try:
                return FSRS(**json.loads(params))
            except:
                pass
        return self.fsrs

    def get_next_flashcards(self, username, limit=None):
        """Get flashcards due for review"""
        now = datetime.now(timezone.utc)
        conn = get_conn()
        cur = conn.cursor()
        try:
            logger.info(f"Fetching flashcards due for review at {now}")

            cur.execute("""
                SELECT 
                    f.flashcard_id, f.question, f.answer, f.state, 
                    f.difficulty, f.stability, f.last_review, f.priority,
                    t.name as topic_name, s.name as subject_name,
                    f.next_review, f.reps, f.lapses
                FROM flashcards f
                LEFT JOIN topics t ON f.topic_id = t.topic_id
                LEFT JOIN subjects s ON t.subject_id = s.subject_id
                WHERE f.username = %s 
                AND f.next_review <= %s
                AND DATE(f.next_review) = CURRENT_DATE
                ORDER BY 
                    f.next_review ASC,
                    f.priority DESC
            """, (username, now))

            cards = []
            for row in cur.fetchall():
                logger.info(f"Found card {row[0]} due at {row[10]} with state {row[3]}")
                card_data = {
                    'id': row[0],
                    'question': row[1],
                    'answer': row[2],
                    'card': Card(
                        due=row[10],
                        stability=row[5],
                        difficulty=row[4],
                        elapsed_days=0,
                        scheduled_days=0,
                        reps=row[11],
                        lapses=row[12],
                        state=State(row[3]),
                        last_review=row[6],
                    ),
                    'priority': row[7],
                    'topic_name': row[8],
                    'subject_name': row[9]
                }
                cards.append(card_data)

            logger.info(f"Retrieved {len(cards)} cards due for review")
            return cards[:limit] if limit else cards
        finally:
            cur.close()
            close_conn(conn)

    def update_flashcard_review(self, flashcard_id, rating, review_time=None):
        """Update flashcard using FSRS algorithm"""
        conn = get_conn()
        cur = conn.cursor()
        try:
            # Get current flashcard state
            cur.execute("""
                SELECT state, difficulty, stability, last_review, username,
                       reps, lapses
                FROM flashcards
                WHERE flashcard_id = %s
            """, (flashcard_id,))
            state, difficulty, stability, last_review, username, reps, lapses = cur.fetchone()

            review_time = review_time or datetime.now(timezone.utc)
            elapsed_days = (review_time - last_review).total_seconds() / (24 * 3600) if last_review else 0

            # Initialize FSRS card with complete state
            card = Card(
                due=review_time,
                stability=stability,
                difficulty=difficulty,
                elapsed_days=int(elapsed_days),
                scheduled_days=0,
                reps=reps or 0,
                lapses=lapses or 0,
                state=State(state),
                last_review=last_review
            )

            fsrs_rating = Rating(rating)
            scheduled_info = self.fsrs.review_card(card, fsrs_rating, review_time)
            new_card = scheduled_info[0]
            review_log = scheduled_info[1]

            # Update flashcard state with all FSRS parameters
            cur.execute("""
                UPDATE flashcards
                SET state = %s,
                    difficulty = %s,
                    stability = %s,
                    last_review = %s,
                    next_review = %s,
                    reps = %s,
                    lapses = %s,
                    reviewed_today = TRUE
                WHERE flashcard_id = %s
            """, (
                new_card.state.value,
                new_card.difficulty,
                new_card.stability,
                review_time,
                new_card.due,
                new_card.reps,
                new_card.lapses,
                flashcard_id
            ))

            # Update flashcard_reviews table
            cur.execute("""
                INSERT INTO flashcard_reviews (
                    flashcard_id,
                    username,
                    rating,
                    review_time,
                    response_time,
                    review_state,
                    created_at,
                    is_initial_review
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                flashcard_id,
                username,
                rating,
                review_time,
                None,
                new_card.state.value,
                review_time,
                False
            ))

            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating flashcard review: {str(e)}")
            return False
        finally:
            cur.close()
            close_conn(conn)

    def get_review_statistics(self, username, period_days=30):
        """Get review statistics for a user"""
        conn = get_conn()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT 
                    COUNT(*) as total_reviews,
                    AVG(rating) as avg_rating,
                    COUNT(CASE WHEN rating = 1 THEN 1 END) as again_count,
                    COUNT(CASE WHEN rating = 2 THEN 1 END) as hard_count,
                    COUNT(CASE WHEN rating = 3 THEN 1 END) as good_count,
                    COUNT(CASE WHEN rating = 4 THEN 1 END) as easy_count
                FROM flashcard_reviews
                WHERE username = %s
                AND review_time >= CURRENT_DATE - INTERVAL '%s days'
            """, (username, period_days))

            return cur.fetchone()
        except Exception as e:
            logger.error(f"Error getting review statistics: {str(e)}")
            return None
        finally:
            cur.close()
            close_conn(conn)

fsrs_manager = FSRSManager()

def init_fsrs_tables():
    """Initialize FSRS-related database tables"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        # Create parameters table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fsrs_user_parameters (
                username TEXT PRIMARY KEY REFERENCES users(username),
                parameters JSONB NOT NULL,
                last_optimization TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index on last_optimization
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_fsrs_last_opt ON fsrs_user_parameters(last_optimization)
        """)

        conn.commit()
        logger.info("FSRS tables initialized successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating FSRS tables: {str(e)}")
    finally:
        cur.close()
        close_conn(conn)