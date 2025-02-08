# sections/attivita_page.py

import streamlit as st
from datetime import datetime, timedelta, timezone
from utils.auth import is_logged_in
from utils.db_connection import get_conn, close_conn
from utils.token_tracker import TokenTracker
from utils.fsrs import fsrs_manager
import logging
import psycopg2
import random
import json

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

@st.cache_data(ttl=60)
def get_today_activities(username, days_in_advance_compiti, days_in_advance_verifiche):
    try:
        conn = get_conn()
        cur = conn.cursor()
        today = datetime.now().date()
        future_date_compiti = today + timedelta(days=days_in_advance_compiti)
        future_date_verifiche = today + timedelta(days=days_in_advance_verifiche)
        cur.execute("""
            SELECT attivita_id, tipo, descrizione, data, data_scadenza, priorita, stato, materia
            FROM attivita
            WHERE username = %s AND 
                  ((tipo = 'Compito' AND data_scadenza BETWEEN %s AND %s) OR
                   (tipo IN ('Interrogazione', 'Verifica') AND data BETWEEN %s AND %s))
                  AND stato = FALSE
            ORDER BY CASE WHEN tipo = 'Compito' THEN data_scadenza ELSE data END ASC, priorita ASC
        """, (username, today, future_date_compiti, today, future_date_verifiche))
        return cur.fetchall()
    except psycopg2.Error as e:
        logger.error(f"Errore nel recupero delle attività di oggi: {e}")
        return []
    finally:
        cur.close()
        close_conn(conn)

@st.cache_data(ttl=60)
def get_upcoming_activities(username, days_in_advance_compiti, days_in_advance_verifiche):
    """Get all upcoming activities including flashcards"""
    activities = []
    today = datetime.now().date()

    # Get regular upcoming activities
    conn = get_conn()
    cur = conn.cursor()
    try:
        future_date_compiti = today + timedelta(days=days_in_advance_compiti)
        future_date_verifiche = today + timedelta(days=days_in_advance_verifiche)
        cur.execute("""
            SELECT attivita_id, tipo, descrizione, data, data_scadenza, priorita, stato, materia
            FROM attivita
            WHERE username = %s 
            AND ((tipo = 'Compito' AND data_scadenza > %s) OR
                 (tipo IN ('Interrogazione', 'Verifica') AND data > %s))
            AND stato = FALSE
            ORDER BY CASE WHEN tipo = 'Compito' THEN data_scadenza ELSE data END ASC, priorita ASC
        """, (username, today, today))
        activities.extend(cur.fetchall())
    finally:
        cur.close()
        close_conn(conn)

    # Add future flashcard reviews
    flashcard_activities = get_flashcard_activities(username, future_days=max(days_in_advance_compiti, days_in_advance_verifiche))
    future_flashcards = [act for act in flashcard_activities if act['date'] > today]

    # Convert flashcard activities to match regular activities format
    for act in future_flashcards:
        formatted_activity = (
            None,  # attivita_id
            'Flashcard Review',  # tipo
            f"Review {sum(int(count) for count in act['topic_counts'].values())} flashcards",  # descrizione
            act['date'],  # data
            None,  # data_scadenza
            2,  # priorita
            act['fully_reviewed'],  # stato
            f"{act['subject']}: {', '.join(act['topic_counts'].keys())}"  # materia
        )
        activities.append(formatted_activity)

    return sorted(activities, key=lambda x: x[3])  # Sort by date

@st.cache_data(ttl=60)
def get_flashcard_activities(username, future_days=30):
    """Get flashcard activities with accurate review status"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        logger.info(f"get_flashcard_activities - Start query execution for {username}")
        cur.execute("""
            WITH grouped_cards AS (
                SELECT 
                    DATE(f.next_review) as review_date,
                    s.name as subject_name,
                    t.name as topic_name,
                    COUNT(*) as cards_count,
                    COUNT(CASE WHEN f.reviewed_today = TRUE THEN 1 END) as reviewed_count
                FROM flashcards f
                JOIN topics t ON f.topic_id = t.topic_id
                JOIN subjects s ON t.subject_id = s.subject_id
                WHERE f.username = %s
                AND DATE(f.next_review) BETWEEN CURRENT_DATE 
                    AND CURRENT_DATE + interval '%s days'
                GROUP BY DATE(f.next_review), s.name, t.name
            )
            SELECT 
                review_date,
                subject_name,
                jsonb_object_agg(topic_name, cards_count::text) as topic_counts,
                bool_and(cards_count = COALESCE(reviewed_count, 0)) as fully_reviewed
            FROM grouped_cards
            GROUP BY review_date, subject_name
            ORDER BY review_date, subject_name
        """, (username, future_days))

        results = cur.fetchall()
        logger.info(f"get_flashcard_activities - Query executed, found {len(results)} rows")

        # Convert tuples to dictionaries
        activities = []
        for row in results:
            activity = {
                'date': row[0],
                'subject': row[1],
                'topic_counts': row[2],
                'fully_reviewed': row[3]
            }
            activities.append(activity)
            logger.info(f"get_flashcard_activities - Processed activity: {activity}")

        return activities

    except Exception as e:
        logger.error(f"Error getting flashcard activities: {str(e)}")
        return []
    finally:
        cur.close()
        close_conn(conn)

# Let's also add a debug function to inspect the flashcards table
def debug_flashcards(username):
    """Debug function to inspect flashcards table"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                f.flashcard_id,
                s.name as subject,
                t.name as topic,
                f.state,
                f.next_review,
                EXISTS (
                    SELECT 1 FROM flashcard_reviews fr
                    WHERE fr.flashcard_id = f.flashcard_id
                    AND fr.review_time::date = CURRENT_DATE
                ) as reviewed_today
            FROM flashcards f
            JOIN topics t ON f.topic_id = t.topic_id
            JOIN subjects s ON t.subject_id = s.subject_id
            WHERE f.username = %s
            ORDER BY f.next_review
        """, (username,))
        cards = cur.fetchall()

        logger.info(f"Debug: Found {len(cards)} flashcards for user {username}")
        for card in cards:
            logger.info(f"Card {card[0]}: subject={card[1]}, topic={card[2]}, state={card[3]}, next_review={card[4]}, reviewed_today={card[5]}")

        return cards
    finally:
        cur.close()
        close_conn(conn)

def display_flashcard_activities(activities, is_today=True):
    """Display flashcard activities with improved styling"""
    today = datetime.now(timezone.utc).date()

    logger.info(f"Displaying flashcard activities for {'today' if is_today else 'future'}")
    logger.info(f"Activities received: {activities}")

    if not activities:
        return

    for activity in activities:
        try:
            # Verify that activity is in the right format
            if not isinstance(activity, dict):
                logger.error(f"Invalid activity format: {activity}")
                continue

            date = activity.get('date')
            subject = activity.get('subject')
            topic_counts = activity.get('topic_counts')
            fully_reviewed = activity.get('fully_reviewed')

            if None in (date, subject, topic_counts):
                logger.error(f"Missing required fields in activity: {activity}")
                continue

            # Filter based on whether we're showing today's or future activities
            if (is_today and date != today) or (not is_today and date <= today):
                continue

            total_cards = sum(int(count) for count in topic_counts.values())

            if total_cards > 0:
                with st.container():
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"### 📚 {subject}")
                        st.markdown(f"**Date:** {date.strftime('%d/%m/%Y')}")

                    with col2:
                        st.markdown(f"**Total:** {total_cards} card{'s' if total_cards > 1 else ''}")

                    st.markdown("#### Topics:")
                    for topic, count in topic_counts.items():
                        st.info(f"• {topic}: {count} card{'s' if int(count) > 1 else ''}")

                    if is_today:
                        if fully_reviewed:
                            st.success("✅ Flashcards reviewed")
                        else:
                            st.warning("📝 Flashcards need review")

                    st.markdown("---")

        except Exception as e:
            logger.error(f"Error processing activity: {str(e)}")
            continue

@st.fragment
def display_activities(activities, is_today=True):
    """Display regular activities (excluding flashcards)"""
    if not activities:
        st.info("Nessuna attività programmata." if is_today else "Nessuna attività futura.")
        return

    # Filter out flashcard activities
    regular_activities = [act for act in activities if act[1] != 'Flashcard Review']

    if not regular_activities:
        return
        
    # Display regular activities
    for activity in regular_activities:
        attivita_id, tipo, descrizione, data, data_scadenza, priorita, stato, materia = activity
        icon = get_icon(tipo)
        priority_str = "Alta" if priorita == 1 else "Media" if priorita == 2 else "Bassa"

        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            if is_today:
                st.checkbox(
                    f"{icon} **#{materia}:** {tipo}",
                    value=stato,
                    key=f"activity_{attivita_id}",
                    on_change=on_checkbox_change,
                    args=(attivita_id,)
                )
            else:
                st.markdown(f"{icon} **#{materia}:** {tipo}")
        with col2:
            if tipo == 'Compito':
                st.write(f"Dettagli: {descrizione}")
            else:
                st.write(f"Argomenti: {descrizione}")
        with col3:
            if tipo == 'Compito':
                st.write(f"📅 {data_scadenza.strftime('%d/%m/%Y')}")
            else:
                st.write(f"📅 {data.strftime('%d/%m/%Y')}")
        st.write(f"Priorità: {priority_str}")
        st.markdown("---")

def on_checkbox_change(attivita_id):
    new_status = st.session_state[f"activity_{attivita_id}"]
    update_activity_status(attivita_id, new_status)
    st.rerun()

def update_activity_status(attivita_id, new_status):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            UPDATE attivita
            SET stato = %s
            WHERE attivita_id = %s
        """, (new_status, attivita_id))
        conn.commit()
        logger.info(f"Stato attività {attivita_id} aggiornato a {new_status}.")

        # Clear the cache to force a refresh
        get_today_activities.clear()
        get_upcoming_activities.clear()
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Errore nell'aggiornamento dello stato dell'attività: {e}")
        st.error(f"Errore nell'aggiornamento dello stato dell'attività: {e}")
    finally:
        cur.close()
        close_conn(conn)

def get_icon(tipo):
    tipo_lower = tipo.lower()
    if tipo_lower == 'interrogazione':
        return "🗣️"
    elif tipo_lower == 'verifica':
        return "📝"
    elif tipo_lower == 'compito':
        return "📚"
    elif tipo_lower == 'quiz':
        return "❓"
    elif tipo_lower == 'test':
        return "✅"
    elif tipo_lower == 'flashcard review':
        return "🔄"
    else:
        return "📌"

def attivita_page(openai_client, token_tracker):
    try:
        if not is_logged_in():
            st.error("Per favore, effettua il login per accedere a questa pagina.")
            return

        st.title("🗂️ Gestione Attività")

        # Store relevant state
        if 'current_review_subject' not in st.session_state:
            st.session_state['current_review_subject'] = None
        if 'current_review_date' not in st.session_state:
            st.session_state['current_review_date'] = None

        with st.expander("Impostazioni"):
            days_in_advance_compiti = st.slider("Mostra compiti in scadenza nei prossimi giorni:", 1, 14, 2)
            days_in_advance_verifiche = st.slider("Mostra verifiche e interrogazioni nei prossimi giorni:", 1, 30, 1)

        tab_oggi, tab_prossimi_giorni = st.tabs(["Oggi", "Prossimi giorni"])

        # Debug current flashcards state
        debug_cards = debug_flashcards(st.session_state.username)
        
        # Add debug logging
        logger.info("attivita_page - Before fetching flashcard activities")
        
        # Get flashcard activities
        flashcard_activities = get_flashcard_activities(st.session_state.username)
        
        logger.info(f"attivita_page - After fetching activities: {flashcard_activities}")

        with tab_oggi:
            st.header(f"📅 Oggi: {datetime.now().strftime('%A, %d %B %Y')}")
            
            # Display regular activities first
            activities = get_today_activities(st.session_state.username, days_in_advance_compiti, days_in_advance_verifiche)
            if activities:
                st.subheader("📝 Attività")
                display_activities(activities, is_today=True)

            # Display flashcard activities
            if flashcard_activities:
                st.subheader("📝 Flashcard Reviews")
                display_flashcard_activities(flashcard_activities, is_today=True)

        with tab_prossimi_giorni:
            st.header("📆 Attività future")
            future_activities = get_upcoming_activities(st.session_state.username, days_in_advance_compiti, days_in_advance_verifiche)
            if future_activities:
                st.subheader("📝 Attività")
                display_activities(future_activities, is_today=False)
            else:
                st.info("Nessuna attività futura")

            if flashcard_activities:
                st.subheader("📝 Upcoming Flashcard Reviews")
                display_flashcard_activities(flashcard_activities, is_today=False)
            else:
                st.info("Nessuna flashcard futura da rivedere")
            
    except Exception as e:
        logger.error(f"Error in attivita_page: {str(e)}", exc_info=True)  # Add exc_info for full traceback
        st.error("Si è verificato un errore. Per favore, ricarica la pagina.")

#if __name__ == "__main__":
    # This block is useful for testing the attivita_page function independently
    #import os
    #from dotenv import load_dotenv

    #load_dotenv()

    #openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    #token_tracker = TokenTracker()

    # Mock the session state for testing
    #if 'username' not in st.session_state:
        #st.session_state.username = "test_user"

    #attivita_page(openai_client, token_tracker)