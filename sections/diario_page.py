# sections/diario_page.py

import streamlit as st
from datetime import datetime, timedelta
from utils.auth import is_logged_in
from utils.db_connection import get_conn, close_conn
from utils.token_tracker import TokenTracker
from pydantic import BaseModel, ValidationError, Field
from typing import List, Optional
import openai
from openai import OpenAIError
import json
import logging
import psycopg2
import calendar

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# Define DiaryEntry model
class DiaryEntry(BaseModel):
    category: str = Field(..., description="tipo di voce in minuscolo: 'interrogazione', 'verifica', 'compito'")
    date: str = Field(..., description="data nel formato YYYY-MM-DD")
    subject: str = Field(..., description="materia in minuscolo")
    topics: List[str] = Field(..., description="lista degli argomenti trattati in minuscolo")
    details: Optional[str] = Field(None, description="dettagli dell'assegnazione per compito in minuscolo")
    class Config:
        extra = 'forbid'

class DiaryResponse(BaseModel):
    entries: List[DiaryEntry]

def validate_user_input(user_input):
    if not user_input.strip():
        st.warning("Per favore, inserisci del testo prima di inviare.")
        return False
    return True

@st.cache_data(ttl=60)
def get_existing_entries(username):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT attivita_id, tipo, data, materia, argomenti, data_scadenza, dettagli, created_at
            FROM attivita
            WHERE username = %s AND DATE(created_at) = CURRENT_DATE
            ORDER BY created_at DESC
        """, (username,))
        attivita = cur.fetchall()
        return attivita
    except psycopg2.Error as e:
        logger.error(f"Errore nel database recuperando le attività: {e}")
        return []
    finally:
        cur.close()
        close_conn(conn)

@st.cache_data(ttl=3600)
def get_upcoming_activities(username, days=7):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT data, tipo, materia
            FROM attivita
            WHERE username = %s AND data >= CURRENT_DATE AND data < CURRENT_DATE + INTERVAL '%s days'
            ORDER BY data, tipo
        """, (username, days))
        activities = cur.fetchall()
        return activities
    except psycopg2.Error as e:
        logger.error(f"Errore nel recupero delle attività future: {e}")
        return []
    finally:
        cur.close()
        close_conn(conn)

def display_calendar(username):
    today = datetime.now().date()
    activities = get_upcoming_activities(username)

    st.subheader("📅 Prossimi 7 giorni")
    cols = st.columns(7)

    for i in range(7):
        day = today + timedelta(days=i)
        with cols[i]:
            st.write(f"**{day.strftime('%d %b')}**")
            st.write(calendar.day_name[day.weekday()])
            day_activities = [a for a in activities if a[0] == day]
            for activity in day_activities:
                st.write(f"- {activity[1]}: {activity[2]}")

@st.fragment
def display_summary(entries):
    try:
        for entry in entries:
            with st.expander(f"{entry[1]} - {entry[3]} - {entry[2]}"):
                st.write(f"Argomenti: {', '.join(entry[4])}")
                if entry[5]:
                    st.write(f"Data scadenza: {entry[5]}")
                if entry[6]:
                    st.write(f"Dettagli: {entry[6]}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✏️ Modifica", key=f"edit_{entry[0]}"):
                        st.session_state.editing_entry = entry
                with col2:
                    if st.button("🗑️ Elimina", key=f"delete_{entry[0]}"):
                        delete_entry(entry[0])
                        st.rerun()

        if 'editing_entry' in st.session_state:
            edit_entry(st.session_state.editing_entry)
    except Exception as e:
        logger.error(f"Errore nella visualizzazione del riepilogo: {e}")
        st.error(f"Si è verificato un errore nella visualizzazione del riepilogo: {e}")

def edit_entry(entry):
    st.subheader(f"Modifica Attività: {entry[1]}")
    with st.form(key=f"edit_form_{entry[0]}"):
        new_tipo = st.selectbox("Tipo", ["Interrogazione", "Verifica", "Compito", "Argomento", "Custom"], index=["Interrogazione", "Verifica", "Compito", "Argomento", "Custom"].index(entry[1]))
        new_data = st.date_input("Data", value=parse_date(entry[2]))
        new_materia = st.text_input("Materia", value=entry[3])
        new_argomenti = st.text_input("Argomenti", value=", ".join(entry[4]) if entry[4] else "")
        new_data_scadenza = st.date_input("Data Scadenza", value=parse_date(entry[5]) if entry[5] else None)
        new_dettagli = st.text_area("Dettagli", value=entry[6] or "")
        submitted = st.form_submit_button("Salva Modifiche")

    if submitted:
        update_entry(entry[0], new_tipo, new_data, new_materia, new_argomenti.split(", "), new_data_scadenza, new_dettagli)
        st.success("Attività aggiornata con successo!")
        del st.session_state.editing_entry
        st.rerun()

def parse_date(date_value):
    if isinstance(date_value, (datetime, datetime.date)):
        return date_value
    elif isinstance(date_value, str):
        try:
            return datetime.strptime(date_value, "%Y-%m-%d").date()
        except ValueError:
            return datetime.now().date()
    else:
        return datetime.now().date()

def update_entry(attivita_id, tipo, data, materia, argomenti, data_scadenza, dettagli):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            UPDATE attivita
            SET tipo = %s, data = %s, materia = %s, argomenti = %s, data_scadenza = %s, dettagli = %s
            WHERE attivita_id = %s AND username = %s
        """, (tipo, data, materia, argomenti, data_scadenza, dettagli, attivita_id, st.session_state.username))
        conn.commit()
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Errore nell'aggiornamento dell'attività: {e}")
        st.error(f"Errore nell'aggiornamento dell'attività: {e}")
    finally:
        cur.close()
        close_conn(conn)

def delete_entry(attivita_id):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM attivita WHERE attivita_id = %s AND username = %s", (attivita_id, st.session_state.username))
        conn.commit()
        st.success("Attività eliminata con successo!")
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Errore nell'eliminazione dell'attività: {e}")
        st.error(f"Errore nell'eliminazione dell'attività: {e}")
    finally:
        cur.close()
        close_conn(conn)

def input_form(openai_client):
    current_date = datetime.now()
    current_year = current_date.year
    reference_date_str = current_date.strftime("%Y-%m-%d")
    user_input = st.text_area(
        "Descrizione Appunto",
        placeholder="Inserisci qui i tuoi appunti, compiti, o dettagli sui test...",
        height=150
    )
    if st.button("📥 Invia Appunti"):
        if not validate_user_input(user_input):
            return
        with st.spinner("Elaborazione dei tuoi appunti..."):
            try:
                system_prompt = (
                    f"Sei un assistente che estrae informazioni accademiche strutturate dai testi degli studenti. "
                    f"Analizza il testo riga per riga o per punti logici distinti e estrai più voci separate. "
                    f"Per ogni voce estrai: tipo (interrogazione, verifica, compito), data (yyyy-mm-dd), materia, argomenti. "
                    f"Usa l'anno corrente ({current_year}) come default se non specificato. "
                    f"Interpreta le date relative come 'lunedi' o 'domani' rispetto alla data corrente {reference_date_str}. "
                    f"Per 'compito', estrai anche i dettagli dell'assegnazione. "
                    f"Assicurati che tutte le date siano nel formato yyyy-mm-dd e tutti i valori siano in minuscolo. "
                    f"Se una riga non contiene informazioni sufficienti o è completamente irrelativa, restituisci un oggetto con uno o più campi impostati a null. \n\n"
                    f"Restituisci un array di oggetti, uno per ogni voce identificata.\n\n"
                    "Esempio di input con più voci:\n"
                    "Interrogazione Storia 15 ottobre 2024 su Rivoluzione Francese e Americana\n"
                    "Compito di Matematica per venerdì 15 Novembre: Fare esercizi sugli insiemi Pag. 159 n. 131-134, 149, 150-152.\n"
                    "\nEsempio di output JSON:\n"
                    "[{"
                    "    \"category\": \"interrogazione\","
                    "    \"date\": \"2024-10-15\","
                    "    \"subject\": \"storia\","
                    "    \"topics\": [\"rivoluzione francese\", \"rivoluzione americana\"]"
                    "},"
                    "{"
                    "    \"category\": \"compito\","
                    "    \"date\": \"2024-11-24\","
                    "    \"subject\": \"matematica\","
                    "    \"details\": \"fare esercizi pag. 159 n. 131-134, 149, 150-152.\","
                    "    \"topics\": [\"insiemi\"]"
                    "}]"
                )
                completion = openai_client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input},
                    ],
                    response_format=DiaryResponse
                )

                parsed_response = completion.choices[0].message.parsed
                logger.debug(f"Raw OpenAI response: {parsed_response}")
                logger.debug(f"Model dump: {parsed_response.model_dump_json()}")

                st.session_state.parsed_entries = parsed_response.entries
                st.rerun()

            except OpenAIError as oe:
                logger.error(f"Errore nell'API di OpenAI: {oe}")
                st.error(f"Errore nella comunicazione con OpenAI: {oe}")
            except Exception as e:
                logger.error(f"Errore inaspettato: {e}")
                st.error(f"Si è verificato un errore: {e}")

def display_parsed_entry():
    if 'parsed_entries' in st.session_state:
        parsed_entries = st.session_state.parsed_entries
        st.subheader("Conferma le Informazioni Estratte")

        category_labels = {
            "interrogazione": "Interrogazione",
            "verifica": "Verifica",
            "compito": "Compito"
        }

        for i, entry in enumerate(parsed_entries):
            with st.expander(f"Voce #{i+1}: {category_labels.get(entry.category.lower(), entry.category)} - {entry.subject.capitalize()}"):
                st.write(f"**Tipo:** {category_labels.get(entry.category.lower(), entry.category)}")
                st.write(f"**Data:** {entry.date or 'Non specificata'}")
                st.write(f"**Materia:** {entry.subject.capitalize()}")
                st.write(f"**Argomenti:** {', '.join(entry.topics or [])}")

                if entry.category.lower() == "compito" and entry.details:
                    st.write(f"**Dettagli:** {entry.details}")

        if st.button("✅ Conferma e Salva Tutto"):
            for entry in parsed_entries:
                save_entry(entry)
            del st.session_state.parsed_entries
            st.rerun()

def save_entry(parsed_entry):
    try:
        conn = get_conn()
        cur = conn.cursor()

        # Set priority based on category
        priorita_val = 1 if parsed_entry.category in ["interrogazione", "verifica"] else (
            2 if parsed_entry.category == "compito" else 3)

        # Set description based on category
        descrizione = parsed_entry.details if parsed_entry.category == "compito" else ", ".join(parsed_entry.topics or [])

        cur.execute("""
            INSERT INTO attivita (username, tipo, descrizione, data, materia, argomenti, dettagli, priorita, stato)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING attivita_id
        """, (
            st.session_state.username,
            parsed_entry.category,
            descrizione,
            parsed_entry.date,
            parsed_entry.subject,
            parsed_entry.topics,
            parsed_entry.details,
            priorita_val,
            False
        ))

        conn.commit()
        st.success("Le tue informazioni sono state salvate con successo!")
        logger.info(f"Nuova attività aggiunta: {descrizione}")

        # Clear the cache to force a refresh
        get_existing_entries.clear()

        # Clear the parsed_entry from session state
        del st.session_state.parsed_entry

    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Errore nel database durante il salvataggio: {e}")
        st.error(f"Errore nel salvataggio dei dati: {e}")
    except Exception as e:
        logger.error(f"Errore inaspettato durante il salvataggio: {e}")
        st.error(f"Si è verificato un errore durante il salvataggio: {e}")
    finally:
        cur.close()
        close_conn(conn)

def diario_page(openai_client, token_tracker: TokenTracker):
    if not is_logged_in():
        st.error("Per favore, effettua il login per accedere a questa pagina.")
        return

    st.title("📓 Diario Accademico")

    # Display the 7-day calendar view
    display_calendar(st.session_state.username)

    # Input form for new entries
    st.subheader("📝 Aggiungi nuova voce")
    input_form(openai_client)
    display_parsed_entry()

    # Display summary of today's entries
    st.subheader("📚 Voci di oggi")
    entries = get_existing_entries(st.session_state.username)
    if entries:
        display_summary(entries)
    else:
        st.info("Non hai ancora inserito attività oggi. Usa il form sopra per aggiungere una nuova voce.")

    # Add a section for viewing all entries
    st.subheader("🗓️ Tutte le voci")
    if st.button("Mostra tutte le voci"):
        all_entries = get_all_entries(st.session_state.username)
        if all_entries:
            display_all_entries(all_entries)
        else:
            st.info("Non hai ancora inserito attività nel diario.")

def get_all_entries(username):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT attivita_id, tipo, data, materia, argomenti, data_scadenza, dettagli, created_at
            FROM attivita
            WHERE username = %s
            ORDER BY data DESC, created_at DESC
        """, (username,))
        attivita = cur.fetchall()
        return attivita
    except psycopg2.Error as e:
        logger.error(f"Errore nel database recuperando tutte le attività: {e}")
        return []
    finally:
        cur.close()
        close_conn(conn)

def display_all_entries(entries):
    for entry in entries:
        with st.expander(f"{entry[2]} - {entry[1]} - {entry[3]}"):
            st.write(f"Argomenti: {', '.join(entry[4])}")
            if entry[5]:
                st.write(f"Data scadenza: {entry[5]}")
            if entry[6]:
                st.write(f"Dettagli: {entry[6]}")
            st.write(f"Creato il: {entry[7]}")

# Add any additional helper functions here if needed

#if __name__ == "__main__":
    # This block is useful for testing the diario_page function independently
    #import os
    #from dotenv import load_dotenv

    #load_dotenv()

    #openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    #token_tracker = TokenTracker()

    # Mock the session state for testing
    #if 'username' not in st.session_state:
        #st.session_state.username = "test_user"

    #diario_page(openai_client, token_tracker)