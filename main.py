# main.py
import streamlit as st
from openai import OpenAI
from utils.auth import (
    login,
    logout,
    is_logged_in,
    register_user,
    init_db,
    init_session_state
)
from utils.token_tracker import TokenTracker
from utils.usage_db import init_usage_db
from utils.feedback_db import init_feedback_db
from sections.admin_dashboard_page import admin_dashboard
from sections.ai_tutor_page import ai_tutor_page
from sections.quiz_page import quizzes_page
from sections.flashcards_page import flashcards_page
from sections.user_usage_page import user_usage_page
from sections.home_page import home_page
from sections.diario_page import diario_page
from sections.attivita_page import attivita_page
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="AI Tutor per Studenti delle Scuole Superiori",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://example.com/help',
        'Report a bug': 'https://example.com/bug',
        'About': '# Questo è un tutor AI per studenti delle scuole superiori in Italia.'
    }
)

# Set up OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI()

@st.cache_resource
def initialize_databases():
    init_db()
    init_usage_db()
    init_feedback_db()
    init_session_state()

@st.cache_resource
def initialize_databases():
    init_db()
    init_usage_db()
    init_feedback_db()
    init_session_state()

# Define functions
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_data
def get_model_choices():
    return ["gpt-4o-mini", "gpt-4o"]

def main():
    # Initialize session state variables
    session_defaults = {
        'subject': 'Matematica',
        'class_year': 3,
        'language': 'Italiano',
        'difficulty': 'Intermedio',
        'processed_docs': None,
        'rag_pipeline': None,
        'token_tracker': TokenTracker(),
        'flashcards': [],
        'current_flashcard_index': 0,
        'quiz': None,
        'quiz_answers': {},
        'quiz_submitted': False,
        'quiz_chat_history': [],
        'quiz_chat_input': "",
        'conversation_history': [],
        'tutor_chat_history': [],
        'current_input': "",
        'feedback_given': {},
        'retrieved_docs': [],
        'user_preferences_set': False,
        'openai_client': None,
        'model_choice': None,
        #'embeddings': None,
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    initialize_databases()
    st.session_state.openai_client = get_openai_client()

    if not is_logged_in():
        login_register_tabs()
    else:
        logged_in_view()

@st.fragment
def login_register_tabs():
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        login()
    with tab2:
        register_form()

@st.fragment
def register_form():
    st.title("📝 Registrazione")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if register_user(username, password):
            st.success("Registrazione avvenuta con successo! Effettua il login.")
            st.rerun()
        else:
            st.error("Il nome utente esiste già.")


def logged_in_view():
    st.sidebar.title(f"👋 Benvenuto, {st.session_state.username}!")
    st.sidebar.title("⚙️ Impostazioni")
    st.session_state.model_choice = st.sidebar.selectbox("Scegli il Modello AI", get_model_choices())

    # Navigation
    st.sidebar.title("🌐 Navigazione")
    page = st.sidebar.radio("Vai a", ["Home","Diario","Attività","Tutor AI", "Flashcard", "Quiz", "Il Tuo Utilizzo", "Dashboard Admin"])

    if st.sidebar.button("🚪 Logout"):
        logout()
        st.rerun()

    # Page routing
    if page == "Home":
        home_page()
    #elif page == "Materia":
        #subject_view()
    elif page == "Diario":
        diario_page(st.session_state.openai_client, st.session_state.token_tracker)
    elif page == "Attività":
        attivita_page(st.session_state.openai_client, st.session_state.token_tracker)
    elif page == "Tutor AI":
        ai_tutor_page(st.session_state.model_choice, st.session_state.openai_client)
    elif page == "Flashcard":
        flashcards_page(st.session_state.model_choice, st.session_state.openai_client)
    elif page == "Quiz":
        quizzes_page(st.session_state.model_choice, st.session_state.openai_client)
    elif page == "Il Tuo Utilizzo":
        user_usage_page()
    elif page == "Dashboard Admin" and st.session_state.is_admin:
        admin_dashboard()
    else:
        st.error("Pagina non accessibile, contatta l'amministratore.")

if __name__ == "__main__":
    try:
        #local_css("assets/styles.css")
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("Si è verificato un errore nell'applicazione. Riprova più tardi.")