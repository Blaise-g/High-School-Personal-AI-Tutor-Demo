# sections/home_page.py
import streamlit as st
from utils.fsrs import FSRS, Card, Rating, State 
from utils.auth import is_logged_in, get_user_progress, update_user_progress
from utils.db_connection import get_conn, close_conn
from genai.document_processing import process_documents
from genai.rag_pipeline import initialize_rag, process_documents_parallel
from genai.quiz_ai import generate_quiz
from genai.flash_ai import generate_flashcards
import psycopg2
from datetime import datetime, timezone
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Force streamlit logger to warning level to reduce noise
logging.getLogger('streamlit').setLevel(logging.WARNING)

# At the top of the home_page.py file
def debug_session_state():
    """Debug current session state"""
    logger.info("Current session state:")
    logger.info(f"current_subject: {st.session_state.get('current_subject')}")
    logger.info(f"model_choice: {st.session_state.get('model_choice')}")
    logger.info(f"openai_client: {st.session_state.get('openai_client') is not None}")

@st.cache_data(ttl=3600)
def get_subjects():
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM subjects ORDER BY name")
        return [row[0] for row in cur.fetchall()]
    except psycopg2.Error as e:
        st.error(f"Error fetching subjects: {e}")
        return []
    finally:
        cur.close()
        close_conn(conn)

@st.cache_data(ttl=300)
def get_topics(subject):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT t.topic_id, t.name, t.emoji, t.start_date, t.end_date
            FROM topics t
            JOIN subjects s ON t.subject_id = s.subject_id
            WHERE s.name = %s
            ORDER BY t.start_date
        """, (subject,))
        return [{'id': row[0], 'name': row[1], 'emoji': row[2], 'start_date': row[3], 'end_date': row[4]} for row in cur.fetchall()]
    except psycopg2.Error as e:
        st.error(f"Error fetching topics: {e}")
        return []
    finally:
        cur.close()
        close_conn(conn)

def create_subject(subject_name):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO subjects (name) VALUES (%s)", (subject_name,))
        conn.commit()
        get_subjects.clear()  # Clear the cache for subjects
        return True
    except psycopg2.Error as e:
        st.error(f"Error creating subject: {e}")
        return False
    finally:
        cur.close()
        close_conn(conn)

def save_user_preferences(username, class_year, language, difficulty):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO user_preferences (username, class_year, language, difficulty)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (username) DO UPDATE
            SET class_year = EXCLUDED.class_year,
                language = EXCLUDED.language,
                difficulty = EXCLUDED.difficulty
        """, (username, class_year, language, difficulty))
        conn.commit()
        return True
    except psycopg2.Error as e:
        st.error(f"Error saving user preferences: {e}")
        return False
    finally:
        cur.close()
        close_conn(conn)
@st.cache_data(ttl=300)
def get_user_preferences(username):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT class_year, language, difficulty FROM user_preferences WHERE username = %s", (username,))
        result = cur.fetchone()
        if result:
            return {
                'class_year': result[0],
                'language': result[1],
                'difficulty': result[2]
            }
        return None
    except psycopg2.Error as e:
        st.error(f"Error fetching user preferences: {e}")
        return None
    finally:
        cur.close()
        close_conn(conn)

def home_page():
    if not is_logged_in():
        st.error("Per favore, effettua il login per accedere a questa pagina.")
        return

    st.title(f"👋 Benvenuto, {st.session_state.username}!")

    # User Preferences Section (remains the same)
    display_user_preferences()

    # Subject Management Section
    st.markdown("---")
    st.header("📚 Gestione Materie")

    tab1, tab2 = st.tabs(["Materie Esistenti", "Crea Nuova Materia"])

    with tab1:
        subjects = get_subjects()
        if subjects:
            selected_subject = st.selectbox("Seleziona una materia", subjects)
            if st.button("Gestisci Materia"):
                handle_subject_selection(selected_subject)
                st.rerun()
        else:
            st.info("Non ci sono materie disponibili. Crea una nuova materia per iniziare.")

    with tab2:
        create_new_subject()

    # Subject Content Management Section
    if 'current_subject' in st.session_state:
        display_subject_content()

def handle_subject_selection(selected_subject):
    """Handle subject selection and state updates"""
    st.session_state.current_subject = selected_subject
    st.session_state.subject = selected_subject
    # Ensure preferences are loaded
    user_prefs = get_user_preferences(st.session_state.username)
    if user_prefs:
        st.session_state.class_year = user_prefs['class_year']
        st.session_state.language = user_prefs['language']
        st.session_state.difficulty = user_prefs['difficulty']

def display_user_preferences():
    """Display and manage user preferences"""
    st.subheader("🎯 Le tue preferenze di studio")
    user_prefs = get_user_preferences(st.session_state.username)

    class_years = [1, 2, 3, 4, 5]
    languages = ["Italiano", "Inglese"]
    difficulty_levels = ["Principiante", "Intermedio", "Avanzato"]

    col1, col2, col3 = st.columns(3)
    with col1:
        class_year = st.selectbox(
            "Anno Scolastico", 
            class_years, 
            index=class_years.index(user_prefs['class_year']) if user_prefs else 0
        )
    with col2:
        language = st.selectbox(
            "Lingua", 
            languages, 
            index=languages.index(user_prefs['language']) if user_prefs else 0
        )
    with col3:
        difficulty = st.selectbox(
            "Difficoltà", 
            difficulty_levels, 
            index=difficulty_levels.index(user_prefs['difficulty']) if user_prefs else 0
        )

    if st.button("Salva Preferenze"):
        if save_user_preferences(st.session_state.username, class_year, language, difficulty):
            st.success("Preferenze salvate con successo!")
            # Update session state
            st.session_state.class_year = class_year
            st.session_state.language = language
            st.session_state.difficulty = difficulty
            get_user_preferences.clear()  # Clear the cache
        else:
            st.error("Errore nel salvataggio delle preferenze.")

def create_new_subject():
    """Handle creation of new subjects"""
    new_subject = st.text_input("Nome della nuova materia")
    if st.button("Crea Materia"):
        if new_subject:
            if create_subject(new_subject):
                st.success(f"Materia '{new_subject}' creata con successo!")
                st.rerun()
            else:
                st.error("Errore nella creazione della materia.")
        else:
            st.warning("Inserisci un nome per la nuova materia.")

def display_subject_content():
    """Display subject content management interface"""
    logger.info("Entering display_subject_content")
    debug_session_state()

    st.markdown("---")
    st.header(f"📚 {st.session_state.current_subject}")

    tab1, tab2, tab3 = st.tabs(["Gestione Argomenti", "Carica Contenuti", "Genera Materiali"])

    with tab1:
        display_topics_management()

    with tab2:
        display_content_upload()

    with tab3:
        logger.info("Entering materials generation tab")
        display_study_materials()

def create_new_topic():
    """Handle creation of new topics"""
    topic_name = st.text_input("Nome dell'argomento")
    topic_emoji = st.text_input("Emoji dell'argomento (opzionale)")
    start_date = st.date_input("Data di inizio")
    end_date = st.date_input("Data di fine")

    if st.button("Crea Argomento"):
        if topic_name and start_date and end_date:
            if create_topic(st.session_state.current_subject, topic_name, topic_emoji, start_date, end_date):
                st.success(f"Argomento '{topic_name}' creato con successo!")
                st.rerun()
            else:
                st.warning("Si è verificato un errore durante la creazione dell'argomento.")
        else:
            st.warning("Per favore, compila tutti i campi obbligatori.")

def create_topic(subject, name, emoji, start_date, end_date):
    """Create a new topic in database"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT subject_id FROM subjects WHERE name = %s", (subject,))
        subject_id = cur.fetchone()[0]
        cur.execute("""
            INSERT INTO topics (subject_id, name, emoji, start_date, end_date)
            VALUES (%s, %s, %s, %s, %s)
        """, (subject_id, name, emoji, start_date, end_date))
        conn.commit()
        get_topics.clear()  # Clear the cache for topics
        return True
    except psycopg2.Error as e:
        st.error(f"Error creating topic: {e}")
        return False
    finally:
        cur.close()
        close_conn(conn)

def get_files_for_topic(topic_id):
    """Get files for a specific topic"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT file_id, filename, file_type, upload_date
            FROM files
            WHERE topic_id = %s
            ORDER BY upload_date DESC
        """, (topic_id,))
        return [{'id': row[0], 'filename': row[1], 'file_type': row[2], 'upload_date': row[3]} 
                for row in cur.fetchall()]
    except psycopg2.Error as e:
        st.error(f"Error fetching files: {e}")
        return []
    finally:
        cur.close()
        close_conn(conn)

def delete_file(file_id):
    """Delete a file from the database"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
        conn.commit()
        get_files_for_topic.clear()  # Clear the cache
        return True
    except psycopg2.Error as e:
        st.error(f"Error deleting file: {e}")
        return False
    finally:
        cur.close()
        close_conn(conn)

def upload_content(topic):
    """Handle content upload"""
    st.subheader("📤 Carica Nuovo Contenuto")
    uploaded_files = st.file_uploader(
        "Carica i tuoi appunti e documenti", 
        accept_multiple_files=True, 
        type=['pdf', 'txt', 'docx', 'csv', 'html'], 
        key=f"uploader_{topic['id']}"
    )
    web_urls = st.text_area("Inserisci gli URL web da includire (uno per riga):")

    if st.button("📄 Elabora Materiali"):
        if not uploaded_files and not web_urls.strip():
            st.warning("Per favore, carica almeno un file o inserisci un URL prima di elaborare i materiali.")
        else:
            process_and_store_documents(topic, uploaded_files, web_urls)

def display_topics_management():
    """Handle topics display and management"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📖 Argomenti Esistenti")
        topics = get_topics(st.session_state.current_subject)
        if topics:
            for topic in topics:
                with st.expander(f"{topic['emoji']} {topic['name']}"):
                    st.write(f"Periodo: {topic['start_date']} - {topic['end_date']}")
                    display_topic_files(topic)
        else:
            st.info("Nessun argomento disponibile per questa materia.")

    with col2:
        st.subheader("➕ Nuovo Argomento")
        create_new_topic()

def display_topic_files(topic):
    """Display files for a specific topic"""
    files = get_files_for_topic(topic['id'])
    if files:
        for file in files:
            col1, col2 = st.columns([3, 1])
            col1.write(f"📄 {file['filename']} ({file['upload_date'].strftime('%d/%m/%Y')})")
            if col2.button("🗑️ Elimina", key=f"delete_{file['id']}"):
                if delete_file(file['id']):
                    st.success(f"File {file['filename']} eliminato con successo.")
                    st.rerun()
    else:
        st.info("Nessun file caricato per questo argomento.")

def display_content_upload():
    """Handle content upload interface"""
    topics = get_topics(st.session_state.current_subject)
    if not topics:
        st.warning("Crea prima un argomento per poter caricare i contenuti.")
        return

    selected_topic = st.selectbox(
        "Seleziona l'argomento per il caricamento",
        options=topics,
        format_func=lambda x: f"{x['emoji']} {x['name']}"
    )

    upload_content(selected_topic)

def store_documents(topic, processed_docs, rag_pipeline):
    """Store documents with their embeddings"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        for doc in processed_docs:
            # Ensure embeddings are properly JSON encoded if not already
            embeddings = doc.get('embeddings', [])
            if isinstance(embeddings, str):
                embeddings_json = embeddings
            else:
                embeddings_json = json.dumps(embeddings)

            # Store document content and metadata
            cur.execute("""
                INSERT INTO files (
                    username, topic_id, filename, file_type, 
                    metadata, file_path, processed_content,
                    embeddings, embedding_model, upload_date
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                st.session_state.username,
                topic['id'],
                doc['metadata']['source'],
                doc['metadata']['type'],
                json.dumps(doc['metadata']),
                doc['metadata']['source'],
                doc['content'],
                embeddings_json,
                "text-embedding-3-small",
                datetime.now(timezone.utc)
            ))
        conn.commit()
        logger.info(f"Successfully stored {len(processed_docs)} documents with embeddings")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing documents: {str(e)}")
        raise
    finally:
        cur.close()
        close_conn(conn)

def process_and_store_documents(topic, uploaded_files, web_urls):
    """Process and store documents with embeddings"""
    with st.spinner("Elaborazione dei materiali..."):
        try:
            web_url_list = [url.strip() for url in web_urls.split('\n') if url.strip()]
            raw_docs = process_documents(uploaded_files, web_url_list)
            processed_docs = process_documents_parallel(raw_docs)

            # Initialize RAG and generate embeddings
            rag_pipeline = initialize_rag(processed_docs, st, st.session_state.token_tracker)
            if not rag_pipeline:
                st.error("Errore nella generazione degli embeddings.")
                return

            # Store documents and embeddings
            store_documents(topic, processed_docs, rag_pipeline)

            st.success("Contenuti elaborati e embeddings generati con successo!")
            st.session_state.subject = st.session_state.current_subject

            # Inform user they can generate materials
            st.info("Ora puoi generare quiz e flashcards nella sezione 'Genera Materiali'")

        except Exception as e:
            st.error(f"Errore nell'elaborazione: {str(e)}")
            logger.error(f"Document processing error: {str(e)}")

def get_topic_embeddings(topic_id):
    """Get stored embeddings for a topic"""
    logger.info(f"Retrieving embeddings for topic {topic_id}")

    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT embeddings, processed_content
            FROM files
            WHERE topic_id = %s
            AND embeddings IS NOT NULL
        """, (topic_id,))

        results = cur.fetchall()
        logger.info(f"Found {len(results)} files with embeddings")

        if not results:
            logger.warning("No embeddings found for topic")
            return None

        # Combine all embeddings and content
        combined_data = {
            'embeddings': [],
            'content': ''
        }

        for emb, content in results:
            try:
                if emb:
                    # Check if emb is already a list or needs to be parsed
                    if isinstance(emb, str):
                        emb_data = json.loads(emb)
                    else:
                        emb_data = emb  # Use as is if already a list
                    combined_data['embeddings'].extend(emb_data)
                if content:
                    combined_data['content'] += f"\n{content}"
            except Exception as e:
                logger.error(f"Error processing embeddings data: {str(e)}")
                continue

        logger.info(f"Successfully combined {len(combined_data['embeddings'])} embeddings")
        logger.info(f"Combined content length: {len(combined_data['content'])}")
        return combined_data

    except Exception as e:
        logger.error(f"Database error retrieving embeddings: {str(e)}", exc_info=True)
        return None
    finally:
        cur.close()
        close_conn(conn)

def check_topic_embeddings(topic_id):
    """Check if topic has stored embeddings"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT COUNT(*)
            FROM files
            WHERE topic_id = %s
            AND embeddings IS NOT NULL
        """, (topic_id,))
        count = cur.fetchone()[0]
        return count > 0
    finally:
        cur.close()
        close_conn(conn)

@st.fragment
def generate_quiz_and_flashcards(st, model_choice, openai_client, topic_id=None):
    """Generate quiz and flashcards for a topic"""
    logger.info("Entering generate_quiz_and_flashcards function")

    # In generate_quiz_and_flashcards function, replace the Quiz Generation Section with:

    # Quiz Generation Section
    st.markdown("### 📝 Quiz Generation")
    st.write("Generate a new quiz based on the topic content.")

    # Create a form for quiz generation just like flashcards
    with st.form(key=f"quiz_form_{topic_id}"):
        num_questions = st.slider("Number of questions", 
                                min_value=3, max_value=20, value=5)
        submit_button = st.form_submit_button("Generate Quiz", 
                                            use_container_width=True)

        if submit_button:
            logger.info(f"Quiz generation button clicked for topic {topic_id}")
            progress_text = st.empty()
            progress_bar = st.progress(0)

            try:
                progress_text.text("Starting quiz generation...")
                progress_bar.progress(10)

                # Get stored embeddings and content
                topic_data = get_topic_embeddings(topic_id)
                if not topic_data:
                    st.error("No processed content found for this topic.")
                    logger.error(f"No embeddings found for topic {topic_id}")
                    return

                logger.info("Retrieved topic embeddings successfully")
                logger.info(f"Content length: {len(topic_data['content']) if topic_data else 0}")

                
                progress_bar.progress(30)
                progress_text.text("Processing content for quiz generation...")

                # Generate quiz using the content
                logger.info("Starting quiz generation with OpenAI")
                quiz, usage = generate_quiz(
                    st,
                    model_choice,
                    openai_client,
                    num_questions=num_questions,
                    context=topic_data['content']
                )
                logger.info(f"Quiz generation OpenAI call completed: {quiz is not None}")

                progress_bar.progress(60)
                progress_text.text("Quiz generated, saving to database...")
                logger.info(f"Quiz generation completed: {quiz is not None}")

                if quiz and usage:
                    st.session_state.quiz = quiz
                    logger.info("Attempting to save quiz to database")

                    if save_quiz_to_db(quiz, topic_id):
                        progress_bar.progress(100)
                        progress_text.text("Quiz generated successfully!")
                        st.success("Quiz generated and saved successfully!")
                        st.session_state.token_tracker.update(usage, model_choice)

                        # Show preview of generated quiz
                        st.markdown("### 📝 Preview of Generated Quiz")
                        for i, question in enumerate(quiz, 1):
                            with st.expander(f"Question {i}", expanded=True):
                                st.info(question['question'])
                                if question.get('type') == 'multiple-choice':
                                    st.write("Options:")
                                    for opt in question['options']:
                                        st.write(f"- {opt}")
                                st.success(f"Answer: {question.get('correct_answer') or question.get('answer')}")
                    else:
                        progress_bar.progress(100)
                        progress_text.text("Error saving quiz!")
                        st.error("Quiz generated but failed to save to database.")
                        logger.error("Failed to save quiz to database")
                else:
                    progress_bar.progress(100)
                    progress_text.text("Error generating quiz!")
                    st.error("Error generating quiz. Please try again.")
                    logger.error("Quiz generation returned None")
            except Exception as e:
                progress_bar.progress(100)
                progress_text.text("Error in quiz generation!")
                logger.error(f"Error in quiz generation: {str(e)}", exc_info=True)
                st.error(f"Error generating quiz: {str(e)}")

    # Separator between sections
    st.markdown("---")

    # Flashcard Generation Section
    st.markdown("### 🗂️ Flashcard Generation")
    st.write("Generate new flashcards based on the topic content.")

    flashcard_btn_key = f"flashcard_btn_{topic_id}_{int(datetime.now().timestamp())}"

    # Create a form for flashcard generation
    with st.form(key=f"flashcard_form_{topic_id}"):
        num_cards = st.slider("Number of flashcards to generate", 
                            min_value=3, max_value=10, value=5)
        submit_button = st.form_submit_button("Generate Flashcards", 
                                            use_container_width=True)

        if submit_button:
            logger.info(f"Flashcard generation button clicked for topic {topic_id}")
            progress_text = st.empty()
            progress_bar = st.progress(0)

            try:
                progress_text.text("Starting flashcard generation...")
                progress_bar.progress(10)

                # Call the flashcard generation function
                flashcards, usage = generate_flashcards(
                    st,
                    model_choice,
                    openai_client,
                    num_cards=num_cards,
                    topic_id=topic_id
                )

                progress_bar.progress(50)
                progress_text.text("Processing generated flashcards...")

                if flashcards and usage:
                    logger.info(f"Successfully generated {len(flashcards)} flashcards")
                    st.session_state.flashcards = flashcards

                    progress_bar.progress(75)
                    progress_text.text("Saving flashcards to database...")

                    if save_flashcards_to_db(flashcards, topic_id):
                        progress_bar.progress(100)
                        progress_text.text("Flashcards generated successfully!")
                        st.success(f"Successfully generated and saved {len(flashcards)} flashcards!")
                        st.session_state.token_tracker.update(usage, model_choice)

                        # Show preview of generated flashcards
                        st.markdown("### 📝 Preview of Generated Flashcards")
                        for i, card in enumerate(flashcards, 1):
                            st.markdown(f"**Flashcard {i}**")
                            st.info(f"Question: {card['question']}")
                            st.success(f"Answer: {card['answer']}")
                            st.markdown("---")
                    else:
                        progress_bar.progress(100)
                        progress_text.text("Error saving flashcards!")
                        logger.error("Failed to save flashcards to database")
                        st.error("Failed to save flashcards to database.")
                else:
                    progress_bar.progress(100)
                    progress_text.text("Error generating flashcards!")
                    logger.error("Flashcard generation returned no results")
                    st.error("Failed to generate flashcards. Please try again.")

            except Exception as e:
                progress_bar.progress(100)
                progress_text.text("Error in flashcard generation!")
                logger.error(f"Error in flashcard generation: {str(e)}", exc_info=True)
                st.error(f"Error generating flashcards: {str(e)}")


def display_study_materials():
    """Handle study materials generation interface"""
    logger.info("Entering display_study_materials")

    topics = get_topics(st.session_state.current_subject)
    logger.info(f"Found {len(topics) if topics else 0} topics")

    if not topics:
        st.warning("Create a topic first and upload content.")
        return

    selected_topic = st.selectbox(
        "Select Topic",
        options=topics,
        format_func=lambda x: f"{x['emoji']} {x['name']}",
        key="study_materials_topic_select"
    )
    logger.info(f"Selected topic ID: {selected_topic['id']}")

    # Check if topic has content
    files = get_files_for_topic(selected_topic['id'])
    logger.info(f"Found {len(files)} files for topic")

    if not files:
        st.warning("Upload content for this topic first.")
        return

    # Show content status
    embeddings_exist = check_topic_embeddings(selected_topic['id'])
    logger.info(f"Embeddings exist: {embeddings_exist}")

    if embeddings_exist:
        st.success("✅ Content processed and ready for generation")

        # Display existing materials counts
        col1, col2 = st.columns(2)
        with col1:
            quiz_count = get_topic_quiz_count(selected_topic['id'])
            logger.info(f"Existing quizzes: {quiz_count}")
            st.metric("Existing Quizzes", quiz_count)
        with col2:
            flashcard_count = get_topic_flashcard_count(selected_topic['id'])
            logger.info(f"Existing flashcards: {flashcard_count}")
            st.metric("Existing Flashcards", flashcard_count)

        st.markdown("---")

        # Call generation function with debug logging
        logger.info("Calling generate_quiz_and_flashcards")
        generate_quiz_and_flashcards(
            st,
            st.session_state.model_choice,
            st.session_state.openai_client,
            selected_topic['id']
        )
        logger.info("Finished generate_quiz_and_flashcards call")
    else:
        st.warning("⚠️ Process content first before generating materials")

def get_topic_quiz_count(topic_id):
    """Get count of existing quizzes for topic"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT COUNT(*)
            FROM quizzes
            WHERE topic_id = %s
        """, (topic_id,))
        return cur.fetchone()[0]
    finally:
        cur.close()
        close_conn(conn)

def get_topic_flashcard_count(topic_id):
    """Get count of existing flashcards for topic"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT COUNT(*)
            FROM flashcards
            WHERE topic_id = %s
        """, (topic_id,))
        return cur.fetchone()[0]
    finally:
        cur.close()
        close_conn(conn)

def display_existing_materials(topic_id):
    """Display existing quizzes and flashcards for the topic"""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Quiz Esistenti")
        existing_quizzes = get_topic_quizzes(topic_id)
        if existing_quizzes:
            for quiz in existing_quizzes:
                st.write(f"Quiz creato il: {quiz['created_at'].strftime('%d/%m/%Y')}")
        else:
            st.info("Nessun quiz esistente per questo argomento.")

    with col2:
        st.subheader("🗂️ Flashcard Esistenti")
        existing_flashcards = get_topic_flashcards(topic_id)
        if existing_flashcards:
            st.write(f"Totale flashcard: {len(existing_flashcards)}")
        else:
            st.info("Nessuna flashcard esistente per questo argomento.")

def get_topic_quizzes(topic_id):
    """Get existing quizzes for a topic"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT quiz_id, created_at, completion_rate, average_score
            FROM quizzes
            WHERE topic_id = %s AND username = %s
            ORDER BY created_at DESC
        """, (topic_id, st.session_state.username))
        return cur.fetchall()
    finally:
        cur.close()
        close_conn(conn)

def get_topic_flashcards(topic_id):
    """Get existing flashcards for a topic"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT flashcard_id, created_at
            FROM flashcards
            WHERE topic_id = %s AND username = %s
            ORDER BY created_at DESC
        """, (topic_id, st.session_state.username))
        return cur.fetchall()
    finally:
        cur.close()
        close_conn(conn)

def save_quiz_to_db(quiz, topic_id):
    """Save generated quiz to database"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        now = datetime.now(timezone.utc)
        logger.info(f"Saving quiz for topic {topic_id} with {len(quiz)} questions")

        # Ensure quiz_data is stored as JSON string
        quiz_data_json = json.dumps(quiz)

        # Insert quiz
        cur.execute("""
            INSERT INTO quizzes (
                username, 
                topic_id,
                quiz_data,
                status,
                total_questions,
                created_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING quiz_id
        """, (
            st.session_state.username,
            topic_id,
            quiz_data_json,  # Store as JSON string
            'ready',
            len(quiz),
            now
        ))
        quiz_id = cur.fetchone()[0]
        logger.info(f"Created quiz with ID: {quiz_id}")

        # Create individual questions records
        for i, question in enumerate(quiz):
            options_json = json.dumps(question.get('options', []))  # Ensure options are JSON string
            cur.execute("""
                INSERT INTO quiz_questions (
                    quiz_id,
                    question_number,
                    question,
                    correct_answer,
                    options,
                    question_type,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                quiz_id,
                i + 1,
                question['question'],
                question.get('correct_answer') or question.get('answer'),
                options_json,
                question.get('type', 'open-ended'),
                now
            ))
            logger.info(f"Saved question {i+1} for quiz {quiz_id}")

        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving quiz to database: {str(e)}")
        return False
    finally:
        cur.close()
        close_conn(conn)

def calculate_quiz_priority(topic_id):
    """Calculate quiz priority based on topic deadline"""
    if not topic_id:
        return 2  # Default medium priority

    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT end_date
            FROM topics
            WHERE topic_id = %s
        """, (topic_id,))
        result = cur.fetchone()
        if not result:
            return 2

        end_date = result[0]
        if not end_date:
            return 2

        days_until_end = (end_date - datetime.now().date()).days

        if days_until_end <= 7:  # Week or less until deadline
            return 1  # High priority
        elif days_until_end <= 30:  # Month or less
            return 2  # Medium priority
        else:
            return 3  # Low priority
    except Exception as e:
        logger.error(f"Error calculating quiz priority: {str(e)}")
        return 2  # Default to medium priority
    finally:
        cur.close()
        close_conn(conn)

def save_flashcards_to_db(flashcards, topic_id=None):
    """Save generated flashcards to database with FSRS parameters"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        priority = calculate_quiz_priority(topic_id) if topic_id else 2
        now = datetime.now(timezone.utc)

        for flashcard in flashcards:
            # Create new card with initial state
            cur.execute("""
                INSERT INTO flashcards (
                    username,
                    topic_id,
                    question,
                    answer,
                    state,
                    difficulty,
                    stability,
                    reps,
                    lapses,
                    last_review,
                    next_review,
                    reviewed_today,
                    priority,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING flashcard_id
            """, (
                st.session_state.username,
                topic_id,
                flashcard['question'],
                flashcard['answer'],
                State.New.value,  # Initial state
                5.0,  # Initial difficulty
                1.0,  # Initial stability
                0,    # No reps yet
                0,    # No lapses
                None, # No last review
                now,  # Due immediately
                False, # Not reviewed
                priority,
                now
            ))

            flashcard_id = cur.fetchone()[0]

            # Create initial review log
            cur.execute("""
                INSERT INTO flashcard_reviews (
                    flashcard_id,
                    username,
                    rating,
                    review_time,
                    response_time,
                    review_state,
                    is_initial_review,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                flashcard_id,
                st.session_state.username,
                0,  # No rating for initial creation
                now,
                None,
                State.New.value,
                True,  # Mark as initial review
                now
            ))

        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving flashcards to database: {str(e)}")
        return False
    finally:
        cur.close()
        close_conn(conn)