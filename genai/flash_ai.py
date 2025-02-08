# genai/flash_ai.py
from .rag_pipeline import retrieve_relevant_docs
from utils.db_connection import get_conn, close_conn
import json
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# flashcards page
def generate_flashcards(st, model_choice, openai_client, num_cards=5, topic_id=None):
    """
    Generates flashcards based on the processed study materials.
    """
    logger.info(f"Starting flashcard generation for topic_id: {topic_id}")

    # Retrieve user preferences from session state
    subject = st.session_state.get('subject')
    class_year = st.session_state.get('class_year')
    language = st.session_state.get('language')
    difficulty = st.session_state.get('difficulty')

    logger.info(f"Retrieved preferences: subject={subject}, year={class_year}, language={language}, difficulty={difficulty}")

    try:
        # Get content from database instead of RAG pipeline
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT processed_content
            FROM files
            WHERE topic_id = %s
        """, (topic_id,))
        results = cur.fetchall()

        if not results:
            logger.error("No content found for topic")
            return None, None

        # Combine all content
        context = "\n".join(row[0] for row in results if row[0])
        logger.info(f"Retrieved content length: {len(context)}")

        # Define the system and user messages
        messages = [
            {
                "role": "system",
                "content": (
                    f"Sei un esperto creatore di flashcard per {subject} a livello {difficulty} "
                    f"per studenti del {class_year}º anno delle scuole superiori in Italia. "
                    f"Crea flashcard basate sul contesto fornito. Ogni flashcard dovrebbe avere una domanda e una risposta. Rispondi in {language}."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Genera {num_cards} flashcard basate sul seguente contesto. "
                    "Formatta l'output come una lista JSON di dizionari, dove ogni dizionario ha le chiavi 'question' e 'answer'.\n\n"
                    f"Contesto: {context}"
                )
            }
        ]

        logger.info("Calling OpenAI API...")
        response = openai_client.chat.completions.create(
            model=model_choice,
            messages=messages,
            max_tokens=1500,
            n=1,
            stop=None,
            temperature=0.7,
        )

        content = response.choices[0].message.content.strip()
        logger.info("Received response from OpenAI")

        # Clean and parse response
        if content.startswith("```") and content.endswith("```"):
            content = content.strip("```")
        if content.lower().startswith("json"):
            content = content[4:].strip()

        flashcards = json.loads(content)
        if not isinstance(flashcards, list):
            raise ValueError("Unexpected format received from AI model")

        usage = response.usage.model_dump()
        logger.info(f"Generated {len(flashcards)} flashcards successfully")

        return flashcards, usage

    except Exception as e:
        logger.error(f"Error generating flashcards: {str(e)}", exc_info=True)
        return None, None

    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            close_conn(conn)

def explain_flashcard(openai_client, model_choice, subject, difficulty, class_year, language, flashcard, context):
    prompt = f"Spiegare in dettaglio il concetto presentato in questa flashcard per {subject} a livello {difficulty} per studenti del {class_year}º anno. Flashcard: {flashcard['question']} - {flashcard['answer']}. Contesto: {context}"
    return generate_ai_response(openai_client, model_choice, prompt, subject, difficulty, class_year, language)
    
def provide_example(openai_client, model_choice, subject, difficulty, class_year, language, flashcard, context):
    prompt = f"Fornire un esempio pratico o un'applicazione del concetto presentato in questa flashcard per {subject} a livello {difficulty} per studenti del {class_year}º anno. Flashcard: {flashcard['question']} - {flashcard['answer']}. Contesto: {context}"
    return generate_ai_response(openai_client, model_choice, prompt, subject, difficulty, class_year, language)
    
def create_mnemonic(openai_client, model_choice, subject, difficulty, class_year, language, flashcard, context):
    prompt = f"Crea una tecnica mnemonica creativa e convolgente per aiutare gli studenti a ricordare facilmente il concetto presentato in questa flashcard.  Fai riferimento a esempi della vita quotidiana o ad associazioni divertenti per {subject} a livello {difficulty}, per studenti del {class_year}º anno. Flashcard: {flashcard['question']} - {flashcard['answer']}. Contesto: {context}"
    return generate_ai_response(openai_client, model_choice, prompt, subject, difficulty, class_year, language)
    
def generate_ai_response(openai_client, model_choice, prompt, subject, difficulty, class_year, language):
    messages = [
        {
            "role": "system",
            "content": f"Sei un tutor AI esperto in {subject} a livello {difficulty} per studenti del {class_year}º anno delle scuole superiori in Italia. Rispondi in {language} in modo conciso, con risposte brevi di 2-3 frasi."
        },
        {"role": "user", "content": prompt}
    ]
    try:
        response = openai_client.chat.completions.create(
            model=model_choice,
            messages=messages,
            max_completion_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )
        content = response.choices[0].message.content.strip()
        usage = response.usage.model_dump()
        return {'content': content, 'usage': usage}
    except Exception as e:
        logger.error(f"Error in generate_ai_response: {str(e)}", exc_info=True)
        return {'content': "Si è verificato un errore nella generazione della risposta.", 'usage': {}}

def flash_chat(user_input, rag_pipeline, model_choice, openai_client, subject, class_year, language, difficulty, conversation_history, current_flashcard):
    # Retrieve relevant documents
    relevant_docs = retrieve_relevant_docs(rag_pipeline, user_input, k=5)
    context = " ".join(doc.page_content for doc in relevant_docs)

    # Create a system message that includes the current flashcard context
    system_message = f"""Sei un tutor AI esperto in {subject} a livello {difficulty} per studenti del {class_year}º anno delle scuole superiori in Italia. 
    Rispondi in {language}. La domanda dell'utente si riferisce alla seguente flashcard:
    Domanda: {current_flashcard['question']}
    Risposta: {current_flashcard['answer']}
    Usa queste informazioni come contesto principale per rispondere alla domanda dell'utente, integrando con il contesto aggiuntivo fornito se necessario."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Contesto aggiuntivo: {context}\n\nDomanda dell'utente: {user_input}"}
    ]

    # Add conversation history
    for msg in conversation_history:
        messages.append({"role": msg[0], "content": msg[1]})

    try:
        response = openai_client.chat.completions.create(
            model=model_choice,
            messages=messages,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )
        content = response.choices[0].message.content.strip()
        usage = response.usage.model_dump()
        return content, usage
    except Exception as e:
        logger.error(f"Error in flash_chat: {str(e)}", exc_info=True)
        return "Si è verificato un errore nella generazione della risposta.", {}