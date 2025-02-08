# quiz_ai.py
from .rag_pipeline import retrieve_relevant_docs
import json
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_quiz(st, model_choice, openai_client, num_questions=5, context=None):
    """
    Generates a quiz based on the processed study materials.
    Args:
        context: The processed content to generate quiz from
    Returns:
        list: A list of quiz questions.
        dict: Usage statistics from the OpenAI API call.
    """
    try:
        if not context:
            st.error("No content provided for quiz generation.")
            return None, {}

        # Define the system and user messages for the OpenAI API
        messages = [
            {
                "role": "system",
                "content": (
                    f"Sei un esperto creatore di quiz per {st.session_state.subject} a livello {st.session_state.difficulty} "
                    f"per studenti del {st.session_state.class_year}º anno delle scuole superiori in Italia. "
                    f"Crea un quiz basato sul contesto fornito. Includi un mix di domande a scelta multipla e aperte. "
                    f"Per le domande a scelta multipla, fornisci sempre 4 opzioni. Per le domande aperte, fornisci una risposta modello. "
                    f"Rispondi in {st.session_state.language}."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Genera un quiz con {num_questions} domande basato sul seguente contesto. "
                    "Formatta l'output come una lista JSON di dizionari, dove ogni dizionario rappresenta una domanda. "
                    "Usa la seguente struttura:\n\n"
                    "Per domande a scelta multipla:\n"
                    "{\n"
                    '    "question": "Testo della domanda",\n'
                    '    "type": "multiple-choice",\n'
                    '    "options": ["Opzione 1", "Opzione 2", "Opzione 3", "Opzione 4"],\n'
                    '    "correct_answer": "L\'opzione corretta"\n'
                    "}\n\n"
                    "Per domande aperte:\n"
                    "{\n"
                    '    "question": "Testo della domanda",\n'
                    '    "type": "open-ended",\n'
                    '    "answer": "Risposta modello"\n'
                    "}\n\n"
                    f"Contesto: {context}"
                )
            }
        ]

        # Call the OpenAI API to generate the quiz
        response = openai_client.chat.completions.create(
            model=model_choice,
            messages=messages,
            max_completion_tokens=1500,
            n=1,
            stop=None,
            temperature=0.7,
        )

        content = response.choices[0].message.content.strip()

        # Clean and parse the response
        if content.startswith("```") and content.endswith("```"):
            content = content.strip("```")
        if content.lower().startswith("json"):
            content = content[4:].strip()

        quiz = json.loads(content)
        if not isinstance(quiz, list):
            raise ValueError("Unexpected format received from AI model")

        usage = response.usage.model_dump()
        return quiz, usage

    except Exception as e:
        logger.error(f"Error in generate_quiz: {str(e)}", exc_info=True)
        st.error(f"Error generating quiz: {str(e)}")
        return None, {}

def get_quiz_hint(question, rag_pipeline, model_choice, openai_client, subject, class_year, language, difficulty):
    """
    Provides a hint for a specific quiz question.
    Returns:
        str: The assistant's hint.
        dict: Usage statistics from the OpenAI API call.
    """
    context = f"Domanda: {question['question']}\n"
    if 'options' in question:
        context += f"Opzioni: {', '.join(question['options'])}\n"
    messages = [
        {
            "role": "system",
            "content": (
                f"Sei un tutor AI che fornisce suggerimenti per una domanda di {subject} a livello {difficulty} "
                f"per studenti del {class_year}º anno delle scuole superiori in Italia. Guida lo studente verso la risposta "
                f"senza rivelarla direttamente. Rispondi in {language}."
            )
        },
        {
            "role": "user",
            "content": f"Contesto: {context}\n\nFornisci un suggerimento per questa domanda:"
        }
    ]
    try:
        response = openai_client.chat.completions.create(
            model=model_choice,
            messages=messages,
            max_completion_tokens=200,
            n=1,
            stop=None,
            temperature=0.7,
        )
        hint = response.choices[0].message.content.strip()
        # Extract usage
        usage = {
            'total_tokens': response.usage.total_tokens,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
        }
        return hint, usage
    except Exception as e:
        st.error(f"Errore durante la generazione del suggerimento: {e}")
        return "Si è verificato un errore nel tutor AI.", {}

def quiz_review_chatbot(st, rag_pipeline, model_choice, openai_client, quiz, user_answers, subject, class_year, language, difficulty, user_input):
    """
    Handles the quiz review chat using the Socratic method.
    Returns:
        str: The assistant's response.
        dict: Usage statistics from the OpenAI API call.
    """
    # Retrieve relevant documents using RAG
    relevant_docs = retrieve_relevant_docs(rag_pipeline, user_input, k=5)
    rag_context = " ".join(doc.page_content for doc in relevant_docs)
    context = (
        f"Materia: {subject}\n"
        f"Difficoltà: {difficulty}\n"
        f"Anno Scolastico: {class_year}\n"
        f"Lingua: {language}\n\n"
        "Domande del Quiz e Risposte:\n"
    )
    for i, question in enumerate(quiz, 1):
        context += f"D{i}: {question['question']}\n"
        context += f"Tua Risposta: {user_answers.get(i, 'Non risposta')}\n"
        if 'answer' in question:
            context += f"Risposta Corretta: {question['answer']}\n"
        elif 'correct_answer' in question:
            context += f"Risposta Corretta: {question['correct_answer']}\n"
        context += "\n"
    # Add RAG context to the existing context
    context += f"\nContesto aggiuntivo dai materiali di studio:\n{rag_context}\n"
    messages = [
        {
            "role": "system",
            "content": (
                f"Sei un tutor AI che rivede un quiz di {subject} a livello {difficulty} per studenti del {class_year}º anno delle scuole superiori in Italia. "
                f"Incoraggia lo studente a ragionare sugli errori prima di fornire risposte complete. Usa il metodo socratico quando appropriato. Rispondi in {language}."
            )
        },
        {
            "role": "user",
            "content": f"Contesto: {context}\n\nDomanda dello studente: {user_input}"
        }
    ]
    try:
        response = openai_client.chat.completions.create(
            model=model_choice,
            messages=messages,
            max_completion_tokens=700,
            n=1,
            stop=None,
            temperature=0.7,
        )
        review_response = response.choices[0].message.content.strip()
        # Extract usage
        usage = {
            'total_tokens': response.usage.total_tokens,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
        }
        return review_response, usage
    except Exception as e:
        st.error(f"Errore durante la revisione del quiz: {e}")
        return "Si è verificato un errore nel tutor AI.", {}
