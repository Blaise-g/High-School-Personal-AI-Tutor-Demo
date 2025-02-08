# tutor_ai.py
from .rag_pipeline import retrieve_relevant_docs
import json
import streamlit as st
import logging
import networkx as nx
from pyvis.network import Network

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_summary(openai_client, model_choice, rag_pipeline, subject, class_year, difficulty, language):
    try:
        logger.info(f"Starting summary generation for {subject}")
        query = f"Genera un riassunto completo per {subject}"
        relevant_docs = retrieve_relevant_docs(rag_pipeline, query, k=10)
        logger.info(f"Retrieved {len(relevant_docs)} relevant documents for summary")
        additional_context = " ".join(doc.page_content for doc in relevant_docs)
        messages = [
            {
                "role": "system",
                "content": (
                    f"""Sei un tutor AI esperto in {subject} per studenti del {class_year}º anno delle scuole superiori in Italia,
                    con competenze di livello {difficulty}. Il tuo compito è creare un riassunto completo, chiaro e ben strutturato del materiale fornito. 
                    Ogni sezione deve avere un titolo descrittivo e coprire le informazioni chiave in modo facile da comprendere.
                    Includi esempi rilevanti, storie o collegamenti a concetti più ampi per facilitare la comprensione degli studenti.
                    Assicurati che il riassunto sia suddiviso in sezioni con intestazioni significative, usando un linguaggio accessibile e coinvolgente."
                    Rispondi in {language}."""
                )
            },
            {
                "role": "user",
                "content": f"Per favore, genera un riassunto dettagliato del materiale di studio.\n\nContesto aggiuntivo: {additional_context}"
            }
        ]
        logger.info(f"Sending request to OpenAI for summary generation")
        response = openai_client.chat.completions.create(
            model=model_choice,
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )
        summary = response.choices[0].message.content.strip()
        usage = response.usage.model_dump()
        logger.info(f"Summary generated successfully. Usage: {usage}")
        return summary, usage
    except Exception as e:
        logger.error(f"Error in generate_summary: {str(e)}", exc_info=True)
        return "Si è verificato un errore nella generazione del riassunto.", {}


def generate_image(openai_client, summary, subject, class_year, difficulty):
    try:
        logger.info("Starting image generation")
        prompt = (
            f"Crea un'immagine educativa e visivamente accattivante che rappresenti i concetti chiave di {subject} per studenti del {class_year}º anno delle scuole superiori. "
            f"Livello di difficoltà: {difficulty}. Contenuto da rappresentare:\n\n{summary}"
        )
        logger.info("Sending request to OpenAI for image generation")
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        logger.info("Image generated successfully")
        return image_url
    except Exception as e:
        logger.error(f"Error in generate_image: {str(e)}", exc_info=True)
        return None


def tutor_ai_chat(st, rag_pipeline, model_choice, openai_client, subject, class_year, language, difficulty, user_input, chat_history):
    try:
        logger.info(f"Starting tutor_ai_chat with input: {user_input}")
        relevant_docs = retrieve_relevant_docs(rag_pipeline, user_input, k=5)
        logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
        additional_context = " ".join(doc.page_content for doc in relevant_docs)
        messages = [
            {
                "role": "system",
                "content": (
                    f"Sei un tutor AI esperto in {subject} a livello {difficulty} per studenti del {class_year}º anno delle scuole superiori in Italia. "
                    f"Usa il metodo socratico per guidare lo studente verso la comprensione quando appropriato. Rispondi in {language}. "
                    f"Sii di supporto, incoraggiante e attento al benessere dello studente."
                )
            }
        ]
        for message in chat_history:
            messages.append({"role": message['role'], "content": message['content']})
        messages.append({"role": "user", "content": f"{user_input}\n\nContesto aggiuntivo: {additional_context}"})
        logger.info(f"Sending request to OpenAI with {len(messages)} messages")
        response = openai_client.chat.completions.create(
            model=model_choice,
            messages=messages,
            max_completion_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7,
        )
        logger.info("Received response from OpenAI")
        assistant_response = response.choices[0].message.content.strip()
        usage = {
            'total_tokens': response.usage.total_tokens,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
        }
        logger.info(f"Chat completed successfully. Usage: {usage}")
        return {
            'content': assistant_response,
            'usage': usage
        }
    except Exception as e:
        logger.error(f"Error in tutor_ai_chat: {str(e)}", exc_info=True)
        return {
            'content': "Si è verificato un errore nel tutor AI.",
            'usage': {}
        }

