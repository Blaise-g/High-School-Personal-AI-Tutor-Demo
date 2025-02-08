# sections/ai_tutor_page.py
import streamlit as st
from genai.tutor_ai import tutor_ai_chat, generate_summary, generate_image
from utils.usage_db import log_usage
from utils.feedback_db import init_feedback_db, log_feedback, get_feedback_summary
from genai.rag_pipeline import retrieve_relevant_docs, summarize_conversation, process_documents_parallel
import datetime
import re
import json
import base64
from pyvis.network import Network
from fpdf import FPDF
from pydantic import BaseModel, Field
from typing import List, Optional
from itertools import cycle
import logging

class SubSubBranch(BaseModel):
    name: str
    details: Optional[str] = Field(None, description="Additional details for the sub-sub-branch.")

class SubBranch(BaseModel):
    name: str
    details: Optional[str] = Field(None, description="Additional information for the sub-branch.")
    sub_sub_branches: Optional[List[SubSubBranch]] = Field(default=None, description="A list of sub-sub-branches, if available.")

class Branch(BaseModel):
    name: str
    color: str
    details: Optional[str] = Field(None, description="Extra description or details for each main branch.")
    sub_branches: Optional[List[SubBranch]] = Field(default=None, description="A list of sub-branches, if available.")

class ConceptMap(BaseModel):
    main_topic: str
    branches: List[Branch]  # This must be a list, and it cannot be empty.

def generate_structured_concept_map(openai_client, model_choice, subject, difficulty, class_year, language, summary):
    # Define the system message with a detailed prompt
    system_message = f"""Sei un esperto nella creazione di mappe concettuali educative per il tema '{subject}', con un livello di difficoltà adatto per studenti del {class_year}º anno delle scuole superiori in Italia.
    Il tuo compito è analizzare il riassunto fornito e distillare le informazioni in una struttura gerarchica chiara e ben articolata. 
    La mappa concettuale deve essere composta da:
    - Un argomento principale che rappresenta il tema centrale.
    - Diverse ramificazioni principali che rappresentano le sezioni chiave del contenuto.
    - Ogni ramo principale dovrebbe avere più livelli di sotto-ramificazioni, ciascuno contenente dettagli e concetti specifici.
    - Ciascuna sotto-ramificazione dovrebbe essere approfondita se ci sono ulteriori dettagli da esplorare, con livelli aggiuntivi di informazioni.
    - I livelli inferiori devono contenere frasi di spiegazione per descrivere ogni concetto in modo che gli studenti possano comprenderlo meglio.
    Organizza le informazioni in modo gerarchico, utilizzando vari livelli per assicurarti che la mappa sia chiara, completa e interessante per gli studenti.
    Rispondi in {language}."""

    # Prepare the messages for the request
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Crea una mappa concettuale dettagliata sulla base del seguente riassunto:\n\n{summary}"}
    ]

    try:
        # Add logging to understand the schema being used
        logging.info("Sending request to OpenAI API with the following schema:")
        logging.info(json.dumps(ConceptMap.schema(), indent=4))

        # Send the request using the structured response format with the Pydantic model as a schema
        completion = openai_client.beta.chat.completions.parse(
            model=model_choice,
            messages=messages,
            response_format=ConceptMap,  # Using the schema defined with Pydantic.
            max_tokens=2000,
            temperature=0.7,
        )

        # Parse the structured response to get the concept map
        concept_map = completion.choices[0].message.parsed
        return concept_map
    except Exception as e:
        st.error(f"Error in generating structured concept map: {str(e)}")
        logging.error(f"Error details: {str(e)}")
        return None

#@st.cache_resource
def visualize_structured_concept_map(concept_map: ConceptMap):
    if not concept_map:
        st.error("Invalid concept map data structure")
        return "<p>Unable to visualize the concept map due to invalid data.</p>"

    # Create the network
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black")

    # Define a set of colors to cycle through for branches
    color_palette = cycle(["#FF6F61", "#6B5B95", "#88B04B", "#F7CAC9", "#92A8D1", "#F4E1D2", "#FFCC5C", "#D4A5A5"])

    # Ensure that the main topic node is added first
    main_node_id = "0"
    logging.info(f"Adding main topic node: {main_node_id} - {concept_map.main_topic}")
    net.add_node(main_node_id, label=concept_map.main_topic, color="#FFFFFF", size=35)

    # Function to recursively add branches and sub-branches
    def add_branch_with_subbranches(branch, parent_id, current_level):
        # Each branch has a unique ID composed of parent_id and current_level to ensure no conflicts
        branch_color = next(color_palette)
        branch_id = f"{parent_id}.{current_level}"

        logging.info(f"Adding branch node: {branch_id} - {branch.name}")
        net.add_node(branch_id, label=branch.name, color=branch_color, size=max(30 - current_level * 3, 10))
        net.add_edge(parent_id, branch_id)  # Connect current branch to its parent

        # If there are details for this branch, add them as tooltip or description boxes
        if branch.details:
            details_node_id = f"{branch_id}.details"
            logging.info(f"Adding branch details node: {details_node_id} - {branch.details}")
            net.add_node(details_node_id, label=branch.details, color="#C0C0C0", size=max(25 - current_level * 3, 8), shape='box')
            net.add_edge(branch_id, details_node_id)

        # Add sub-branches recursively
        for idx, sub_branch in enumerate(branch.sub_branches or [], start=1):
            sub_branch_id = f"{branch_id}.{idx}"
            logging.info(f"Adding sub-branch node: {sub_branch_id} - {sub_branch.name}")
            net.add_node(sub_branch_id, label=sub_branch.name, color=branch_color, size=max(28 - (current_level + 1) * 3, 8))
            net.add_edge(branch_id, sub_branch_id)

            if sub_branch.details:
                sub_branch_details_id = f"{sub_branch_id}.details"
                logging.info(f"Adding sub-branch details node: {sub_branch_details_id} - {sub_branch.details}")
                net.add_node(sub_branch_details_id, label=sub_branch.details, color="#E0E0E0", size=max(20 - (current_level + 1) * 3, 6), shape='box')
                net.add_edge(sub_branch_id, sub_branch_details_id)

            # Add sub-sub-branches if they exist
            for sub_sub_idx, sub_sub_branch in enumerate(sub_branch.sub_sub_branches or [], start=1):
                sub_sub_branch_id = f"{sub_branch_id}.{sub_sub_idx}"
                logging.info(f"Adding sub-sub-branch node: {sub_sub_branch_id} - {sub_sub_branch.name}")
                net.add_node(sub_sub_branch_id, label=sub_sub_branch.name, color=branch_color, size=max(25 - (current_level + 2) * 3, 6))
                net.add_edge(sub_branch_id, sub_sub_branch_id)

                if sub_sub_branch.details:
                    sub_sub_branch_details_id = f"{sub_sub_branch_id}.details"
                    logging.info(f"Adding sub-sub-branch details node: {sub_sub_branch_details_id} - {sub_sub_branch.details}")
                    net.add_node(sub_sub_branch_details_id, label=sub_sub_branch.details, color="#F0F0F0", size=max(18 - (current_level + 2) * 3, 5), shape='box')
                    net.add_edge(sub_sub_branch_id, sub_sub_branch_details_id)

    # Start adding branches from level 1
    for idx, branch in enumerate(concept_map.branches, start=1):
        add_branch_with_subbranches(branch, main_node_id, idx)

    # Toggle physics for better visualization and add control buttons
    net.toggle_physics(True)
    net.show_buttons(filter_=['physics'])

    # Generate and return the HTML for rendering the concept map
    return net.generate_html()

#@st.cache_data(ttl=3600)
def create_pdf(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    return pdf.output(dest="S").encode("latin-1")

@st.fragment
def display_chat_interface(st, model_choice, openai_client):
    st.subheader(f"💬 Chatta con il tuo Tutor di {st.session_state.subject}")
    # Create a container for the chat history
    chat_container = st.container()
    # Create a fixed input area at the bottom
    with st.container():
        user_input = st.text_input("La tua domanda:", key="ai_tutor_input")
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            send_button = st.button("Invia", key="ai_tutor_send")
        with col2:
            if st.button("🔄 Resetta Chat"):
                st.session_state.tutor_chat_history = []
                st.rerun()
    # Display chat history
    with chat_container:
        for idx, message in enumerate(st.session_state.tutor_chat_history):
            if message["role"] == "user":
                st.markdown(f"🙋‍♂️ **Tu**: {message['content']}")
            else:
                st.markdown(f"🤖 **Tutor AI**: {message['content']}")
                # Add feedback for each AI response
                with st.expander("📢 Feedback sulla risposta", expanded=False):
                    display_chat_feedback(st, idx, message, model_choice)
    # Handle user input
    if send_button and user_input:
        st.session_state.tutor_chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Il tutor sta elaborando la tua domanda..."):
            response = tutor_ai_chat(
                st,
                st.session_state.rag_pipeline,
                model_choice,
                openai_client,
                st.session_state.subject,
                st.session_state.class_year,
                st.session_state.language,
                st.session_state.difficulty,
                user_input,
                st.session_state.tutor_chat_history
            )
            if 'usage' in response and response['usage']:
                st.session_state.token_tracker.update(response['usage'], model_choice)
                cost = st.session_state.token_tracker.calculate_cost(response['usage'], model_choice)
                log_usage(st.session_state.username, model_choice, response['usage'], cost)
            ai_response = response.get('content', "Si è verificato un errore nel tutor AI.")
            st.session_state.tutor_chat_history.append({"role": "assistant", "content": ai_response})
        st.rerun()


#@st.fragment
def display_formatted_summary(summary):
    st.header("📚 Riassunto")
    sections = summary.split('\n\n')

    # Create a sidebar table of contents for better accessibility
    with st.sidebar:  # Move the TOC to the sidebar for improved usability
        st.subheader("Indice")
        for i, section in enumerate(sections):
            if section.strip():
                title = section.split('\n')[0].strip('#').strip()
                st.markdown(f"- [{title}](#section-{i})")

    # Horizontal line for separation of content
    st.markdown("---")

    # Display summary content with expandable sections for better readability
    for i, section in enumerate(sections):
        if section.strip():
            with st.expander(f"{section.splitlines()[0].strip('#').strip()}"):
                st.markdown(section)
                st.markdown(f"<div id='section-{i}'></div>", unsafe_allow_html=True)
            st.markdown("---")

#@st.fragment
#def display_concept_map(concept_map_html):
    #st.header("🗺️ Mappa Concettuale")
    #st.components.v1.html(concept_map_html, height=600)

def ai_tutor_page(model_choice, openai_client):
    st.title("💡 Tutor AI")
    if st.session_state.rag_pipeline is None:
        st.warning("Per favore, elabora prima i tuoi materiali di studio nella pagina principale.")
        return

    # Generate and summary
    if st.button("Genera Riassunto"):
        with st.spinner("Generando riassunto..."):
            summary, usage = generate_summary(openai_client, model_choice, st.session_state.rag_pipeline, 
                                              st.session_state.subject, st.session_state.class_year,
                                              st.session_state.difficulty, st.session_state.language)
            st.session_state.summary = summary
            log_usage(st.session_state.username, model_choice, usage, 
                      st.session_state.token_tracker.calculate_cost(usage, model_choice))
    
        # Display formatted summary
        display_formatted_summary(st.session_state.summary)
    
    # Generate image
    if st.button("Genera Immagine Riassuntiva"):
        with st.spinner("Generando l'immagine riassuntiva..."):
            image_url = generate_image(
                openai_client, 
                st.session_state.rag_pipeline, 
                st.session_state.summary,
                st.session_state.subject,
                st.session_state.class_year,
                st.session_state.difficulty
            )
            st.session_state.summary_image = image_url
    
        if 'summary_image' in st.session_state:
            st.image(st.session_state.summary_image, use_column_width=True)

    # Generate and display concept map
    if st.button("Genera Mappa Concettuale"):
        with st.spinner("Generando la mappa concettuale..."):
            concept_map = generate_structured_concept_map(openai_client, model_choice, st.session_state.subject,
                                                            st.session_state.difficulty,
                                                          st.session_state.class_year, st.session_state.language, st.session_state.summary)
            if concept_map:
                st.session_state.concept_map = concept_map
                st.success("Mappa concettuale generata con successo!")
            else:
                st.error("Si è verificato un errore nella generazione della mappa concettuale.")
    
    if 'concept_map' in st.session_state:
        st.header("🗺️ Mappa Concettuale")
        concept_map_html = visualize_structured_concept_map(st.session_state.concept_map)
        st.components.v1.html(concept_map_html, height=600)
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            summary_pdf = create_pdf(st.session_state.summary)
            st.download_button(
                label="Scarica Riassunto (PDF)",
                data=summary_pdf,
                file_name="riassunto.pdf",
                mime="application/pdf"
            )
        with col2:
            st.download_button(
                label="Scarica Mappa Concettuale (HTML)",
                data=concept_map_html,
                file_name="mappa_concettuale.html",
                mime="text/html"
            )
    
    if 'tutor_chat_history' not in st.session_state:
        st.session_state.tutor_chat_history = []
    if 'current_input' not in st.session_state:
        st.session_state.current_input = ""
    # Display token usage and cost in the sidebar
    display_token_usage(st)
    
    # Main chat interface
    display_chat_interface(st, model_choice, openai_client)
    # Suggested topics
    st.subheader("🚀 Argomenti Suggeriti")
    suggested_topics = [
        "Spiegami i concetti chiave dei materiali caricati",
        "Quali sono le cose importanti da ricordare?",
        "Quali sono gli errori comuni da evitare?",
        "Come si applica questo concetto nella vita reale?",
        "Puoi riassumere l'argomento per me?",
        "Come posso prepararmi per l'esame su questo argomento?",
    ]
    cols = st.columns(2)
    for i, topic in enumerate(suggested_topics):
        if cols[i % 2].button(topic, key=f"topic_{i}"):
            st.session_state.current_input = topic
            st.session_state.tutor_chat_history.append({"role": "user", "content": topic})
            with st.spinner("Il tutor sta elaborando la tua domanda..."):
                response = tutor_ai_chat(
                    st,
                    st.session_state.rag_pipeline,
                    model_choice,
                    openai_client,
                    st.session_state.subject,
                    st.session_state.class_year,
                    st.session_state.language,
                    st.session_state.difficulty,
                    topic,
                    st.session_state.tutor_chat_history
                )
                # Update token tracker and log usage
                if 'usage' in response and response['usage']:
                    st.session_state.token_tracker.update(response['usage'], model_choice)
                    cost = st.session_state.token_tracker.calculate_cost(response['usage'], model_choice)
                    log_usage(st.session_state.username, model_choice, response['usage'], cost)
                ai_response = response.get('content', "Si è verificato un errore nel tutor AI.")
                st.session_state.tutor_chat_history.append({"role": "assistant", "content": ai_response})
            st.rerun()
    st.markdown("---")



def display_token_usage(st):
    st.sidebar.subheader("📊 Utilizzo dei Token e Costo")
    token_summary = st.session_state.token_tracker.get_summary()
    st.sidebar.write(f"**Total Tokens:** {token_summary['total_tokens']:,}")
    st.sidebar.write(f"**Input Tokens:** {token_summary['input_tokens']:,}")
    st.sidebar.write(f"**Output Tokens:** {token_summary['output_tokens']:,}")
    st.sidebar.write(f"**Total Cost:** ${token_summary['cost']:.4f}")

    st.sidebar.subheader("💻 Utilizzo Specifico del Modello")
    for model, usage in token_summary['model_usage'].items():
        with st.sidebar.expander(f"{model} Usage"):
            st.write(f"**Total Tokens:** {usage['total_tokens']:,}")
            st.write(f"**Input Tokens:** {usage['input_tokens']:,}")
            st.write(f"**Output Tokens:** {usage['output_tokens']:,}")
            st.write(f"**Cost:** ${usage['cost']:.4f}")

def display_suggested_topics(st):
    st.subheader("📚 Argomenti Suggeriti")
    suggested_topics = ["Equazioni di secondo grado", "Teorema di Pitagora", "Funzioni trigonometriche"]
    cols = st.columns(3)
    for i, topic in enumerate(suggested_topics):
        with cols[i]:
            if st.button(topic, key=f"topic_{i}"):
                st.session_state.current_input = f"Puoi spiegarmi {topic}?"
                st.rerun()

def display_feature_feedback(st, feature, content, model_choice):
    if f'feedback_{feature}' not in st.session_state:
        st.session_state[f'feedback_{feature}'] = {'rating': None, 'comment': ''}
    cols = st.columns([1, 1, 4])
    with cols[0]:
        if st.button("👍", key=f"thumbs_up_{feature}"):
            st.session_state[f'feedback_{feature}']['rating'] = 1
    with cols[1]:
        if st.button("👎", key=f"thumbs_down_{feature}"):
            st.session_state[f'feedback_{feature}']['rating'] = 0

    st.session_state[f'feedback_{feature}']['comment'] = st.text_area("Lascia un commento (opzionale)", key=f"comment_{feature}")

    if st.button("Invia Feedback", key=f"submit_feedback_{feature}"):
        if st.session_state[f'feedback_{feature}']['rating'] is not None:
            log_feature_feedback(st, feature, content, model_choice, 
                                 st.session_state[f'feedback_{feature}']['rating'], 
                                 st.session_state[f'feedback_{feature}']['comment'])
            st.success("Grazie per il tuo feedback!")
            st.session_state[f'feedback_{feature}'] = {'rating': None, 'comment': ''}
        else:
            st.warning("Per favore, seleziona un rating (👍 o 👎) prima di inviare il feedback.")

def log_feature_feedback(st, feature, content, model_choice, rating, comments=''):
    log_feedback(
        st.session_state.username,
        feature,
        '',
        '',
        content,
        rating,
        comments,
        model_choice,
        ''
    )

def display_chat_feedback(st, idx, message, model_choice):
    if f'chat_feedback_{idx}' not in st.session_state:
        st.session_state[f'chat_feedback_{idx}'] = {'rating': None, 'comment': ''}
    cols = st.columns([1, 1, 4])
    with cols[0]:
        if st.button("👍", key=f"thumbs_up_ai_tutor_{idx}"):
            st.session_state[f'chat_feedback_{idx}']['rating'] = 1
    with cols[1]:
        if st.button("👎", key=f"thumbs_down_ai_tutor_{idx}"):
            st.session_state[f'chat_feedback_{idx}']['rating'] = 0

    st.session_state[f'chat_feedback_{idx}']['comment'] = st.text_input("Lascia un commento (opzionale)", key=f"comment_ai_tutor_{idx}")

    if st.button("Invia Feedback", key=f"send_comment_ai_tutor_{idx}"):
        if st.session_state[f'chat_feedback_{idx}']['rating'] is not None:
            log_chat_feedback(st, idx, message, model_choice, 
                              st.session_state[f'chat_feedback_{idx}']['rating'], 
                              st.session_state[f'chat_feedback_{idx}']['comment'])
            st.success("Grazie per il tuo feedback!")
            st.session_state[f'chat_feedback_{idx}'] = {'rating': None, 'comment': ''}
        else:
            st.warning("Per favore, seleziona un rating (👍 o 👎) prima di inviare il feedback.")

def log_chat_feedback(st, idx, message, model_choice, rating, comments=''):
    log_feedback(
        st.session_state.username,
        'ai_tutor',
        idx,
        st.session_state.tutor_chat_history[idx - 1]['content'] if idx > 0 else '',
        message['content'],
        rating,
        comments,
        model_choice,
        json.dumps(st.session_state.tutor_chat_history)
    )
    st.session_state.tutor_chat_history[idx]['feedback_given'] = True
    st.success("Grazie per il tuo feedback!" if rating != -1 else "Grazie per il tuo commento!")
    st.rerun()