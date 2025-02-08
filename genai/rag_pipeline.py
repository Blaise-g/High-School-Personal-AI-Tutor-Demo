
# rag_pipeline.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
import os
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.db_connection import get_conn, close_conn
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def summarize_long_document(document: str, max_tokens: int = 5000) -> str:
    """
    Summarizes a long document using OpenAI's GPT model.
    """
    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        doc = Document(page_content=document)
        summary = chain.invoke({"input_documents": [doc]})
        return summary['output_text']
    except Exception as e:
        logger.error(f"Error summarizing document: {str(e)}")
        return document[:max_tokens]  # Fallback to truncation

def process_document(doc: Dict[str, Any], max_tokens: int = 5000) -> Dict[str, Any]:
    """
    Processes a single document, summarizing if it's too long.
    """
    try:
        if len(doc['content']) > max_tokens:
            doc['content'] = summarize_long_document(doc['content'], max_tokens)
        return doc
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return None

def initialize_rag(processed_docs: List[Dict[str, Any]], st, token_tracker) -> Any:
    """
    Initializes the RAG pipeline by creating a vector store from processed documents.
    """
    try:
        logger.info(f"Initializing RAG pipeline with {len(processed_docs)} documents")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = []
        metadatas = []
        chunk_embeddings = []
        
        embedding_model = "text-embedding-3-small"
        embeddings = OpenAIEmbeddings(model=embedding_model)

        # Process each document
        for doc in processed_docs:
            try:
                text = doc["content"]
                metadata = doc["metadata"].copy()
                
                # Split text into chunks
                chunks = text_splitter.split_text(text)
                logger.info(f"Created {len(chunks)} chunks for document: {metadata['source']}")
                
                # Process each chunk
                for idx, chunk in enumerate(chunks):
                    try:
                        # Generate embedding for chunk
                        vector = embeddings.embed_query(chunk)
                        
                        # Store chunk and metadata
                        documents.append(chunk)
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_id"] = idx
                        metadatas.append(chunk_metadata)
                        
                        # Store embedding information
                        chunk_embeddings.append({
                            "chunk_id": idx,
                            "text": chunk,
                            "vector": vector
                        })
                        
                    except Exception as chunk_error:
                        logger.error(f"Error processing chunk {idx}: {str(chunk_error)}")
                        continue
                
                # Add embeddings to document for storage
                doc["embeddings"] = chunk_embeddings
                
            except Exception as doc_error:
                logger.error(f"Error processing document {doc.get('metadata', {}).get('source', 'unknown')}: {str(doc_error)}")
                continue

        if not documents:
            raise ValueError("No documents were successfully processed")

        # Create FAISS index
        vectorstore = FAISS.from_texts(
            texts=documents,
            embedding=embeddings,
            metadatas=metadatas
        )

        # Log embedding usage
        total_tokens = sum(len(doc.split()) for doc in documents)
        token_tracker.update({"prompt_tokens": total_tokens, "completion_tokens": 0}, embedding_model)

        logger.info(f"Successfully created vector store with {len(documents)} chunks")
        return vectorstore

    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {str(e)}", exc_info=True)
        if st:
            st.error(f"Error initializing RAG pipeline: {str(e)}")
        return None

def get_rag_context(query: str, topic_id: int = None) -> List[Any]:
    """Get context using stored embeddings"""
    try:
        conn = get_conn()
        cur = conn.cursor()

        # Get stored embeddings for topic
        cur.execute("""
            SELECT f.embeddings, f.embedding_model
            FROM files f
            JOIN topics t ON f.topic_id = t.topic_id
            WHERE t.topic_id = %s
        """, (topic_id,))

        results = cur.fetchall()
        if not results:
            return []

        # Create temporary FAISS index from stored embeddings
        embedding_model = "text-embedding-3-small"
        embeddings = OpenAIEmbeddings(model=embedding_model)

        documents = []
        vectors = []
        for stored_embeddings, _ in results:
            for chunk in stored_embeddings:
                documents.append(chunk["text"])
                vectors.append(chunk["vector"])

        vectorstore = FAISS.from_embeddings(vectors, embeddings, documents)

        # Perform similarity search
        query_embedding = embeddings.embed_query(query)
        similar_docs = vectorstore.similarity_search_by_vector(query_embedding, k=5)

        return similar_docs
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return []

def retrieve_relevant_docs(vectorstore: Any, query: str, k: int = 5) -> List[Any]:
    """
    Retrieves the top-k most relevant documents for a given query.
    """
    try:
        if not vectorstore:
            logger.warning("Vector store is not initialized")
            return []

        logger.info(f"Retrieving {k} relevant documents for query: {query}")
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)

        # Log retrieved documents for debugging
        logger.debug(f"Retrieved {len(docs_and_scores)} documents")
        for doc, score in docs_and_scores:
            logger.debug(f"Document score: {score}")

        return [doc for doc, score in docs_and_scores]

    except Exception as e:
        logger.error(f"Error retrieving relevant documents: {str(e)}")
        return []

def summarize_conversation(conversation_history: List[str], max_tokens: int = 2000) -> str:
    """
    Summarizes the conversation history.
    """
    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        summary_prompt = f"Summarize the following conversation:\n\n{' '.join(conversation_history)}"
        summary = llm.predict(summary_prompt)
        return summary
    except Exception as e:
        logger.error(f"Error summarizing conversation: {e}")
        return ' '.join(conversation_history[-5:])  # Fallback to last 5 messages if summarization fails

def process_documents_parallel(docs: List[Dict[str, Any]], max_tokens: int = 5000) -> List[Dict[str, Any]]:
    """
    Processes multiple documents in parallel.
    """
    try:
        with ThreadPoolExecutor() as executor:
            future_to_doc = {executor.submit(process_document, doc, max_tokens): doc for doc in docs}
            processed_docs = []

            for future in as_completed(future_to_doc):
                try:
                    result = future.result()
                    if result:
                        processed_docs.append(result)
                except Exception as e:
                    logger.error(f"Error in document processing: {str(e)}")
                    continue

        return processed_docs
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}")
        return []