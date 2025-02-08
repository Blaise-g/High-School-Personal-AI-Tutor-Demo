# genai/ocument_processing.py
import PyPDF2
import docx
import csv
import requests
from bs4 import BeautifulSoup
from io import BytesIO, StringIO
import trafilatura
import logging
import chardet
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_encoding(file_content: bytes) -> str:
    """
    Detects the encoding of the given file content.
    """
    result = chardet.detect(file_content)
    return result['encoding']

def process_pdf(file_content: bytes) -> str:
    """
    Extracts text from a PDF file.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return ""

def process_docx(file_content: bytes) -> str:
    """
    Extracts text from a DOCX file.
    """
    try:
        doc = docx.Document(BytesIO(file_content))
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error processing DOCX: {str(e)}")
        return ""

def process_txt(file_content: bytes) -> str:
    """
    Extracts text from a TXT file.
    """
    try:
        encoding = detect_encoding(file_content)
        return file_content.decode(encoding).strip()
    except Exception as e:
        logger.error(f"Error processing TXT: {str(e)}")
        return ""

def process_csv(file_content: bytes) -> str:
    """
    Extracts text from a CSV file.
    """
    try:
        encoding = detect_encoding(file_content)
        csv_data = file_content.decode(encoding)
        csv_reader = csv.reader(StringIO(csv_data))
        return "\n".join([", ".join(row) for row in csv_reader])
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        return ""

def process_html(file_content: bytes) -> str:
    """
    Extracts text from an HTML file.
    """
    try:
        soup = BeautifulSoup(file_content, 'html.parser')
        return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        logger.error(f"Error processing HTML: {str(e)}")
        return ""

def scrape_website(url: str) -> str:
    """
    Scrapes and extracts text content from a website.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded, include_links=False, include_images=False, include_formatting=False)
        if text:
            return text.strip()
        else:
            # Fallback to BeautifulSoup if trafilatura fails
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        logger.error(f"Error scraping website {url}: {str(e)}")
        return ""

def is_valid_url(url: str) -> bool:
    """
    Validates if the provided string is a valid URL.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def process_file(file: Any) -> Dict[str, Any]:
    """
    Processes a single uploaded file based on its MIME type.
    """
    try:
        file_content = file.read()
        file_type = file.type
        filename = file.name
        if file_type == "application/pdf":
            text = process_pdf(file_content)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = process_docx(file_content)
        elif file_type == "text/plain":
            text = process_txt(file_content)
        elif file_type == "text/csv":
            text = process_csv(file_content)
        elif file_type in ["text/html", "application/xhtml+xml"]:
            text = process_html(file_content)
        else:
            logger.warning(f"Unsupported file type: {file_type}")
            return None
        return {
            "content": text,
            "metadata": {"source": filename, "type": file_type}
        }
    except Exception as e:
        logger.error(f"Error processing file {file.name}: {str(e)}")
        return None

def process_documents(uploaded_files: List[Any], web_urls: List[str]) -> List[Dict[str, Any]]:
    """
    Processes multiple uploaded files and web URLs concurrently.
    """
    processed_docs = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Process files
        file_futures = {executor.submit(process_file, file): file for file in uploaded_files}

        # Process URLs
        url_futures = {executor.submit(scrape_website, url): url for url in web_urls if is_valid_url(url)}

        # Combine all futures
        all_futures = {**file_futures, **url_futures}

        for future in as_completed(all_futures):
            source = all_futures[future]
            try:
                result = future.result()
                if result:
                    processed_docs.append(result)
            except Exception as e:
                logger.error(f"Error processing {source}: {str(e)}")
    return processed_docs