import os
import glob
import json
import logging
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, UnstructuredHTMLLoader, UnstructuredPDFLoader
import requests
from common.config import CONFLUENCE_API_URL, CONFLUENCE_API_KEY, CONFLUENCE_API_USER, MANTIS_API_URL, MANTIS_API_KEY

logger = logging.getLogger(__name__)

def load_documents(from_confluence=False, from_mantis=False):
    documents = []

    documents.extend(load_local_files(ext_list=['*.txt']))

    if from_confluence:
        documents.extend(fetch_confluence_pages())
    if from_mantis:
        documents.extend(fetch_mantis_issues())

    return documents

def fetch_confluence_pages():
    """Fetches the content of a Confluence page using the REST API."""
    documents = []
    CONFLUENCE_AUTH = (CONFLUENCE_API_USER, CONFLUENCE_API_KEY)  # Replace with actual Confluence username and API token

    for page_id in os.getenv("CONFLUENCE_PAGE_IDS", "").split(","):
        url = f"{CONFLUENCE_API_URL}/{page_id}?expand=body.storage"
        response = requests.get(url, auth=CONFLUENCE_AUTH )
        if response.status_code == 200:
            content = response.json()["body"]["storage"]["value"]
            documents.append(Document(page_content=content, metadata={"source": f"Confluence - {page_id}"}))
        else:
            logger.error(f"Error fetching Confluence page {page_id}. Status: {response.status_code}")
    return documents

def fetch_all_confluence_pages():
    """Fetches the content of all Confluence pages using the REST API."""
    documents = []
    CONFLUENCE_AUTH = (CONFLUENCE_API_USER, CONFLUENCE_API_KEY)  # Replace with actual Confluence username and API token

    # Confluence API endpoint for fetching pages
    url = f"{CONFLUENCE_API_URL}/rest/api/content"
    
    # Pagination variables
    start = 0
    limit = 25  # Adjust as needed, up to 100

    while True:
        params = {
            'start': start,
            'limit': limit,
            'expand': 'body.storage',  # Expand the body content
        }

        # Send the request to get pages
        response = requests.get(url, auth=CONFLUENCE_AUTH, params=params)
        if response.status_code == 200:
            data = response.json()
            for page in data['results']:
                content = page["body"]["storage"]["value"]
                page_id = page["id"]
                documents.append(Document(page_content=content, metadata={"source": f"Confluence - {page_id}"}))

            # Check if there are more pages
            if 'next' in data['_links']:
                start += limit  # Move to the next page
            else:
                break  # No more pages to fetch
        else:
            logger.error(f"Error fetching Confluence pages. Status: {response.status_code}")
            break

    return documents

def fetch_mantis_issues():
    documents = []
    headers = {"Authorization": f"Bearer {MANTIS_API_KEY}"}
    response = requests.get(f"{MANTIS_API_URL}/issues", headers=headers)
    if response.status_code == 200:
        issues = response.json()
        for issue in issues:
            content = f"{issue['summary']}\n{issue['description']}"
            documents.append(Document(page_content=content, metadata={"source": "Mantis"}))
    else:
        logger.error(f"Error fetching Mantis issues. Status: {response.status_code}")
    return documents

# --------------------
# Helper functions for reading docs
# --------------------
def load_local_files(docs_folder="docs",ext_list=['*.pdf', '*.txt', '*.html']):
    documents = []
    # Load local files
    for root, _, files in os.walk(docs_folder):
        for file_name in files:
            path = os.path.join(root, file_name)
            try:
                if file_name.endswith('.pdf'):
                    loader = UnstructuredPDFLoader(path)
                elif file_name.endswith('.txt'):
                    print(file_name)
                    loader = TextLoader(path, encoding = 'UTF-8')
                elif file_name.endswith('.html'):
                    loader = UnstructuredHTMLLoader(path)
                else:
                    continue
                
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = loader.load()
                    documents.append({"text":content,"metadata":{"file_name":path}})
            except Exception as e:
                logger.error(f"Error loading file {path}: {e}")
           
    return documents