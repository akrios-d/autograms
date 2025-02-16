import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Configuration variables
DATA_DIR = os.getenv("DATA_DIR", "./data/")
CONFLUENCE_API_URL = os.getenv("CONFLUENCE_API_URL")
CONFLUENCE_API_KEY = os.getenv("CONFLUENCE_API_KEY")
CONFLUENCE_API_USER= os.getenv("CONFLUENCE_API_USER")
CONFLUENCE_PAGE_IDS = os.getenv("CONFLUENCE_PAGE_IDS", "").split(",")
MANTIS_API_URL = os.getenv("MANTIS_API_URL")
MANTIS_API_KEY = os.getenv("MANTIS_API_KEY")