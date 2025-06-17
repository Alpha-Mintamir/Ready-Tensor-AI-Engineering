# RAG_APP/core/config.py

from pathlib import Path
from dotenv import load_dotenv
import os
import logging
from logging.handlers import RotatingFileHandler

# Load .env file from root
# env_path = Path(__file__).parent.parent / ".env"
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log",maxBytes=1_000_000, backupCount=5),
        logging.StreamHandler()
    ]
)
# Initialize logger
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for the RAG application."""
    # ENV Variables
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY is not set in the environment variables.")
        raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DOCUMENTS_PATH = BASE_DIR / "documents"
    

    # Vectorstore
    CHROMA_COLLECTION_NAME = "rag_app"
    CHROMA_PATH = BASE_DIR / "chroma_db"

