from pathlib import Path
import sys
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from processing.doc_processor import load_docs, split_docs
from tenacity import retry, stop_after_attempt, wait_exponential
import os
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")    

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_embeddings():
    """
    Initializes and returns GoogleGenerativeAIEmbeddings for text embedding.
    Uses exponential backoff for retries in case of failure.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document"
    )

embeddings = get_embeddings()

DOCUMENTS = load_docs(folder_path = Path(__file__).parent.parent / "documents")
CHUNKS = split_docs(DOCUMENTS)

def initialize_chroma_db():
    if os.path.exists(CHROMA_PATH):
        chroma_db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name="rag_app"
        )
    else: 
        chroma_db = Chroma.from_documents(
            documents=CHUNKS, 
            embedding=embeddings, 
            persist_directory=CHROMA_PATH, 
            collection_name="rag_app"
            )
    return chroma_db

chroma_db = initialize_chroma_db()

def query_db(query: str):
    """
    Queries the Chroma database with the provided query string.
    Returns the results from the database.
    """
    print(f"Querying database with: {query}")
    try:
        retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
        results = retriever.invoke(input=query)
        return results
    except Exception as e:
        print(f"Error querying database: {e}")
        return []
