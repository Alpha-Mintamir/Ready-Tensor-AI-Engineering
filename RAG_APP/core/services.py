from core.generation import get_rag_response
from processing.embeddings import chroma_db, get_embeddings
from processing.doc_processor import load_docs, split_docs
from config import Config, logger

config = Config()

DOCUMENTS_DIR = config.DOCUMENTS_PATH


def generate_answer(query):
    """
    Generate an answer using the RAG pipeline.
    """
    return get_rag_response(query)

def upload_file(file):
    """
    Upload a file and process it for the RAG pipeline.
    """
    if not file:
        return "No file uploaded."
    if not file.name.endswith(('.pdf', '.docx', '.txt')):
        logger.warning(f"Unsupported file type: {file.name}")
        return "Unsupported file type. Please upload a PDF, DOCX, or TXT file."

    # Save the uploaded file
    file_path = DOCUMENTS_DIR / f"{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    
    logger.info(f"File {file.name} uploaded successfully to {file_path}")
    return f"File {file.name} uploaded successfully."


def process_and_index_files(path: str):
    """
    Process and add new documents to the vector DB after upload.
    """
    loaded_docs = load_docs(str(DOCUMENTS_DIR))
    try:
        if not loaded_docs:
            return "No valid documents found."
        chunks = split_docs(loaded_docs)
        if not chunks:
            return "No valid document chunks found."
        # Initialize embeddings
        embeddings = get_embeddings()
        chroma_db.add_documents(
            documents=chunks, 
            embedding=embeddings
        )

        
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return "Error processing documents."



# def save_chat_history(session_id, history):
#     CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
#     with open(CHAT_HISTORY_DIR / f"{session_id}.json", "w", encoding="utf-8") as f:
#         json.dump(history, f, ensure_ascii=False, indent=2)


# def load_chat_history(session_id):
#     try:
#         with open(CHAT_HISTORY_DIR / f"{session_id}.json", "r", encoding="utf-8") as f:
#             return json.load(f)
#     except FileNotFoundError:
#         return []


# def list_threads():
#     if not CHAT_HISTORY_DIR.exists():
#         return []
#     return [f.stem for f in CHAT_HISTORY_DIR.glob("*.json")]