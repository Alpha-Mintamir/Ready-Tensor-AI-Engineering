from core.generation import get_rag_response
from processing.embeddings import chroma_db, get_embeddings
from processing.doc_processor import load_docs, split_docs
from history import ConversationHistory
from config import Config, logger
from typing import Optional
import os

config = Config()
history = ConversationHistory()

DOCUMENTS_DIR = config.DOCUMENTS_PATH
CHROMA_PATH = config.CHROMA_PATH


def generate_answer(query, history: Optional[str] = ""):
    """
    Generate an answer using the RAG pipeline.
    """
    return get_rag_response(query, history=history)

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
    Returns:
        str: Success or error message.
    """
    loaded_docs = load_docs(path)
    try:
        if not loaded_docs:
            logger.warning("No valid documents found.")
            return "No valid documents found."
        chunks = split_docs(loaded_docs)
        if not chunks:
            logger.warning("No valid document chunks found after splitting.")
            return "No valid document chunks found."
        if not os.path.exists(CHROMA_PATH):
            logger.error("Chroma DB not found. Initialize it first.")
            return "Chroma DB not found. Initialize it first."

        chroma_db.add_documents(documents=chunks)
        chroma_db.persist()
        logger.info(f"Processed and indexed {len(chunks)} document chunks.")
        return f"Processed and indexed file: {path} document chunks."
        
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return "Error processing documents."
    
def load_chat_history(conversation_id: str):
    """
    Load chat history for a given coversation ID.
    """
    history.load_messages(conversation_id=conversation_id)
    if not history:
        logger.info(f"No chat history found for session {conversation_id}.")
        return []
    logger.info(f"Loaded chat history for session {conversation_id}.")
    return history

def add_message(role: str, content: str, conversation_id):
    message = history.add_message(
        role=role,
        content=content,
        conversation_id=conversation_id
    )
    if message:
        logger.info(f"Added message to conversation {conversation_id}: {content}")
    else:
        logger.error(f"Failed to add message to conversation {conversation_id}.")
    
    return {'details': 'Message added successfully.'}

def create_new_chat():
    """
    Create a new chat session and return the conversation ID.
    """
    history.create_new_chat()
    logger.info(f"Created new chat session with ID: {history._conversation_id}")
    return {'conversation_id': history._conversation_id}

def get_conversation_id():
    """
    Get the current conversation ID.
    """
    conversation_id = history.get_conversation_id()
    if not conversation_id:
        logger.error("No active conversation found.")
        return {'error': 'No active conversation found.'}
    
    logger.info(f"Current conversation ID: {conversation_id}")
    return {'conversation_id': conversation_id}

def get_last_n_messages(n: int, conversation_id: str):
    """
    Get the last n messages from the conversation.
    """
    messages = history.last_n_messages(n=n, conversation_id=conversation_id)
    if not messages:
        logger.warning(f"No messages found for conversation {conversation_id}.")
        return {'messages': []}
    
    logger.info(f"Retrieved last {n} messages for conversation {conversation_id}.")
    return [msg.model_dump() for msg in messages]

def format_chat_history(chat_history: list):
    """
    Format chat history for display.
    """
    formatted_history = []
    for msg in chat_history:
        if msg['role'] == 'user':
            formatted_history.append(f"User: {msg['content']}")
        else:
            formatted_history.append(f"Bot: {msg['content']}")
    
    return formatted_history