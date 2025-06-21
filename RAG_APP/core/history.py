from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import json
from uuid import uuid4
from core.config import Config
config = Config()

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    created_at: str


class ConversationHistory:
    def __init__(self, file_path: Path = config.CHAT_HISTORY_PATH):
        self._file_path = file_path
        self._messages: List[Message] = []
        self._conversation_id: Optional[str] = None

    def create_new_chat(self) -> str:
        """Generates a new conversation file with UUID"""
        conversation_id = f"conv_{uuid4().hex[:8]}"
        self._conversation_id = conversation_id
        timestamp = datetime.now().isoformat()
        self._get_filepath(conversation_id)
        self._file_path.mkdir(parents=True, exist_ok=True)
        conversation_data = {
            "conversation_id": conversation_id,
            "created_at": timestamp,
            "messages": []
        }
        with open(self._get_filepath(conversation_id), 'w') as f:
            json.dump(conversation_data, f, indent=2)

        return conversation_id
    

    def load_messages(self, conversation_id: str) -> List[Message]:
        """Load messages for a single conversation"""
        path = self._get_filepath(conversation_id)
        with open(path, 'r') as f:
            data = json.load(f)
            return [Message(**msg) for msg in data.get("messages", [])]
        
    def add_message(self, role: str, content: str, conversation_id: str):
        """
        Add a message to the conversation history.
        """
        target_id = conversation_id or self._current_conversation_id
        if not target_id:
            raise ValueError("No active conversation")
        
        # 1. Load existing messages
        messages = self.load_messages(target_id)
        
        # 2. Append new message
        messages.append(Message(
            role=role,
            content=content,
            created_at=datetime.now().isoformat()
        ))
        
        # 3. Save back to file
        self._save(target_id, messages)

    def get_conversation_id(self) -> Optional[str]:
        """
        Get the current conversation ID.
        """
        return self._conversation_id

    def last_n_messages(self, n: int, conversation_id: str) -> List[Message]:
        """
        Get the last n messages from the conversation to be used as context.
        """
        messages = self.load_messages(conversation_id)
        return messages[-n:] if len(messages) >= n else messages
    
    def _get_filepath(self, conversation_id: str) -> Path:
        return self.file_path / f"{conversation_id}.json"
    
        
    def _save(self, conversation_id: str, messages: List[Message]):
        """
        Save messages to a JSON file.
        """
        path = self._get_filepath(conversation_id)
        data = {
            "conversation_id": conversation_id,
            "messages": [msg.model_dump() for msg in messages]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    


    