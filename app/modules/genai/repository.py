"""
NxZen AI Studio

GenAI Repository
"""

from __future__ import annotations
import uuid
from app.modules.genai.models import ChatSession, ChatLog

class GenAIRepository:
    def __init__(self, db):
        self.db = db

    def create_session(self) -> ChatSession:
        session = ChatSession(id=str(uuid.uuid4()))
        self.db.add(session)
        self.db.commit()
        return session

    def save_message(self, session_id: str, messages, response):
        log = ChatLog(
            id=str(uuid.uuid4()),
            session_id=session_id,
            request_payload=[m.dict() for m in messages],
            response_payload=response
        )
        self.db.add(log)
        self.db.commit()