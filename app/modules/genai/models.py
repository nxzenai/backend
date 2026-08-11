"""
NxZen AI Studio

GenAI Models
"""

from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import Column, String, JSON, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class ChatSession(Base):
    __tablename__ = "genai_chat_sessions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatLog(Base):
    __tablename__ = "genai_chat_logs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String)
    request_payload = Column(JSON)
    response_payload = Column(JSON)