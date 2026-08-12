"""
NxZen AI Studio

GenAI Dependencies
"""

from __future__ import annotations
from fastapi import Depends
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from app.modules.genai.models import Base

# Create an isolated SQLite DB for the AI module to bypass MongoDB
engine = create_engine("sqlite:///./genai.db", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from app.modules.genai.repository import GenAIRepository
from app.modules.genai.service import GenAIService

def get_genai_service(db: Session = Depends(get_db)) -> GenAIService:
    repo = GenAIRepository(db)
    return GenAIService(repo=repo)