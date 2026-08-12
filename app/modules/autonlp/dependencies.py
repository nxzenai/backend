"""
NxZen AI Studio

AutoNLP Dependencies
"""

from __future__ import annotations
from fastapi import Depends
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from app.modules.autonlp.models import Base

# Create an isolated SQLite DB for the AI module to bypass MongoDB
engine = create_engine("sqlite:///./autonlp.db", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from app.modules.autonlp.repository import AutoNLPRepository
from app.modules.autonlp.service import AutoNLPService

def get_autonlp_service(db: Session = Depends(get_db)) -> AutoNLPService:
    repo = AutoNLPRepository(db)
    return AutoNLPService(repo=repo)