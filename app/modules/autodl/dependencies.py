"""
NxZen AI Studio

AutoDL Dependencies
"""

from __future__ import annotations
from fastapi import Depends
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from app.modules.autodl.models import Base

# Create an isolated SQLite DB for the AI module to bypass MongoDB
engine = create_engine("sqlite:///./autodl.db", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from app.modules.autodl.repository import AutoDLRepository
from app.modules.autodl.service import AutoDLService

def get_autodl_service(db: Session = Depends(get_db)) -> AutoDLService:
    repo = AutoDLRepository(db)
    return AutoDLService(repo=repo)