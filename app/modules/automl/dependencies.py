"""
NxZen AI Studio

AutoML Dependencies

Dependency injection for the AutoML module.
"""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database import get_db

from app.modules.automl.repository import (
    AutoMLRepository,
)

from app.modules.automl.service import (
    AutoMLService,
)


##########################################################
# Repository Dependency
##########################################################

def get_automl_repository(
    db: Session = Depends(get_db),
) -> AutoMLRepository:
    """
    Returns an AutoML repository instance.
    """

    return AutoMLRepository(db)


##########################################################
# Service Dependency
##########################################################

def get_automl_service(
    repository: AutoMLRepository = Depends(
        get_automl_repository,
    ),
) -> AutoMLService:
    """
    Returns an AutoML service instance.
    """

    return AutoMLService(
        repository=repository,
    )