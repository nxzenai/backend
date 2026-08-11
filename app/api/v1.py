from fastapi import APIRouter

from app.modules.auth.router import router as auth_router
from app.modules.system.api.router import router as system_router
from app.modules.notebooks.router import (
    router as notebook_router,
)
from app.modules.execution.router import router as execution_router
from app.modules.crm.router import router as crm_router
from app.modules.datasets.router import (
    router as dataset_router,
)
from app.modules.sql.router import (
    router as sql_router,
)
from app.modules.automl.router import (
    router as automl_router,
)

api_router = APIRouter(
    prefix="/api/v1"
)

api_router.include_router(system_router)
api_router.include_router(auth_router)
api_router.include_router(
    notebook_router,
)

api_router.include_router(execution_router)
api_router.include_router(
    dataset_router,
)
api_router.include_router(
    sql_router
)
api_router.include_router(
    automl_router,
)
api_router.include_router(crm_router)
