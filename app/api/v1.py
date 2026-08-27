from fastapi import APIRouter

from app.modules.auth.router import router as auth_router
from app.modules.system.api.router import router as system_router
from app.modules.notebooks.router import (
    router as notebook_router,
)
from app.modules.execution.router import router as execution_router
from app.modules.crm.router import router as crm_router
from app.modules.eda.router import router as eda_router
from app.modules.sql.router import (
    router as sql_router,
)
from app.modules.automl.router import (
    router as automl_router,
)

# --- NEW: Import our AI Module Routers ---
from app.modules.autonlp.router import router as autonlp_router
from app.modules.autodl_v2.router import router as autodl_v2_router
from app.modules.genai.router import router as genai_router
from app.modules.ai_registry.router import router as ai_registry_router

api_router = APIRouter(
    prefix="/api/v1"
)

api_router.include_router(system_router)
api_router.include_router(auth_router)
api_router.include_router(
    notebook_router,
)

api_router.include_router(execution_router)
api_router.include_router(eda_router)
api_router.include_router(
    sql_router
)
api_router.include_router(
    automl_router,
)
api_router.include_router(crm_router)

# --- NEW: Register our AI Module Routers ---
api_router.include_router(autonlp_router)
api_router.include_router(autodl_v2_router, prefix="/autodl")
api_router.include_router(
    autodl_v2_router, prefix="/autodl-v2", include_in_schema=False,
)
api_router.include_router(ai_registry_router)
api_router.include_router(genai_router)
