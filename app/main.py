from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config.settings import settings
from app.core.exceptions.handlers import register_exception_handlers
from app.core.logging.logger import logger
from app.lifespan import lifespan

from app.api.v1 import api_router
from app.modules.system.api.router import router as system_router

from app.shared.responses import APIResponse


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)


# ---------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------

origins = [
    "http://localhost:3001",
    "http://127.0.0.1:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Global Exception Handlers
# ---------------------------------------------------------

register_exception_handlers(app)


# ---------------------------------------------------------
# API Routers
# ---------------------------------------------------------

app.include_router(api_router)
app.include_router(system_router)


logger.info("NxZenAI Studio Backend Initialised")


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------

@app.get("/", response_model=APIResponse)
async def root():
    logger.info("Root endpoint accessed")

    return APIResponse(
        success=True,
        message="NxZenAI Studio Backend is running.",
        data={
            "application": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "status": "running",
        },
    )


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------

@app.get("/health", response_model=APIResponse)
async def health():
    return APIResponse(
        success=True,
        message="Backend Healthy",
        data={
            "status": "healthy",
            "application": settings.app_name,
            "version": settings.app_version,
        },
    )