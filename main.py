from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.leads import router as lead_router

from app.api.v1 import api_router
from app.core.exceptions.handlers import register_exception_handlers
from app.lifespan import lifespan

app = FastAPI(
    title="NEXTGENAI API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.nxzenai.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # "http://localhost:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handlers for the AI Studio modules (auth, crm, datasets, etc.)
register_exception_handlers(app)

# ---------------------------------------------------------
# Routers
# ---------------------------------------------------------

# Legacy public marketing site API (lead capture form on nxzenai.com)
app.include_router(lead_router)

# AI Studio API consumed by the dashboard/studio frontend (/api/v1/...)
app.include_router(api_router)


@app.get("/")
async def root():
    return {
        "message": "NEXTGENAI Backend Running"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }
