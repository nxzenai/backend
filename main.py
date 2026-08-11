from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import api_router
from app.core.config.settings import settings
from app.core.exceptions.handlers import register_exception_handlers
from app.lifespan import lifespan
from routers.leads import router as lead_router

app = FastAPI(
    title="NxZenAI API",
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

# Both frontends are served by the merged deployment. Keep the local development
# ports as well as the production domain enabled for compatibility.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        #"http://127.0.0.1:3000",
        #"http://127.0.0.1:3001",
        "https://www.nxzenai.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

# Preserve the marketing API and mount the application router under /api/v1.
app.include_router(lead_router)
app.include_router(api_router)


@app.get("/")
async def root():
    return {"message": "NxZenAI Backend Running"}

