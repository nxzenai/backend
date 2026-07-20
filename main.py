from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.leads import router as lead_router

app = FastAPI(
    title="NEXTGENAI API",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # "http://localhost:3001"
        "www.nxzenai.com"
        
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(lead_router)

@app.get("/")
async def root():
    return {
        "message": "NEXTGENAI Backend Running"
    }