"""
FastAPI Application — Arogentis
=================================
Main entrypoint for the EEG schizophrenia screening REST API.

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Swagger UI: http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.eeg_router import router as eeg_router
from backend.services.model_service import ModelService

app = FastAPI(
    title="Arogentis — EEG Schizophrenia Screening API",
    description=(
        "AI-powered EEG analysis for early schizophrenia risk detection. "
        "Uses frequency-domain biomarkers, machine learning, and SHAP explainability. "
        "**For research use only. Not a medical diagnostic tool.**"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Allow all origins for hackathon. Lock down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup Event ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Load ML model and SHAP explainer into memory on startup (warm-up)."""
    ModelService.load()

# ─── Routers ───────────────────────────────────────────────────────────────────
app.include_router(eeg_router, prefix="/api/v1")

# ─── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "Arogentis EEG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
