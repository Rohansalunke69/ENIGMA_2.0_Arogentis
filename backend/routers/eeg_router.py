"""
EEG Router — Arogentis FastAPI
"""

import os
import shutil
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.schemas import AnalysisResponse, HealthResponse
from backend.services.eeg_service import EEGService
from backend.services.model_service import ModelService

router = APIRouter(tags=["EEG Analysis"])

SUPPORTED_FORMATS = (".edf", ".fif")


@router.get("/health", response_model=HealthResponse)
def health_check():
    """API health check — also reports model load status."""
    return HealthResponse(
        status="ok",
        service="arogentis-eeg",
        version="1.0.0",
        model_loaded=ModelService.is_loaded(),
    )


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_eeg(file: UploadFile = File(...)):
    """
    Upload an EEG file (.edf or .fif) and receive a schizophrenia risk score.

    The full pipeline runs server-side:
      1. Preprocessing (filter + epoch + artifact rejection)
      2. Feature extraction (band power, entropy, ratios)
      3. Risk scoring (probability [0–1] with tier)
      4. SHAP explanation (top biomarker attributions)
    """
    filename = file.filename or ""
    if not any(filename.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{filename}'. Only {SUPPORTED_FORMATS} are supported.",
        )

    if not ModelService.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first: python train.py",
        )

    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = EEGService.run_full_pipeline(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        os.unlink(tmp_path)

    return result
