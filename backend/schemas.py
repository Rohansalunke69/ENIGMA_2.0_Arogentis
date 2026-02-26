"""
Pydantic Schemas — Arogentis FastAPI
"""

from pydantic import BaseModel, Field
from typing import List


class TopFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_risk" | "decreases_risk"


class AnalysisResponse(BaseModel):
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Schizophrenia risk score [0–1]")
    risk_tier: str           = Field(..., description="Risk tier: Low / Moderate / High / Critical")
    tier_color: str          = Field(..., description="Hex color for UI rendering")
    interpretation: str      = Field(..., description="Clinical interpretation text")
    top_features: List[TopFeature]
    band_powers_summary: dict = Field(default_factory=dict, description="Mean band powers across channels")
    n_epochs_analyzed: int
    n_channels: int
    sampling_rate: float
    channel_names: List[str]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
