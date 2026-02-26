"""
EEG Service — Arogentis
=========================
Business logic layer: orchestrates preprocessing → feature extraction → scoring → explanation.

WHY A SEPARATE SERVICE LAYER?
  - Keeps FastAPI routers thin (HTTP concerns only)
  - Makes the core logic independently testable without HTTP overhead
  - Service can be reused directly by Streamlit dashboard (bypassing API)
"""

import logging
import os
import tempfile

import numpy as np

from pipeline.preprocessing import run_preprocessing
from pipeline.feature_extraction import extract_all_features
from backend.services.model_service import ModelService

logger = logging.getLogger(__name__)


class EEGService:
    """Runs the full EEG analysis pipeline on an uploaded file."""

    @staticmethod
    def run_full_pipeline(filepath: str) -> dict:
        """
        Full pipeline: EEG file → risk score + SHAP explanation.

        Args:
            filepath: Path to a temporary .edf or .fif EEG file.

        Returns:
            dict matching AnalysisResponse schema.

        Raises:
            ValueError: If no clean epochs remain after artifact rejection.
            RuntimeError: If model is not loaded.
        """
        logger.info(f"Starting EEG analysis pipeline for: {filepath}")

        # ── Step 1: Preprocess ────────────────────────────────────────────────
        epochs = run_preprocessing(filepath)
        n_epochs = len(epochs)

        if n_epochs == 0:
            raise ValueError(
                "No clean epochs remained after artifact rejection. "
                "The EEG recording may be too short or heavily contaminated."
            )

        channel_names = epochs.ch_names
        sfreq = epochs.info["sfreq"]

        # ── Step 2: Feature Extraction ────────────────────────────────────────
        X, feature_names = extract_all_features(epochs, aggregate=True)
        features_1d = X[0]  # shape: (n_features,)

        # ── Step 3: Risk Scoring ──────────────────────────────────────────────
        result = ModelService.predict(features_1d)

        # ── Step 4: SHAP Explanation ──────────────────────────────────────────
        shap_result = ModelService.explain(features_1d)

        logger.info(
            f"Analysis complete: risk={result.risk_probability:.3f} "
            f"tier={result.risk_tier} | epochs={n_epochs} "
            f"channels={len(channel_names)}"
        )

        return {
            "risk_probability":   result.risk_probability,
            "risk_tier":          result.risk_tier,
            "tier_color":         result.tier_color,
            "interpretation":     result.interpretation,
            "top_features":       shap_result.get("top_features", []),
            "band_powers_summary": result.biomarker_summary,
            "n_epochs_analyzed":  n_epochs,
            "n_channels":         len(channel_names),
            "sampling_rate":      float(sfreq),
            "channel_names":      channel_names,
        }
