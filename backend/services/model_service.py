"""
Model Service — Arogentis
===========================
Singleton model loader: loads the trained pipeline once at API startup.

WHY SINGLETON?
  - Model loading (joblib.load + SHAP init) is expensive (~1–3 seconds)
  - FastAPI is async — multiple concurrent requests would try to load simultaneously
  - Singleton ensures one load on startup, then instant inference per request
"""

import logging
import os

import numpy as np

from models.risk_scorer import SchizophreniaRiskScorer, RiskResult
from explainability.shap_explainer import EEGShapExplainer

logger = logging.getLogger(__name__)

_scorer: SchizophreniaRiskScorer | None = None
_explainer: EEGShapExplainer | None = None
_feature_names: list[str] | None = None


class ModelService:
    """Singleton service for model inference and SHAP explanation."""

    MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/rf_model.pkl")
    FEATURE_NAMES_PATH = "data/features/feature_names.txt"

    @classmethod
    def load(cls) -> None:
        """Load model and SHAP explainer into module-level singletons."""
        global _scorer, _explainer, _feature_names

        if not os.path.exists(cls.MODEL_PATH):
            logger.warning(
                f"Model not found at {cls.MODEL_PATH}. "
                "Run training first: python -m models.baseline_model"
            )
            return

        _feature_names = cls._load_feature_names()
        _scorer = SchizophreniaRiskScorer(cls.MODEL_PATH)

        if _feature_names:
            try:
                _explainer = EEGShapExplainer(cls.MODEL_PATH, _feature_names)
                logger.info("SHAP explainer loaded successfully.")
            except Exception as e:
                logger.warning(f"SHAP explainer failed to load: {e}. Predictions still available.")

        logger.info("ModelService ready.")

    @classmethod
    def is_loaded(cls) -> bool:
        return _scorer is not None

    @classmethod
    def get_feature_names(cls) -> list[str]:
        return _feature_names or []

    @classmethod
    def predict(cls, features: np.ndarray) -> RiskResult:
        if _scorer is None:
            raise RuntimeError("Model not loaded. Call ModelService.load() first.")
        return _scorer.score(features, _feature_names)

    @classmethod
    def explain(cls, features: np.ndarray) -> dict:
        if _explainer is None:
            return {"error": "SHAP explainer not available.", "top_features": []}
        return _explainer.explain(features)

    @classmethod
    def shap_waterfall_bytes(cls, features: np.ndarray) -> bytes | None:
        if _explainer is None:
            return None
        return _explainer.plot_waterfall(features)

    @classmethod
    def _load_feature_names(cls) -> list[str] | None:
        if os.path.exists(cls.FEATURE_NAMES_PATH):
            with open(cls.FEATURE_NAMES_PATH) as f:
                return [l.strip() for l in f if l.strip()]
        logger.warning(f"Feature names file not found: {cls.FEATURE_NAMES_PATH}")
        return None
