"""
Schizophrenia Risk Scorer — Arogentis
========================================
Wraps any trained sklearn-compatible model to produce a clinically interpretable
probability-based risk score with tier classification.

Design principle:
  Output is not "schizophrenia = yes/no" but rather a continuous risk score [0, 1].
  This is consistent with how clinical tools are used — psychiatrists need a spectrum,
  not a binary label, to guide differential diagnosis decisions.
"""

import logging
from dataclasses import dataclass

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ─── Risk Tier Definitions ─────────────────────────────────────────────────────
# Boundaries chosen to mirror clinical triage thresholds.
# Low: within 1 SD of normal population distribution
# Moderate: borderline — needs follow-up but not acute intervention
# High: significant deviation — refer to psychiatrist
# Critical: severe — immediate clinical assessment required
RISK_TIERS = [
    (0.00, 0.30, "Low Risk",      "#2ecc71"),  # green
    (0.30, 0.55, "Moderate Risk", "#f39c12"),  # amber
    (0.55, 0.75, "High Risk",     "#e74c3c"),  # red
    (0.75, 1.01, "Critical Risk", "#8e0000"),  # dark red
]

CLINICAL_INTERPRETATIONS = {
    "Low Risk": (
        "EEG biomarkers are within normal limits. "
        "Gamma coherence, alpha power, and spectral entropy show no significant deviations. "
        "No schizophrenia-spectrum signal detected."
    ),
    "Moderate Risk": (
        "Mild abnormalities detected in frequency-domain biomarkers. "
        "Some deviation in alpha/theta ratio or spectral entropy observed. "
        "Recommend clinical follow-up and repeat EEG assessment in 3 months."
    ),
    "High Risk": (
        "Significant biomarker deviations consistent with psychosis-spectrum patterns. "
        "Reduced gamma power and elevated spectral entropy detected. "
        "Clinical consultation with a psychiatrist is strongly advised."
    ),
    "Critical Risk": (
        "Severe EEG abnormalities highly consistent with schizophrenia-spectrum disorder. "
        "Multiple biomarkers (gamma deficit, alpha suppression, elevated entropy) flagged. "
        "Immediate psychiatric evaluation is recommended."
    ),
}


@dataclass
class RiskResult:
    risk_probability: float
    risk_tier: str
    tier_color: str
    interpretation: str
    biomarker_summary: dict


class SchizophreniaRiskScorer:
    """Wraps a trained sklearn pipeline to produce risk scores."""

    def __init__(self, model_path: str):
        self.pipeline = joblib.load(model_path)
        logger.info(f"Risk scorer loaded model from: {model_path}")

    def score(self, features: np.ndarray, feature_names: list[str] = None) -> RiskResult:
        """
        Compute schizophrenia risk score from a feature vector.

        Args:
            features:      1D array of shape (n_features,) or 2D (1, n_features).
            feature_names: Optional list for biomarker summary extraction.

        Returns:
            RiskResult dataclass.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        proba = self.pipeline.predict_proba(features)[0]
        prob  = float(proba[1])   # class 1 = schizophrenia probability

        tier, color = self._get_tier(prob)
        biomarker_summary = self._extract_biomarker_summary(features[0], feature_names)

        return RiskResult(
            risk_probability=round(prob, 4),
            risk_tier=tier,
            tier_color=color,
            interpretation=CLINICAL_INTERPRETATIONS[tier],
            biomarker_summary=biomarker_summary,
        )

    def _get_tier(self, prob: float) -> tuple[str, str]:
        for lo, hi, tier, color in RISK_TIERS:
            if lo <= prob < hi:
                return tier, color
        return "Critical Risk", "#8e0000"

    def _extract_biomarker_summary(
        self,
        features: np.ndarray,
        feature_names: list[str] | None,
    ) -> dict:
        """
        Extract mean values of key biomarkers across channels for dashboard display.
        """
        if feature_names is None:
            return {}

        summary = {}
        biomarkers = [
            "abs_delta", "abs_theta", "abs_alpha", "abs_beta", "abs_gamma",
            "spectral_entropy", "alpha_theta_ratio", "gamma_theta_ratio",
        ]
        for bio in biomarkers:
            indices = [i for i, name in enumerate(feature_names) if name.startswith(bio)]
            if indices:
                summary[bio] = float(features[indices].mean())

        return summary
