"""
SHAP Explainability — Arogentis
==================================
Uses SHAP TreeExplainer to attribute model predictions to individual EEG biomarkers.

WHY SHAP?
  - SHAP (SHapley Additive exPlanations) has solid mathematical foundations (game theory)
  - TreeExplainer is exact for tree-based models (RF, XGBoost) — no approximation error
  - Clinicians can directly see: "gamma_CH4 pushed risk +0.23 → elevated gamma deficit"
  - Unlike feature importance (global), SHAP gives per-sample explanation (local)
  - Essential for clinical interpretability and regulatory compliance (EU AI Act context)
"""

import io
import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

logger = logging.getLogger(__name__)


class EEGShapExplainer:
    """SHAP explainability wrapper for tree-based schizophrenia risk models."""

    def __init__(self, model_path: str, feature_names: list[str]):
        """
        Args:
            model_path:    Path to saved sklearn Pipeline (.pkl).
            feature_names: Ordered list matching feature columns.
        """
        self.pipeline     = joblib.load(model_path)
        self.clf          = self.pipeline.named_steps["clf"]
        self.scaler       = self.pipeline.named_steps["scaler"]
        self.feature_names = feature_names
        self.explainer    = shap.TreeExplainer(self.clf)
        logger.info(f"SHAP explainer initialised with {len(feature_names)} features.")

    def explain(self, features: np.ndarray) -> dict:
        """
        Compute SHAP values for a single subject feature vector.

        Args:
            features: 1D array of shape (n_features,).

        Returns:
            dict with shap_values, top 10 features, and base value.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        X_scaled = self.scaler.transform(features)
        shap_values = self.explainer.shap_values(X_scaled)

        # For binary classification RF: shap_values is list [class0, class1]
        # For XGBoost: shap_values is 2D array
        if isinstance(shap_values, list):
            sv = shap_values[1][0]   # positive class (schizophrenia)
        else:
            sv = shap_values[0]

        top_idx = np.argsort(np.abs(sv))[::-1][:12]

        return {
            "shap_values":   sv.tolist(),
            "feature_names": self.feature_names,
            "base_value":    float(self.explainer.expected_value[1]
                                   if isinstance(self.explainer.expected_value, (list, np.ndarray))
                                   else self.explainer.expected_value),
            "top_features": [
                {
                    "feature":     self.feature_names[i],
                    "shap_value":  float(sv[i]),
                    "direction":   "increases_risk" if sv[i] > 0 else "decreases_risk",
                }
                for i in top_idx
            ],
        }

    def plot_waterfall(
        self,
        features: np.ndarray,
        max_display: int = 12,
        save_path: str = None,
    ) -> bytes | None:
        """
        Render SHAP waterfall plot.

        Args:
            features:    1D feature array.
            max_display: Max number of features to show in the plot.
            save_path:   If provided, save PNG to this path. Otherwise return PNG bytes.

        Returns:
            PNG bytes if save_path is None, else None.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        X_scaled = self.scaler.transform(features)
        shap_exp = self.explainer(X_scaled)

        # Pick positive class explanation
        if shap_exp.values.ndim == 3:
            vals = shap_exp.values[:, :, 1]
        else:
            vals = shap_exp.values

        exp = shap.Explanation(
            values=vals[0],
            base_values=shap_exp.base_values[0] if shap_exp.base_values.ndim > 0 else shap_exp.base_values,
            data=X_scaled[0],
            feature_names=self.feature_names,
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, max_display=max_display, show=False)
        plt.title("SHAP Feature Attribution — Schizophrenia Risk", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            return None
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close()
            buf.seek(0)
            return buf.read()

    def plot_bar_summary(
        self,
        X: np.ndarray,
        save_path: str = None,
        max_display: int = 15,
    ) -> bytes | None:
        """
        Global feature importance: mean absolute SHAP across multiple subjects.
        Useful for research-grade summary plots.
        """
        X_scaled = self.scaler.transform(X)
        shap_values = self.explainer.shap_values(X_scaled)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv, X_scaled,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
            plot_type="bar",
        )
        plt.title("Global Biomarker Importance (SHAP)", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            return None
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close()
            buf.seek(0)
            return buf.read()
