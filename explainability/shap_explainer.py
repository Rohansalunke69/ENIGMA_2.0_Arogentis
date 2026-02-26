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
    """SHAP explainability wrapper for schizophrenia risk models.

    Auto-detects model type and selects the correct SHAP explainer:
      - TreeExplainer for: RandomForest, XGBoost, LightGBM, GradientBoosting
      - KernelExplainer for: SVM, LogisticRegression, other models
    """

    # Model types that support TreeExplainer (exact, fast)
    TREE_MODEL_TYPES = (
        "RandomForestClassifier", "GradientBoostingClassifier",
        "XGBClassifier", "LGBMClassifier",
        "ExtraTreesClassifier", "DecisionTreeClassifier",
    )

    def __init__(self, model_path: str, feature_names: list[str]):
        """
        Args:
            model_path:    Path to saved sklearn Pipeline (.pkl).
            feature_names: Ordered list matching feature columns.
        """
        self.pipeline      = joblib.load(model_path)
        self.clf           = self.pipeline.named_steps["clf"]
        self.scaler        = self.pipeline.named_steps["scaler"]
        self.feature_names = feature_names
        self.explainer_type = None

        clf_name = type(self.clf).__name__

        if clf_name in self.TREE_MODEL_TYPES:
            # TreeExplainer: exact SHAP values, fast
            self.explainer = shap.TreeExplainer(self.clf)
            self.explainer_type = "tree"
            logger.info(f"SHAP TreeExplainer initialised for {clf_name} with {len(feature_names)} features.")
        else:
            # KernelExplainer: model-agnostic, works with SVM/LR/any model
            # Use a small background sample for efficiency
            logger.info(f"Model is {clf_name} — using SHAP KernelExplainer (model-agnostic).")
            # Create a small background dataset (zeros as baseline)
            background = np.zeros((1, len(feature_names)))
            self.explainer = shap.KernelExplainer(
                self.pipeline.predict_proba, background
            )
            self.explainer_type = "kernel"
            logger.info(f"SHAP KernelExplainer initialised for {clf_name} with {len(feature_names)} features.")

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

        if self.explainer_type == "tree":
            # TreeExplainer operates on the raw classifier → needs scaled input
            X_input = self.scaler.transform(features)
            shap_values = self.explainer.shap_values(X_input)
        else:
            # KernelExplainer wraps pipeline.predict_proba → pass raw features
            X_input = features
            shap_values = self.explainer.shap_values(X_input, nsamples=100)

        # For binary classification: shap_values may be list [class0, class1] or 2D/3D array
        if isinstance(shap_values, list):
            sv = shap_values[1][0]   # positive class (schizophrenia)
        elif shap_values.ndim == 3:
            sv = shap_values[0, :, 1]  # (samples, features, classes) → class 1
        else:
            sv = shap_values[0]

        top_idx = np.argsort(np.abs(sv))[::-1][:12]

        # Extract base value safely
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base_val = float(ev[1]) if len(ev) > 1 else float(ev[0])
        else:
            base_val = float(ev)

        return {
            "shap_values":   sv.tolist(),
            "feature_names": self.feature_names,
            "base_value":    base_val,
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

        # Safely extract scalar base_value (handles scalar, 1D, and multi-class)
        try:
            bv = shap_exp.base_values
            if isinstance(bv, np.ndarray):
                if bv.ndim == 0:
                    base_val = float(bv)
                elif bv.ndim == 1:
                    # [base_class0, base_class1] → pick class 1 (schizophrenia)
                    base_val = float(bv[1]) if bv.shape[0] > 1 else float(bv[0])
                elif bv.ndim == 2:
                    # (n_samples, n_classes) → first sample, class 1
                    base_val = float(bv[0, 1]) if bv.shape[1] > 1 else float(bv[0, 0])
                else:
                    base_val = float(bv.flat[0])
            else:
                base_val = float(bv)
        except Exception:
            base_val = 0.0

        exp = shap.Explanation(
            values=vals[0],
            base_values=base_val,
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
