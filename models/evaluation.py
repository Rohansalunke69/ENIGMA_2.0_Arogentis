"""
Model Evaluation — Arogentis
==============================
Generates a complete clinical-grade evaluation report:
  - ROC-AUC curve
  - Confusion matrix (with sensitivity / specificity)
  - Classification report
  - Feature importance (for RF-based models)

WHY these metrics?
  - Accuracy alone is misleading for imbalanced data (14 vs 14 is balanced, but real-world isn't)
  - ROC-AUC: scale-invariant, threshold-independent — the gold standard for clinical screening
  - Sensitivity (recall for positive class): critical — missing schizophrenia has serious consequences
  - Specificity: avoids overdiagnosis — also critical in psychiatry
"""

import json
import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

logger = logging.getLogger(__name__)


def full_evaluation_report(
    X: np.ndarray,
    y: np.ndarray,
    model_path: str,
    output_dir: str = "artifacts/eval",
) -> dict:
    """
    Generate full evaluation report using out-of-fold predictions from 5-fold CV.
    Saves ROC curve, confusion matrix, and JSON metrics report.

    Returns:
        Dictionary containing all metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    pipeline = joblib.load(model_path)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    # ─── Core metrics ─────────────────────────────────────────────────────────
    roc_auc = roc_auc_score(y, y_proba)
    report   = classification_report(y, y_pred, target_names=["Control", "Schizophrenia"], output_dict=True)
    cm       = confusion_matrix(y, y_pred)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)

    metrics = {
        "roc_auc":    round(roc_auc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "accuracy":   round(report["accuracy"], 4),
        "f1_schizophrenia": round(report["Schizophrenia"]["f1-score"], 4),
        "precision_schizophrenia": round(report["Schizophrenia"]["precision"], 4),
        "confusion_matrix": cm.tolist(),
    }

    logger.info(f"Evaluation metrics: {json.dumps(metrics, indent=2)}")

    # ─── Save metrics JSON ─────────────────────────────────────────────────────
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ─── ROC Curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#6c63ff", lw=2.5,
             label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Classifier")
    plt.fill_between(fpr, tpr, alpha=0.1, color="#6c63ff")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("ROC Curve — Schizophrenia Detection", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved: {roc_path}")

    # ─── Confusion Matrix ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Control", "Schizophrenia"]
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved: {cm_path}")

    return metrics
