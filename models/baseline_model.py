"""
Baseline ML Models — Arogentis
================================
RandomForest + SVM pipelines with stratified cross-validation.

Design choices:
  - StandardScaler: EEG features span many orders of magnitude (µV² vs ratio values)
  - class_weight='balanced': Clinical EEG datasets are always imbalanced
  - StratifiedKFold: preserves class ratio in each fold (critical for small N)
  - ROC-AUC scoring: correct metric for imbalanced binary classification
  - Probability output: we are a risk scoring system, not a binary classifier
"""

import logging
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


# ─── Pipeline Factories ───────────────────────────────────────────────────────

def build_rf_pipeline() -> Pipeline:
    """
    RandomForest pipeline.
    n_estimators=300: more trees = lower variance, stable at this dataset size.
    max_depth=10: prevents overfitting on small clinical datasets (N~28).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_svm_pipeline() -> Pipeline:
    """
    RBF-SVM pipeline.
    SVM with probability=True enables calibrated probability output via Platt scaling.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )),
    ])


# ─── Training & Evaluation ─────────────────────────────────────────────────────

def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    model_name: str = "model",
    n_splits: int = 5,
) -> dict:
    """
    Stratified K-Fold cross-validation with multiple metrics.

    Returns dict with: roc_auc, accuracy, f1, sensitivity, specificity (mean ± std).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        "roc_auc":  "roc_auc",
        "accuracy": "accuracy",
        "f1":       "f1",
        "sensitivity": "recall",
        "specificity": "precision",
    }
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    metrics = {}
    for key in scoring:
        scores = results[f"test_{key}"]
        metrics[key] = {"mean": float(scores.mean()), "std": float(scores.std())}
        logger.info(f"[{model_name}] {key}: {scores.mean():.3f} ± {scores.std():.3f}")

    return metrics


def train_and_save(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    save_path: str,
) -> Pipeline:
    """
    Fit pipeline on full dataset and save to disk.

    NOTE: We evaluate with CV first, then fit on ALL data for deployment.
    """
    pipeline.fit(X, y)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    logger.info(f"Model saved to: {save_path}")
    return pipeline


def run_baseline_training(
    X: np.ndarray,
    y: np.ndarray,
    save_dir: str = "artifacts",
) -> dict:
    """
    Full baseline training: RF + SVM evaluation + save best model.

    Returns:
        Dictionary with metrics for both models and path to the best saved model.
    """
    logger.info(f"Training baseline models on {X.shape[0]} subjects, {X.shape[1]} features.")

    rf_pipeline = build_rf_pipeline()
    svm_pipeline = build_svm_pipeline()

    rf_metrics  = cross_validate_model(X, y, rf_pipeline,  model_name="RandomForest")
    svm_metrics = cross_validate_model(X, y, svm_pipeline, model_name="SVM")

    # Choose best model by ROC-AUC
    if rf_metrics["roc_auc"]["mean"] >= svm_metrics["roc_auc"]["mean"]:
        best_pipeline, best_name = rf_pipeline, "RandomForest"
    else:
        best_pipeline, best_name = svm_pipeline, "SVM"

    logger.info(f"Best baseline model: {best_name}")
    save_path = os.path.join(save_dir, "rf_model.pkl")
    train_and_save(X, y, best_pipeline, save_path)

    return {
        "random_forest": rf_metrics,
        "svm": svm_metrics,
        "best_model": best_name,
        "model_path": save_path,
    }
