"""
Advanced Models — Arogentis
==============================
XGBoost + LightGBM with hyperparameter tuning via RandomizedSearchCV.

Why XGBoost/LightGBM over RandomForest?
  - Gradient boosting corrects misclassifications iteratively → better on small datasets
  - Native support for imbalanced data (scale_pos_weight)
  - Faster inference than deep models → suitable for 24h hackathon
  - SHAP TreeExplainer works natively with both (exact, not approximate)
"""

import logging
import os

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def build_xgb_pipeline(scale_pos_weight: float = 1.0) -> Pipeline:
    """
    XGBoost pipeline with sensible defaults for small clinical datasets.

    scale_pos_weight: ratio of negatives to positives — corrects class imbalance.
    E.g., 14 controls / 14 schiz → scale_pos_weight = 1.0 (balanced dataset)
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=4,           # shallow trees prevent overfitting (N~28)
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def tune_xgb(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 20,
    save_path: str = "artifacts/xgb_model.pkl",
) -> Pipeline:
    """
    Randomized hyperparameter search for XGBoost.

    n_iter=20 is sufficient for hackathon — covers the most impactful parameters.
    """
    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = neg / (pos + 1e-9)

    pipeline = build_xgb_pipeline(scale_pos_weight=scale_pos_weight)

    param_dist = {
        "clf__n_estimators":   [100, 200, 300, 500],
        "clf__max_depth":      [3, 4, 5, 6],
        "clf__learning_rate":  [0.01, 0.05, 0.1, 0.2],
        "clf__subsample":      [0.6, 0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline, param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y)

    logger.info(
        f"Best XGBoost ROC-AUC: {search.best_score_:.3f} | "
        f"Params: {search.best_params_}"
    )

    best = search.best_estimator_
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(best, save_path)
    logger.info(f"XGBoost model saved to: {save_path}")
    return best
