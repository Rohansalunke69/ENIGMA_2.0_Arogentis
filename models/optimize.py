"""
Model Optimization Pipeline — Arogentis
=========================================
Scientific optimization of schizophrenia EEG detection:
  1. SHAP-based feature selection (475 → top-K)
  2. Leave-One-Subject-Out Cross-Validation (LOSOCV)
  3. Multi-model comparison: XGBoost, RandomForest, LightGBM
  4. Train vs test gap (overfitting detection)
  5. SHAP summary plot (explainability)
  6. Literature-calibrated performance expectations

Dataset: RepOD — Olejarczyk & Jernajczyk (2017)
  - 14 schizophrenia + 14 healthy controls
  - 19-channel 10-20 EEG, 250 Hz, eyes-closed resting state
  Source: https://doi.org/10.18150/repod.0107441

Usage:
    python train.py --optimize
    python -c "from models.optimize import run_optimization; run_optimization()"
"""

import json
import logging
import os
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    LeaveOneOut, StratifiedKFold, cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Try importing LightGBM (optional)
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logger.warning("LightGBM not installed. Skipping LGBM model.")

# Try importing SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not installed. Skipping SHAP explainability.")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def shap_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    top_k: int = 50,
    output_dir: str = "artifacts/optimize",
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    SHAP-based feature selection.

    Method:
      1. Train XGBoost on full data
      2. Compute SHAP values for each feature
      3. Rank by mean(|SHAP|) — measures each feature's contribution
      4. Select top-K features

    Why SHAP over PCA/correlation:
      - Preserves feature interpretability (we know WHICH brain regions matter)
      - Captures non-linear interactions that XGBoost uses
      - Clinical requirement: must explain WHY a prediction was made

    Why top_k=50:
      - Rule of thumb: N/p ratio should be > 0.5 for stable classification
      - 28 subjects / 50 features = 0.56 — borderline but reasonable
      - Literature: 30–100 features optimal for EEG classification (Sabeti 2009)

    Returns:
        X_selected, selected_feature_names, shap_importances (sorted)
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"SHAP feature selection: {X.shape[1]} → {top_k} features")

    # Train a quick XGBoost for SHAP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    xgb = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="auc",
        random_state=42, n_jobs=-1,
    )
    xgb.fit(X_scaled, y)

    if HAS_SHAP:
        # TreeExplainer for exact SHAP values
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_scaled)

        # Handle binary classification SHAP output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 (schizophrenia)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    else:
        # Fallback: use XGBoost built-in feature importance
        mean_abs_shap = xgb.feature_importances_
        logger.info("Using XGBoost feature_importances_ as SHAP fallback")

    # Rank and select top-K
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
    selected_names = [feature_names[i] for i in top_indices]
    importances = mean_abs_shap[top_indices]

    # Save feature rankings
    ranking_path = os.path.join(output_dir, "feature_ranking.txt")
    with open(ranking_path, "w") as f:
        f.write("rank,feature,importance\n")
        for rank, idx in enumerate(np.argsort(mean_abs_shap)[::-1]):
            f.write(f"{rank+1},{feature_names[idx]},{mean_abs_shap[idx]:.6f}\n")
    logger.info(f"Feature ranking saved: {ranking_path}")

    # Plot top 20
    fig, ax = plt.subplots(figsize=(10, 6))
    top20_idx = top_indices[:20]
    top20_names = [feature_names[i] for i in top20_idx]
    top20_imp = mean_abs_shap[top20_idx]
    ax.barh(range(len(top20_names)), top20_imp[::-1], color="#6c63ff", alpha=0.85)
    ax.set_yticks(range(len(top20_names)))
    ax.set_yticklabels(top20_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("Top 20 Features — SHAP Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_top20.png"), dpi=150)
    plt.close()

    # SHAP summary plot
    if HAS_SHAP:
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values[:, top_indices], X_scaled[:, top_indices],
                feature_names=selected_names, show=False, max_display=20,
            )
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=150)
            plt.close()
            logger.info("SHAP summary plot saved")
        except Exception as e:
            logger.warning(f"SHAP summary plot failed: {e}")

    X_selected = X[:, top_indices]
    logger.info(
        f"Selected top {top_k} features. "
        f"Top 5: {selected_names[:5]}"
    )
    return X_selected, selected_names, importances


def selectkbest_baseline(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    top_k: int = 50,
) -> tuple[np.ndarray, list[str]]:
    """
    SelectKBest with mutual information — baseline comparison for SHAP.

    Mutual information captures non-linear dependencies (unlike f_classif).
    """
    selector = SelectKBest(mutual_info_classif, k=top_k)
    X_selected = selector.fit_transform(X, y)
    mask = selector.get_support()
    selected_names = [n for n, m in zip(feature_names, mask) if m]
    logger.info(f"SelectKBest (MI): top {top_k} features selected")
    return X_selected, selected_names


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: LEAVE-ONE-SUBJECT-OUT CV
# ═══════════════════════════════════════════════════════════════════════════════

def losocv_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    model_name: str = "Model",
) -> dict:
    """
    Leave-One-Subject-Out Cross Validation.

    For 28 subjects: 28 folds, each with 27 train + 1 test.

    Why LOSOCV:
      - Gold standard for small clinical datasets
      - Each subject is a test set exactly once → no subject leakage possible
      - More unbiased than 5-fold for N < 50

    Also computes train accuracy per fold to detect overfitting.
    """
    from sklearn.base import clone

    loo = LeaveOneOut()
    y_true_all = []
    y_pred_all = []
    y_proba_all = []
    train_accs = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone pipeline for each fold to prevent state leakage
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)

        # Train accuracy (for overfitting detection)
        train_pred = fold_pipeline.predict(X_train)
        train_accs.append(accuracy_score(y_train, train_pred))

        # Test prediction
        y_pred = fold_pipeline.predict(X_test)
        y_proba = fold_pipeline.predict_proba(X_test)[:, 1]

        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred[0])
        y_proba_all.append(y_proba[0])

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    y_proba = np.array(y_proba_all)
    train_acc_mean = np.mean(train_accs)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)

    # Train–test gap (overfitting indicator)
    gap = train_acc_mean - acc

    metrics = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "train_accuracy": round(train_acc_mean, 4),
        "train_test_gap": round(gap, 4),
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
    }

    logger.info(
        f"[{model_name}] LOSOCV: Acc={acc:.3f} | AUC={auc:.3f} | "
        f"Sens={sensitivity:.3f} | Spec={specificity:.3f} | "
        f"Train={train_acc_mean:.3f} | Gap={gap:.3f}"
    )

    return metrics, y_true, y_proba


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: MODEL BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_models(n_features: int) -> dict[str, Pipeline]:
    """
    Build optimized model pipelines.

    Regularization choices for N=28:
      - XGBoost: shallow trees (max_depth=3), high regularization (reg_alpha, reg_lambda)
      - RF: limited depth (max_depth=6), minimum leaf samples
      - LightGBM: num_leaves=15 (2^4-1), strong L1/L2
    """
    models = {}

    # XGBoost — primary model
    models["XGBoost"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,        # L1 regularization — feature sparsity
            reg_lambda=2.0,       # L2 regularization — weight shrinkage
            min_child_weight=3,   # prevents overfitting on small groups
            scale_pos_weight=1.0,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # Random Forest — baseline
    models["RandomForest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=6,            # strict depth limit for N=28
            min_samples_leaf=3,     # prevent single-sample leaves
            min_samples_split=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # LightGBM (if available)
    if HAS_LGBM:
        models["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=150,
                num_leaves=15,         # 2^4-1, prevents overfitting
                max_depth=4,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=2.0,
                min_child_samples=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )),
        ])

    return models


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: MASTER OPTIMIZATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_optimization(
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    feature_names: Optional[list[str]] = None,
    top_k: int = 50,
    output_dir: str = "artifacts/optimize",
) -> dict:
    """
    Full optimization pipeline.

    Steps:
      1. Load data (if not provided)
      2. SHAP feature selection (475 → top_k)
      3. LOSOCV evaluation for all models
      4. Generate comprehensive report
      5. Save best model

    Returns:
        dict with all metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data if not provided
    if X is None:
        X = np.load("data/features/X.npy")
        y = np.load("data/features/y.npy")
        with open("data/features/feature_names.txt") as f:
            feature_names = [line.strip() for line in f if line.strip()]

    n_subjects, n_features = X.shape
    logger.info(f"{'='*60}")
    logger.info(f"OPTIMIZATION PIPELINE")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {n_subjects} subjects, {n_features} features")
    logger.info(f"Labels: SZ={int(y.sum())} | HC={int((y==0).sum())}")

    # ── Step 1: Feature Selection ─────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("STEP 1: SHAP Feature Selection")
    logger.info(f"{'='*60}")

    X_shap, shap_names, shap_imp = shap_feature_selection(
        X, y, feature_names, top_k=top_k, output_dir=output_dir
    )

    # Also run SelectKBest as comparison
    X_kbest, kbest_names = selectkbest_baseline(X, y, feature_names, top_k=top_k)

    # ── Step 2: LOSOCV with all models ────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("STEP 2: Leave-One-Subject-Out Cross-Validation")
    logger.info(f"{'='*60}")

    models = build_models(n_features=top_k)
    all_results = {}
    roc_data = {}

    for name, pipeline in models.items():
        logger.info(f"\n--- {name} (SHAP top-{top_k}) ---")
        metrics, y_true, y_proba = losocv_evaluate(X_shap, y, pipeline, model_name=name)
        all_results[name] = metrics
        roc_data[name] = (y_true, y_proba)

    # Also test full features with XGBoost for comparison
    logger.info(f"\n--- XGBoost (ALL {n_features} features — baseline) ---")
    xgb_full = build_models(n_features)["XGBoost"]
    metrics_full, y_true_full, y_proba_full = losocv_evaluate(
        X, y, xgb_full, model_name="XGBoost-ALL"
    )
    all_results["XGBoost-ALL-features"] = metrics_full

    # ── Step 3: Results Table ─────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"{'Model':<22} {'Acc':>6} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'F1':>6} {'Gap':>6}")
    logger.info("-" * 60)
    for name, m in all_results.items():
        logger.info(
            f"{name:<22} {m['accuracy']:6.3f} {m['roc_auc']:6.3f} "
            f"{m['recall_sensitivity']:6.3f} {m['specificity']:6.3f} "
            f"{m['f1_score']:6.3f} {m['train_test_gap']:6.3f}"
        )

    # ── Step 4: Find Best Model ───────────────────────────────────────────────
    # Excluding "ALL features" from best model selection
    filtered = {k: v for k, v in all_results.items() if "ALL" not in k}
    best_name = max(filtered, key=lambda k: filtered[k]["roc_auc"])
    best_metrics = filtered[best_name]

    logger.info(f"\n🏆 Best Model: {best_name} (ROC-AUC={best_metrics['roc_auc']:.3f})")

    # ── Step 5: Save best model (trained on all data) ─────────────────────────
    best_pipeline = build_models(top_k)[best_name]
    best_pipeline.fit(X_shap, y)
    model_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(best_pipeline, model_path)
    logger.info(f"Best model saved: {model_path}")

    # Save selected feature names
    with open(os.path.join(output_dir, "selected_features.txt"), "w") as f:
        f.write("\n".join(shap_names))

    # ── Step 6: ROC Curves ────────────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    colors = ["#6c63ff", "#ff6584", "#2ecc71", "#f39c12"]
    for i, (name, (yt, yp)) in enumerate(roc_data.items()):
        fpr, tpr, _ = roc_curve(yt, yp)
        auc_val = roc_auc_score(yt, yp)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f"{name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.5)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves — LOSOCV Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_comparison.png"), dpi=150)
    plt.close()

    # ── Step 7: Confusion Matrices ────────────────────────────────────────────
    for name, (yt, yp) in roc_data.items():
        yp_bin = (yp >= 0.5).astype(int)
        cm = confusion_matrix(yt, yp_bin)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Control", "Schizophrenia"])
        ax.set_yticklabels(["Control", "Schizophrenia"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=18, fontweight="bold",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cm_{name.lower().replace(' ', '_')}.png"), dpi=150)
        plt.close()

    # ── Step 8: Save full report ──────────────────────────────────────────────
    report = {
        "dataset": {
            "source": "RepOD — Olejarczyk & Jernajczyk (2017)",
            "doi": "10.18150/repod.0107441",
            "paradigm": "resting-state, eyes closed, awake",
            "n_subjects": int(n_subjects),
            "n_schizophrenia": int(y.sum()),
            "n_controls": int((y == 0).sum()),
            "original_features": int(n_features),
            "selected_features": top_k,
        },
        "feature_selection": {
            "method": "SHAP (TreeExplainer)" if HAS_SHAP else "XGBoost feature_importances_",
            "top_k": top_k,
            "top_10_features": shap_names[:10],
        },
        "cv_method": "Leave-One-Subject-Out (LOSOCV)",
        "results": all_results,
        "best_model": best_name,
        "best_metrics": best_metrics,
    }

    report_path = os.path.join(output_dir, "optimization_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Full report saved: {report_path}")

    # ── Step 9: Scientific Analysis ───────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("SCIENTIFIC ANALYSIS")
    logger.info(f"{'='*60}")

    best_auc = best_metrics["roc_auc"]
    best_gap = best_metrics["train_test_gap"]

    logger.info(f"\n📊 Best ROC-AUC: {best_auc:.3f}")
    logger.info(f"📊 Train-Test Gap: {best_gap:.3f}")

    if best_auc > 0.9:
        logger.warning(
            "⚠️ ROC-AUC > 0.9 detected. Possible explanations:\n"
            "  1. Overfitting — check train-test gap\n"
            "  2. Data leakage — verify subject-wise split\n"
            "  3. Very clean dataset with strong biomarkers\n"
            "Published benchmarks for this dataset: 0.65–0.85 (Olejarczyk 2017)"
        )
    elif best_auc > 0.75:
        logger.info(
            "✅ Performance is within expected range for this dataset.\n"
            "Published benchmarks: 0.65–0.85 (Olejarczyk 2017, Sabeti 2009)."
        )
    else:
        logger.info(
            "Performance is moderate. Expected for N=28 with LOSOCV.\n"
            "Factors: small sample, high inter-subject variability, limited channels.\n"
            "Published range: 0.60–0.85 depending on features and methods."
        )

    if best_gap > 0.3:
        logger.warning(
            f"⚠️ Train-test gap = {best_gap:.3f}. Possible overfitting.\n"
            "Recommendations: increase regularization, reduce features further."
        )
    else:
        logger.info(f"✅ Train-test gap = {best_gap:.3f}. No significant overfitting.")

    logger.info(f"\n{'='*60}")
    logger.info(f"📁 All results saved to: {output_dir}")
    logger.info(f"📁 Best model: {model_path}")
    logger.info(f"{'='*60}")

    return report
