"""
Training Script — Arogentis
===============================
Run this to generate a trained model artifact before launching the API or dashboard.

Usage:
    # With synthetic data (no real EEG needed — for testing):
    python train.py --synthetic

    # With real PhysioNet data:
    python train.py --data_dir data/raw --label_file data/labels.csv

    # With advanced XGBoost tuning:
    python train.py --synthetic --advanced
"""

import argparse
import logging
import numpy as np
import os

import compat  # noqa: F401 — patch NumPy 2.x before MNE loads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Arogentis Model Training")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic EEG data (no real data needed)")
    parser.add_argument("--physionet", action="store_true",
                        help="Download + train on PhysioNet schizophrenia resting-state EEG")
    parser.add_argument("--advanced", action="store_true",
                        help="Also train XGBoost with hyperparameter tuning")
    parser.add_argument("--optimize", action="store_true",
                        help="Run full scientific optimization: SHAP feature selection + LOSOCV + multi-model")
    parser.add_argument("--data_dir", default="data/raw",
                        help="Directory with raw EEG files")
    parser.add_argument("--label_file", default="data/labels.csv",
                        help="CSV with columns: filename, label")
    parser.add_argument("--output_dir", default="data/features",
                        help="Where to save X.npy, y.npy, feature_names.txt")
    parser.add_argument("--model_dir", default="artifacts",
                        help="Where to save trained model .pkl files")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    # ── Step 1: Build or load feature dataset ─────────────────────────────────
    X_path    = os.path.join(args.output_dir, "X.npy")
    y_path    = os.path.join(args.output_dir, "y.npy")
    feat_path = os.path.join(args.output_dir, "feature_names.txt")

    if args.synthetic or args.physionet or not (os.path.exists(X_path) and os.path.exists(y_path)):
        logger.info("Generating dataset...")
        from pipeline.dataset_builder import generate_synthetic_dataset, build_dataset
        if args.synthetic:
            X, y, feature_names = generate_synthetic_dataset(output_dir=args.output_dir)
        elif args.physionet:
            # Download PhysioNet schizophrenia resting-state EEG dataset
            logger.info("=" * 60)
            logger.info("DOWNLOADING PhysioNet Schizophrenia Resting-State EEG Dataset")
            logger.info("Olejarczyk & Jernajczyk (2017) — 14 SZ + 14 HC subjects")
            logger.info("=" * 60)
            from pipeline.data_downloader import download_dataset, generate_labels_csv
            download_dataset(output_dir=args.data_dir)
            label_file = generate_labels_csv(
                output_dir=os.path.dirname(args.data_dir) or "data",
                data_dir=args.data_dir,
            )
            X, y, feature_names = build_dataset(
                data_dir=args.data_dir,
                label_file=label_file,
                output_dir=args.output_dir,
            )
        else:
            X, y, feature_names = build_dataset(
                data_dir=args.data_dir,
                label_file=args.label_file,
                output_dir=args.output_dir,
            )
    else:
        logger.info(f"Loading existing dataset from {args.output_dir}...")
        X = np.load(X_path)
        y = np.load(y_path)
        with open(feat_path) as f:
            feature_names = [l.strip() for l in f if l.strip()]

    logger.info(f"Dataset: X={X.shape}, y={y.shape} | "
                f"Schizophrenia: {y.sum()} | Controls: {(y==0).sum()}")

    # ── Step 2: Baseline Model (RandomForest / SVM) ───────────────────────────
    from models.baseline_model import run_baseline_training
    baseline_results = run_baseline_training(X, y, save_dir=args.model_dir)
    best_name = baseline_results["best_model"]
    rf_auc  = baseline_results["random_forest"]["roc_auc"]["mean"]
    svm_auc = baseline_results["svm"]["roc_auc"]["mean"]
    winner_auc = rf_auc if best_name == "RandomForest" else svm_auc
    logger.info(f"Baseline complete. Best model: {best_name} | ROC-AUC: {winner_auc:.3f}")

    # ── Step 3: Evaluation Report ─────────────────────────────────────────────
    from models.evaluation import full_evaluation_report
    metrics = full_evaluation_report(
        X, y,
        model_path=baseline_results["model_path"],
        output_dir=os.path.join(args.model_dir, "eval"),
    )
    logger.info(
        f"Evaluation → ROC-AUC: {metrics['roc_auc']:.3f} | "
        f"Sensitivity: {metrics['sensitivity']:.3f} | "
        f"Specificity: {metrics['specificity']:.3f}"
    )

    # ── Step 4 (Optional): Advanced XGBoost ───────────────────────────────────
    if args.advanced:
        logger.info("Training advanced XGBoost model with hyperparameter tuning...")
        from models.advanced_model import tune_xgb
        xgb_model = tune_xgb(
            X, y,
            n_iter=20,
            save_path=os.path.join(args.model_dir, "xgb_model.pkl"),
        )
        logger.info("XGBoost training complete.")

    # ── Step 5 (Optional): Full Scientific Optimization ───────────────────────
    if args.optimize:
        logger.info("Running full scientific optimization pipeline...")
        from models.optimize import run_optimization
        report = run_optimization(
            X=X, y=y, feature_names=feature_names,
            top_k=50,
            output_dir=os.path.join(args.model_dir, "optimize"),
        )
        logger.info(f"Optimization complete. Best: {report['best_model']} "
                    f"(AUC={report['best_metrics']['roc_auc']:.3f})")

    logger.info("✅ All training complete. Model saved to: " + args.model_dir)
    logger.info("Launch API:       uvicorn backend.main:app --reload")
    logger.info("Launch Dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
