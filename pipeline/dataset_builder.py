"""
Dataset Builder — Arogentis
============================
End-to-end pipeline orchestrator: raw EEG folder → feature CSV + labels.

Usage:
    python -m pipeline.dataset_builder \\
        --data_dir data/raw \\
        --output_dir data/features \\
        --label_file data/labels.csv

Label CSV format:
    filename,label
    subject01.edf,1        # 1 = schizophrenia
    subject02.edf,0        # 0 = healthy control
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.preprocessing import run_preprocessing
from pipeline.feature_extraction import extract_all_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def build_dataset(
    data_dir: str,
    label_file: str,
    output_dir: str,
    notch_freq: float = 50.0,
    epoch_duration: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build feature matrix X and label vector y from a folder of EEG files.

    Args:
        data_dir:       Directory containing raw .edf/.fif files.
        label_file:     CSV with columns [filename, label].
        output_dir:     Where to save X.npy, y.npy, feature_names.txt.
        notch_freq:     Notch filter frequency (50 Hz default).
        epoch_duration: Epoch length in seconds (default 2.0).

    Returns:
        X, y, feature_names
    """
    os.makedirs(output_dir, exist_ok=True)
    labels_df = pd.read_csv(label_file)

    X_rows, y_rows = [], []
    feature_names = None
    failed = []

    for _, row in labels_df.iterrows():
        filepath = os.path.join(data_dir, row["filename"])
        if not os.path.exists(filepath):
            logger.warning(f"File not found, skipping: {filepath}")
            continue

        try:
            logger.info(f"Processing: {row['filename']} (label={row['label']})")
            epochs = run_preprocessing(
                filepath,
                notch_freq=notch_freq,
                epoch_duration=epoch_duration,
            )

            if len(epochs) == 0:
                logger.warning(f"No clean epochs after artifact rejection: {row['filename']}")
                continue

            X_subj, names = extract_all_features(epochs, aggregate=True)
            X_rows.append(X_subj[0])
            y_rows.append(int(row["label"]))
            feature_names = names

        except Exception as e:
            logger.error(f"Failed to process {row['filename']}: {e}")
            failed.append(row["filename"])

    if not X_rows:
        raise RuntimeError("No subjects were successfully processed. Check data_dir and label_file.")

    X = np.array(X_rows)   # (n_subjects, n_features)
    y = np.array(y_rows)   # (n_subjects,)

    # ─── Save outputs ──────────────────────────────────────────────────────────
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))

    logger.info(
        f"Dataset built: {X.shape[0]} subjects, {X.shape[1]} features. "
        f"Schizophrenia: {y.sum()} | Controls: {(y == 0).sum()}. "
        f"Failed: {len(failed)}. Saved to: {output_dir}"
    )
    if failed:
        logger.warning(f"Failed files: {failed}")

    return X, y, feature_names


def generate_synthetic_dataset(
    n_schizophrenia: int = 14,
    n_controls: int = 14,
    n_channels: int = 19,
    sfreq: float = 250.0,
    duration_seconds: float = 60.0,
    output_dir: str = "data/features",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate a synthetic EEG feature dataset for testing without real data.

    Models schizophrenia-like patterns:
      - Elevated gamma/theta ratio variance
      - Higher spectral entropy
      - Lower alpha/theta ratio

    Returns:
        X, y, feature_names — ready for model training.
    """
    import mne
    from pipeline.feature_extraction import extract_all_features, BANDS

    rng = np.random.RandomState(seed)
    os.makedirs(output_dir, exist_ok=True)

    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info.set_montage("standard_1020", on_missing="ignore")

    n_times = int(sfreq * duration_seconds)

    X_rows, y_rows = [], []
    feature_names = None

    for label, n_subj in [(1, n_schizophrenia), (0, n_controls)]:
        for i in range(n_subj):
            # Simulate label-specific EEG patterns
            if label == 1:  # schizophrenia: elevated delta + theta, reduced alpha + gamma
                freqs_weights = {"delta": 2.5, "theta": 2.0, "alpha": 0.5, "beta": 1.0, "gamma": 0.3}
            else:            # healthy: balanced, strong alpha
                freqs_weights = {"delta": 0.8, "theta": 0.9, "alpha": 2.0, "beta": 1.2, "gamma": 1.0}

            data = np.zeros((n_channels, n_times))
            t = np.linspace(0, duration_seconds, n_times)
            for band, (fmin, fmax) in BANDS.items():
                center_freq = (fmin + fmax) / 2
                amp = freqs_weights.get(band, 1.0) * 10e-6  # µV scale
                data += amp * np.sin(2 * np.pi * center_freq * t) + rng.normal(0, 2e-6, (n_channels, n_times))

            # ── Avoid mne.make_fixed_length_epochs which triggers the Numba path ──
            # Instead: manually slice data into 2-second epochs using numpy,
            # then create MNE Epochs directly, keeping MNE's Numba JIT dormant.
            n_epoch_samples = int(sfreq * 2.0)
            n_complete_epochs = n_times // n_epoch_samples
            events = np.column_stack([
                np.arange(n_complete_epochs) * n_epoch_samples,
                np.zeros(n_complete_epochs, dtype=int),
                np.ones(n_complete_epochs, dtype=int),
            ])
            raw = mne.io.RawArray(data, info, verbose=False)
            epochs = mne.Epochs(
                raw, events,
                tmin=0.0, tmax=2.0,
                baseline=None,
                preload=True,
                verbose=False,
            )
            X_subj, names = extract_all_features(epochs, aggregate=True)
            X_rows.append(X_subj[0])
            y_rows.append(label)
            feature_names = names

    X = np.array(X_rows)
    y = np.array(y_rows)

    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))

    logger.info(f"Synthetic dataset saved: {X.shape} | Labels: {y.tolist()}")
    return X, y, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arogentis Dataset Builder")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--label_file", default="data/labels.csv")
    parser.add_argument("--output_dir", default="data/features")
    parser.add_argument("--notch_freq", type=float, default=50.0)
    parser.add_argument("--epoch_duration", type=float, default=2.0)
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic dataset for testing (no real EEG needed)")
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_dataset(output_dir=args.output_dir)
    else:
        build_dataset(
            data_dir=args.data_dir,
            label_file=args.label_file,
            output_dir=args.output_dir,
            notch_freq=args.notch_freq,
            epoch_duration=args.epoch_duration,
        )
