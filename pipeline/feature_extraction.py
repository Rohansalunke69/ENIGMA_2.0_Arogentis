"""
EEG Feature Extraction — Arogentis
====================================
Extracts neuroscientifically validated frequency-domain biomarkers from MNE Epochs.

Biomarkers computed:
  1. Band power (delta, theta, alpha, beta, gamma) — per channel via Welch PSD
  2. Relative band power — normalised by total power (removes amplitude bias)
  3. Spectral entropy — measures signal disorder/complexity
  4. Alpha/Theta ratio — composite cognitive biomarker
  5. Peak frequency — dominant oscillation frequency per band

All features are averaged across epochs to yield one feature vector per EEG recording,
suitable for subject-level (not epoch-level) classification.
"""

import logging
from typing import Optional

import mne
import numpy as np
from mne.time_frequency import psd_array_welch
from scipy.stats import entropy as scipy_entropy

logger = logging.getLogger(__name__)

# ─── Clinically Validated Frequency Bands ─────────────────────────────────────
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),   # slow oscillations — elevated in schizophrenia prefrontal regions
    "theta": (4.0, 8.0),   # working memory encoding — coherence disrupted in schizophrenia
    "alpha": (8.0, 13.0),  # cortical inhibition — reduced = hyperexcitability marker
    "beta":  (13.0, 30.0), # active cognition — desynchronisation pattern differs
    "gamma": (30.0, 45.0), # perceptual binding, NMDA-receptor linked — most significant marker
    # Cap at 45 Hz to avoid EMG contamination from facial muscles
}


def compute_band_powers(
    epochs: mne.Epochs,
    absolute: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute per-channel absolute (and optionally relative) band power via Welch PSD.

    Args:
        epochs:   Clean MNE Epochs.
        absolute: If True, returns absolute band power. Always also computes relative.

    Returns:
        features:      (n_epochs, n_channels * n_bands) feature matrix.
        feature_names: Human-readable name for each column.
    """
    sfreq = epochs.info["sfreq"]
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    n_epochs, n_channels, _ = data.shape
    n_bands = len(BANDS)

    all_features = []
    for epoch in data:
        # epoch shape: (n_channels, n_times)
        psds, freqs = psd_array_welch(
            epoch, sfreq=sfreq, fmin=0.5, fmax=45.0,
            n_fft=int(sfreq * 2),  # 2-second FFT window = 0.5 Hz resolution
            verbose=False
        )
        # psds shape: (n_channels, n_freqs)
        total_power = psds.sum(axis=1, keepdims=True) + 1e-12  # avoid div by zero

        epoch_feats = []
        for band_name, (fmin, fmax) in BANDS.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            abs_power = psds[:, mask].mean(axis=1)            # (n_channels,)
            rel_power = (psds[:, mask].sum(axis=1)) / total_power.squeeze()
            epoch_feats.append(abs_power)
            epoch_feats.append(rel_power)

        all_features.append(np.concatenate(epoch_feats))

    features = np.array(all_features)  # (n_epochs, n_channels * n_bands * 2)

    ch_names = [f"CH{i+1}" for i in range(n_channels)]
    names = []
    for band_name in BANDS:
        names += [f"abs_{band_name}_{ch}" for ch in ch_names]
        names += [f"rel_{band_name}_{ch}" for ch in ch_names]

    return features, names


def compute_spectral_entropy(epochs: mne.Epochs) -> tuple[np.ndarray, list[str]]:
    """
    Compute spectral entropy per channel.

    Spectral entropy = Shannon entropy of the normalised PSD.
    Higher entropy → less organised oscillatory structure → schizophrenia signature.

    Returns:
        features:      (n_epochs, n_channels)
        feature_names: List of column name strings.
    """
    sfreq = epochs.info["sfreq"]
    data = epochs.get_data()
    n_channels = data.shape[1]

    entropies = []
    for epoch in data:
        psds, _ = psd_array_welch(
            epoch, sfreq=sfreq, fmin=0.5, fmax=45.0,
            n_fft=int(sfreq * 2), verbose=False
        )
        psds_norm = psds / (psds.sum(axis=1, keepdims=True) + 1e-12)
        ent = scipy_entropy(psds_norm, axis=1)   # (n_channels,)
        entropies.append(ent)

    features = np.array(entropies)  # (n_epochs, n_channels)
    names = [f"spectral_entropy_CH{i+1}" for i in range(n_channels)]
    return features, names


def compute_alpha_theta_ratio(epochs: mne.Epochs) -> tuple[np.ndarray, list[str]]:
    """
    Compute Alpha/Theta power ratio per channel.

    Clinical relevance: Reduced alpha/theta ratio correlates with:
      - Positive symptom severity (hallucinations, delusions)
      - Working memory impairment in schizophrenia

    Returns:
        features:      (n_epochs, n_channels)
        feature_names: List of column name strings.
    """
    sfreq = epochs.info["sfreq"]
    data = epochs.get_data()
    n_channels = data.shape[1]

    ratios = []
    for epoch in data:
        psds, freqs = psd_array_welch(
            epoch, sfreq=sfreq, fmin=0.5, fmax=45.0,
            n_fft=int(sfreq * 2), verbose=False
        )
        alpha = psds[:, (freqs >= 8.0) & (freqs < 13.0)].mean(axis=1)
        theta = psds[:, (freqs >= 4.0) & (freqs < 8.0)].mean(axis=1)
        ratio = alpha / (theta + 1e-12)
        ratios.append(ratio)

    features = np.array(ratios)  # (n_epochs, n_channels)
    names = [f"alpha_theta_ratio_CH{i+1}" for i in range(n_channels)]
    return features, names


def compute_gamma_theta_ratio(epochs: mne.Epochs) -> tuple[np.ndarray, list[str]]:
    """
    Compute Gamma/Theta ratio — proxy for NMDA receptor function.

    WHY: Gamma oscillations depend on GABAergic interneurons driven by NMDA receptors.
    NMDA hypofunction is the leading neurochemical hypothesis for schizophrenia.
    Reduced gamma/theta ratio = disrupted NMDA signalling.
    """
    sfreq = epochs.info["sfreq"]
    data = epochs.get_data()
    n_channels = data.shape[1]

    ratios = []
    for epoch in data:
        psds, freqs = psd_array_welch(
            epoch, sfreq=sfreq, fmin=0.5, fmax=45.0,
            n_fft=int(sfreq * 2), verbose=False
        )
        gamma = psds[:, (freqs >= 30.0) & (freqs < 45.0)].mean(axis=1)
        theta = psds[:, (freqs >= 4.0) & (freqs < 8.0)].mean(axis=1)
        ratio = gamma / (theta + 1e-12)
        ratios.append(ratio)

    features = np.array(ratios)
    names = [f"gamma_theta_ratio_CH{i+1}" for i in range(n_channels)]
    return features, names


def extract_all_features(
    epochs: mne.Epochs,
    aggregate: bool = True
) -> tuple[np.ndarray, list[str]]:
    """
    Combine all biomarkers into a single feature matrix.

    Args:
        epochs:    Clean MNE Epochs.
        aggregate: If True, average features across epochs → single row per subject.
                   If False, return per-epoch feature matrix.

    Returns:
        X:     Feature matrix. Shape (1, n_features) if aggregate else (n_epochs, n_features).
        names: Feature name list for interpretability and SHAP axis labels.
    """
    band_feats, band_names     = compute_band_powers(epochs)
    ent_feats,  ent_names      = compute_spectral_entropy(epochs)
    at_feats,   at_names       = compute_alpha_theta_ratio(epochs)
    gt_feats,   gt_names       = compute_gamma_theta_ratio(epochs)

    X = np.concatenate([band_feats, ent_feats, at_feats, gt_feats], axis=1)
    names = band_names + ent_names + at_names + gt_names

    if aggregate:
        X = X.mean(axis=0, keepdims=True)  # average across epochs → (1, n_features)

    logger.info(
        f"Extracted {X.shape[1]} features from {len(epochs)} epochs. "
        f"Output shape: {X.shape}"
    )
    return X, names
