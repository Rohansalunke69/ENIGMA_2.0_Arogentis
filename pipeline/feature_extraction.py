"""
EEG Feature Extraction — Arogentis
====================================
Extracts neuroscientifically validated biomarkers from resting-state EEG epochs.

Biomarkers computed:
  1. Band power (delta, theta, alpha, beta, gamma) — absolute & relative via Welch PSD
  2. Spectral entropy — Shannon entropy of normalised PSD (signal disorder)
  3. Alpha/Theta ratio — cognitive engagement / cortical inhibition marker
  4. Gamma/Theta ratio — NMDA receptor function proxy
  5. Statistical features — mean, variance, skewness, kurtosis per channel
  6. Hjorth parameters — activity, mobility, complexity per channel
  7. Differential entropy — log-variance of band-filtered signal per band

WHY RESTING-STATE EEG FOR SCHIZOPHRENIA:
  Resting-state (awake, eyes closed) EEG captures spontaneous brain oscillations
  WITHOUT confounds from task performance or sleep staging. Schizophrenia disrupts
  intrinsic neural oscillatory patterns (gamma/theta coherence, alpha blocking)
  that are best captured at rest. Sleep EEG (PSG) reflects entirely different
  physiology (sleep spindles, K-complexes, slow-wave oscillations) and is NOT
  appropriate for psychiatric disorder classification.

  References:
    - Olejarczyk & Jernajczyk (2017): Resting-state EEG in schizophrenia
    - Uhlhaas & Singer (2010): Gamma oscillations and NMDA hypothesis
    - Boutros et al. (2008): Alpha power deficit in schizophrenia
"""

import logging
from typing import Optional

import mne
import numpy as np
from mne.time_frequency import psd_array_welch
from scipy.stats import entropy as scipy_entropy, skew, kurtosis

logger = logging.getLogger(__name__)

# ─── Clinically Validated Frequency Bands ─────────────────────────────────────
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),   # slow oscillations — elevated in schizophrenia prefrontal regions
    "theta": (4.0, 8.0),   # working memory encoding — coherence disrupted in schizophrenia
    "alpha": (8.0, 13.0),  # cortical inhibition — reduced = hyperexcitability marker
    "beta":  (13.0, 30.0), # active cognition — desynchronisation pattern differs
    "gamma": (30.0, 40.0), # perceptual binding, NMDA-receptor linked — most significant marker
    # Cap at 40 Hz (bandpass limit) to avoid EMG contamination
}


# ═══════════════════════════════════════════════════════════════════════════════
# FREQUENCY-DOMAIN FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_band_powers(
    epochs: mne.Epochs,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute per-channel absolute + relative band power via Welch PSD.
    Also computes differential entropy per band (log-variance of band power).

    Clinical relevance:
      - Elevated delta/theta in frontal regions → positive symptom severity
      - Reduced alpha power → cortical hyperexcitability
      - Reduced gamma power → NMDA hypofunction (glutamate hypothesis)

    Returns:
        features:      (n_epochs, n_channels * n_bands * 3) feature matrix.
        feature_names: Human-readable name for each column.
    """
    sfreq = epochs.info["sfreq"]
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    n_epochs, n_channels, _ = data.shape

    all_features = []
    for epoch in data:
        # epoch shape: (n_channels, n_times)
        psds, freqs = psd_array_welch(
            epoch, sfreq=sfreq, fmin=1.0, fmax=40.0,
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
            # Differential entropy: log of band power variance
            diff_ent = np.log(psds[:, mask].var(axis=1) + 1e-12)
            epoch_feats.append(abs_power)
            epoch_feats.append(rel_power)
            epoch_feats.append(diff_ent)

        all_features.append(np.concatenate(epoch_feats))

    features = np.array(all_features)  # (n_epochs, n_channels * n_bands * 3)

    ch_names = [f"CH{i+1}" for i in range(n_channels)]
    names = []
    for band_name in BANDS:
        names += [f"abs_{band_name}_{ch}" for ch in ch_names]
        names += [f"rel_{band_name}_{ch}" for ch in ch_names]
        names += [f"diffent_{band_name}_{ch}" for ch in ch_names]

    return features, names


def compute_spectral_entropy(epochs: mne.Epochs) -> tuple[np.ndarray, list[str]]:
    """
    Compute spectral entropy per channel — Shannon entropy of the normalised PSD.

    Clinical relevance:
      Higher spectral entropy → less organised oscillatory structure
      → disrupted neural synchrony → schizophrenia signature.
      Schizophrenia patients show elevated spectral entropy in frontal/temporal
      regions due to disorganised gamma-band activity (Uhlhaas & Singer, 2010).

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
            epoch, sfreq=sfreq, fmin=1.0, fmax=40.0,
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
    Alpha/Theta power ratio per channel.

    Clinical relevance:
      Reduced alpha/theta ratio correlates with:
        - Positive symptom severity (hallucinations, delusions)
        - Working memory impairment in schizophrenia
        - Cortical inhibition deficit (alpha suppression + theta excess)
      Reference: Boutros et al. (2008)
    """
    sfreq = epochs.info["sfreq"]
    data = epochs.get_data()
    n_channels = data.shape[1]

    ratios = []
    for epoch in data:
        psds, freqs = psd_array_welch(
            epoch, sfreq=sfreq, fmin=1.0, fmax=40.0,
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
    Gamma/Theta ratio — proxy for NMDA receptor function.

    Clinical relevance:
      Gamma oscillations depend on GABAergic interneurons driven by NMDA receptors.
      NMDA hypofunction is the leading neurochemical hypothesis for schizophrenia.
      Reduced gamma/theta ratio = disrupted NMDA signalling.
      Reference: Uhlhaas & Singer (2010)
    """
    sfreq = epochs.info["sfreq"]
    data = epochs.get_data()
    n_channels = data.shape[1]

    ratios = []
    for epoch in data:
        psds, freqs = psd_array_welch(
            epoch, sfreq=sfreq, fmin=1.0, fmax=40.0,
            n_fft=int(sfreq * 2), verbose=False
        )
        gamma = psds[:, (freqs >= 30.0) & (freqs < 40.0)].mean(axis=1)
        theta = psds[:, (freqs >= 4.0) & (freqs < 8.0)].mean(axis=1)
        ratio = gamma / (theta + 1e-12)
        ratios.append(ratio)

    features = np.array(ratios)
    names = [f"gamma_theta_ratio_CH{i+1}" for i in range(n_channels)]
    return features, names


# ═══════════════════════════════════════════════════════════════════════════════
# TIME-DOMAIN FEATURES (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_statistical_features(epochs: mne.Epochs) -> tuple[np.ndarray, list[str]]:
    """
    Compute per-channel time-domain statistical features.

    Features per channel: mean, variance, skewness, kurtosis.

    Clinical relevance:
      - Variance: schizophrenia EEG shows elevated amplitude variance in frontal
        regions due to disorganised cortical excitability.
      - Skewness: asymmetry in EEG amplitude distribution differs between patients
        and controls — reflects non-linear dynamics of neural circuits.
      - Kurtosis: measures "tailedness" — higher kurtosis = more extreme amplitude
        events (bursting), lower kurtosis = more uniform signal. Schizophrenia shows
        altered kurtosis patterns, especially in gamma band.
      Reference: Akar et al. (2015); Sabeti et al. (2009)
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_channels = data.shape[1]

    all_features = []
    for epoch in data:
        # epoch shape: (n_channels, n_times)
        feat_mean = epoch.mean(axis=1)           # (n_channels,)
        feat_var = epoch.var(axis=1)              # (n_channels,)
        feat_skew = skew(epoch, axis=1)           # (n_channels,)
        feat_kurt = kurtosis(epoch, axis=1)       # (n_channels,)
        all_features.append(np.concatenate([feat_mean, feat_var, feat_skew, feat_kurt]))

    features = np.array(all_features)  # (n_epochs, n_channels * 4)

    ch_names = [f"CH{i+1}" for i in range(n_channels)]
    names = (
        [f"mean_{ch}" for ch in ch_names] +
        [f"variance_{ch}" for ch in ch_names] +
        [f"skewness_{ch}" for ch in ch_names] +
        [f"kurtosis_{ch}" for ch in ch_names]
    )
    return features, names


def compute_hjorth_parameters(epochs: mne.Epochs) -> tuple[np.ndarray, list[str]]:
    """
    Compute Hjorth parameters per channel: Activity, Mobility, Complexity.

    Hjorth (1970) parameters capture signal characteristics in 3 numbers:
      - Activity:   variance of the signal — total power proxy
      - Mobility:   std of first derivative / std of signal — dominant frequency proxy
      - Complexity: mobility of first derivative / mobility of signal — bandwidth proxy

    Clinical relevance:
      - Schizophrenia shows REDUCED Hjorth complexity in prefrontal EEG, reflecting
        disorganised signal structure (reduced information content).
      - Mobility changes correlate with altered dominant frequency (theta/alpha shift).
      - Activity captures amplitude differences between patients and controls.
      Reference: Sabeti et al. (2009); Akar et al. (2015)
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_channels = data.shape[1]

    all_features = []
    for epoch in data:
        # epoch shape: (n_channels, n_times)
        # First derivative
        d1 = np.diff(epoch, axis=1)
        # Second derivative
        d2 = np.diff(d1, axis=1)

        # Activity = variance of signal
        activity = epoch.var(axis=1)  # (n_channels,)

        # Mobility = std(d1) / std(signal)
        mobility = d1.std(axis=1) / (epoch.std(axis=1) + 1e-12)

        # Complexity = mobility(d1) / mobility(signal)
        mob_d1 = d2.std(axis=1) / (d1.std(axis=1) + 1e-12)
        complexity = mob_d1 / (mobility + 1e-12)

        all_features.append(np.concatenate([activity, mobility, complexity]))

    features = np.array(all_features)  # (n_epochs, n_channels * 3)

    ch_names = [f"CH{i+1}" for i in range(n_channels)]
    names = (
        [f"hjorth_activity_{ch}" for ch in ch_names] +
        [f"hjorth_mobility_{ch}" for ch in ch_names] +
        [f"hjorth_complexity_{ch}" for ch in ch_names]
    )
    return features, names


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER FEATURE COMBINER
# ═══════════════════════════════════════════════════════════════════════════════

def extract_all_features(
    epochs: mne.Epochs,
    aggregate: bool = True
) -> tuple[np.ndarray, list[str]]:
    """
    Combine all biomarkers into a single feature matrix.

    Feature groups (all per-channel, per-epoch):
      1. Band powers (absolute + relative + differential entropy) → 5 bands × 3 × n_ch
      2. Spectral entropy                                        → n_ch
      3. Alpha/Theta ratio                                       → n_ch
      4. Gamma/Theta ratio                                       → n_ch
      5. Statistical (mean, var, skew, kurtosis)                 → 4 × n_ch
      6. Hjorth (activity, mobility, complexity)                 → 3 × n_ch

    For 19 channels: 5×3×19 + 19 + 19 + 19 + 4×19 + 3×19 = 285+19+19+19+76+57 = 475 features

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
    stat_feats, stat_names     = compute_statistical_features(epochs)
    hjorth_feats, hjorth_names = compute_hjorth_parameters(epochs)

    X = np.concatenate([
        band_feats, ent_feats, at_feats, gt_feats, stat_feats, hjorth_feats
    ], axis=1)
    names = band_names + ent_names + at_names + gt_names + stat_names + hjorth_names

    if aggregate:
        X = X.mean(axis=0, keepdims=True)  # average across epochs → (1, n_features)

    logger.info(
        f"Extracted {X.shape[1]} features from {len(epochs)} epochs. "
        f"Output shape: {X.shape}"
    )
    return X, names
