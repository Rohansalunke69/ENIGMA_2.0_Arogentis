"""
Topomap Visualization — Arogentis
====================================
Renders MNE topographic head maps coloured by SHAP feature importance.

WHY TOPOMAPS?
  - Clinicians think spatially: "which brain region is affected?"
  - Topomap maps electrode-level SHAP values onto the 10-20 scalp layout
  - This converts abstract ML attribution into brain anatomy language
  - Essential for neuroscientist reviewers and clinical acceptance

The electrode-level importance is computed by averaging |SHAP values| across all
features that belong to each channel (delta, theta, alpha, beta, gamma, entropy, ratios).
"""

import io
import logging

import matplotlib.pyplot as plt
import mne
import numpy as np

logger = logging.getLogger(__name__)


def _channel_importance_from_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    n_channels: int,
) -> np.ndarray:
    """
    Aggregate |SHAP values| per channel by averaging across all features
    that contain the channel index in their name.

    Returns:
        importance: (n_channels,) array of mean absolute SHAP importance.
    """
    importance = np.zeros(n_channels)
    counts = np.zeros(n_channels)

    for i, name in enumerate(feature_names):
        for ch_idx in range(n_channels):
            ch_tag = f"CH{ch_idx + 1}"
            if ch_tag in name:
                importance[ch_idx] += abs(shap_values[i])
                counts[ch_idx] += 1

    with np.errstate(invalid="ignore"):
        importance = np.where(counts > 0, importance / counts, 0.0)

    return importance


def render_topomap(
    shap_values: np.ndarray,
    feature_names: list[str],
    channel_names: list[str],
    sfreq: float = 250.0,
    title: str = "Brain Region Importance (SHAP)",
    save_path: str = None,
) -> bytes | None:
    """
    Render a topomap of per-channel SHAP importance using MNE's 10-20 layout.

    Args:
        shap_values:   1D SHAP array (n_features,).
        feature_names: List of feature name strings.
        channel_names: EEG channel names in 10-20 system (e.g. ['Fp1','F3',...]).
        sfreq:         Sampling frequency (needed to create MNE Info).
        title:         Plot title.
        save_path:     If given, save PNG here. Otherwise return PNG bytes.

    Returns:
        PNG bytes if save_path is None, else None.
    """
    n_channels = len(channel_names)
    importance = _channel_importance_from_shap(shap_values, feature_names, n_channels)

    # Build minimal MNE info with standard montage
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, on_missing="ignore")
    except Exception as e:
        logger.warning(f"Could not set standard montage: {e}. Falling back to generic layout.")

    # Render topomap
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        im, _ = mne.viz.plot_topomap(
            importance,
            info,
            axes=ax,
            show=False,
            cmap="RdYlGn_r",   # red = high importance (problematic region), green = low
            vlim=(importance.min(), importance.max()),
            contours=4,
            sensors=True,
            names=channel_names,
        )
        plt.colorbar(im, ax=ax, label="Mean |SHAP|", shrink=0.7, pad=0.02)
    except Exception as e:
        logger.error(f"Topomap rendering failed: {e}")
        ax.text(0.5, 0.5, "Topomap unavailable\n(standard montage required)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
