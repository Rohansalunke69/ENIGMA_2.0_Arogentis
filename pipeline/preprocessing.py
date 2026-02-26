"""
EEG Preprocessing Pipeline — Arogentis
======================================
Steps: Load → Bandpass Filter → Notch Filter → Average Reference → Epoch → Artifact Rejection

WHY EACH STEP:
  - Bandpass (0.5–100 Hz): removes DC drift (below 0.5) and high-frequency noise
  - Notch (50 Hz):          removes powerline interference (Indian/EU standard)
  - Average reference:      equalizes electrode potential, removes common-mode noise
  - 2-second epochs:        enough samples for 0.5 Hz frequency resolution via Welch
  - 150 µV rejection:       clinical threshold to exclude eye blinks / EMG artifacts
"""

import logging
import mne
import numpy as np

mne.set_log_level("WARNING")  # suppress MNE verbosity in production
logger = logging.getLogger(__name__)


def load_raw_eeg(filepath: str) -> mne.io.Raw:
    """
    Load raw EEG from .edf or .fif file.

    Args:
        filepath: Absolute path to the EEG file.

    Returns:
        mne.io.Raw instance with data preloaded into memory.

    Raises:
        ValueError: If file format is not supported.
        FileNotFoundError: If the file does not exist.
    """
    filepath = str(filepath)
    if filepath.endswith(".edf"):
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    elif filepath.endswith(".fif"):
        raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported file format. Expected .edf or .fif, got: {filepath}")

    logger.info(
        f"Loaded EEG: {filepath} | "
        f"Channels: {len(raw.ch_names)} | "
        f"Duration: {raw.times[-1]:.1f}s | "
        f"Sampling rate: {raw.info['sfreq']} Hz"
    )
    return raw


def preprocess(raw: mne.io.Raw, notch_freq: float = 50.0) -> mne.io.Raw:
    """
    Apply standard clinical EEG preprocessing.

    Args:
        raw:        Loaded MNE Raw object.
        notch_freq: Power line interference frequency. 50 Hz (India/EU), 60 Hz (US).

    Returns:
        Preprocessed MNE Raw object.
    """
    logger.info("Applying bandpass filter (0.5–100 Hz)...")
    raw.filter(l_freq=0.5, h_freq=100.0, fir_design="firwin", verbose=False)

    logger.info(f"Applying notch filter ({notch_freq} Hz)...")
    raw.notch_filter(freqs=[notch_freq], verbose=False)

    logger.info("Setting average reference...")
    raw.set_eeg_reference("average", projection=False, verbose=False)

    return raw


def epoch_data(raw: mne.io.Raw, epoch_duration: float = 2.0) -> mne.Epochs:
    """
    Create fixed-length non-overlapping epochs.

    Why 2 seconds?
        - Nyquist: 2s gives 0.5 Hz frequency resolution in Welch PSD
        - Enough temporal coverage without losing stationarity assumption
        - Standard in schizophrenia EEG research (Olejarczyk & Jernajczyk, 2017)

    NOTE: We manually build the events array instead of using
    mne.make_fixed_length_events because that function triggers MNE's
    Numba JIT path which is incompatible with Python 3.14 + NumPy 2.x.

    Args:
        raw:            Preprocessed MNE Raw object.
        epoch_duration: Length of each epoch in seconds.

    Returns:
        MNE Epochs object.
    """
    sfreq = raw.info["sfreq"]
    n_epoch_samples = int(sfreq * epoch_duration)
    n_times = raw.n_times
    n_complete_epochs = n_times // n_epoch_samples

    # Build events array manually: (n_epochs, 3) — [sample, 0, event_id=1]
    event_samples = np.arange(n_complete_epochs) * n_epoch_samples
    events = np.column_stack([
        event_samples,
        np.zeros(n_complete_epochs, dtype=int),
        np.ones(n_complete_epochs, dtype=int),
    ])

    epochs = mne.Epochs(
        raw, events,
        tmin=0.0, tmax=epoch_duration,
        baseline=None,
        preload=True,
        verbose=False,
    )
    logger.info(f"Created {len(epochs)} epochs of {epoch_duration}s each.")
    return epochs


def reject_artifacts(epochs: mne.Epochs, peak_to_peak_uv: float = 150.0) -> mne.Epochs:
    """
    Reject epochs containing artifacts based on peak-to-peak amplitude.

    Clinical threshold: 150 µV is standard for resting-state wakefulness EEG.
    Values > 150 µV almost always indicate eye blinks, jaw movement, or cable artifacts.

    Args:
        epochs:           MNE Epochs object.
        peak_to_peak_uv:  Rejection threshold in microvolts.

    Returns:
        Clean MNE Epochs object with bad epochs removed.
    """
    threshold = peak_to_peak_uv * 1e-6  # convert µV → V (MNE internal unit)
    reject_criteria = dict(eeg=threshold)
    n_before = len(epochs)
    epochs.drop_bad(reject=reject_criteria)
    n_after = len(epochs)
    logger.info(
        f"Artifact rejection: {n_before - n_after}/{n_before} epochs rejected "
        f"({(n_before - n_after) / n_before * 100:.1f}% removed). "
        f"{n_after} clean epochs remaining."
    )
    return epochs


def run_preprocessing(
    filepath: str,
    notch_freq: float = 50.0,
    epoch_duration: float = 2.0,
    peak_to_peak_uv: float = 150.0,
) -> mne.Epochs:
    """
    Full preprocessing pipeline: Load → Filter → Epoch → Reject.

    Args:
        filepath:         Path to the raw EEG file (.edf or .fif).
        notch_freq:       Power line frequency in Hz.
        epoch_duration:   Duration of each epoch in seconds.
        peak_to_peak_uv:  Artifact rejection threshold in µV.

    Returns:
        Clean MNE Epochs object ready for feature extraction.
    """
    raw = load_raw_eeg(filepath)
    raw = preprocess(raw, notch_freq=notch_freq)
    epochs = epoch_data(raw, epoch_duration=epoch_duration)
    epochs = reject_artifacts(epochs, peak_to_peak_uv=peak_to_peak_uv)
    return epochs
