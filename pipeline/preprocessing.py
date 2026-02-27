"""
EEG Preprocessing Pipeline — Arogentis
======================================
Designed for RESTING-STATE EEG ONLY (awake subjects, eyes closed).
Steps: Load → EEG Channel Selection → Bandpass Filter → Notch Filter → Average Reference → Epoch → Artifact Rejection

WHY EACH STEP:
  - Bandpass (1–40 Hz): removes DC drift and slow movement artifact (below 1 Hz)
                        and high-frequency EMG/noise. 40 Hz cap is standard for
                        resting-state clinical EEG; avoids facial muscle contamination.
  - Notch (50 Hz):      removes powerline interference (Indian/EU/Polish standard)
  - Average reference:  equalizes electrode potential, removes common-mode noise
  - 2-second epochs:    gives 0.5 Hz frequency resolution via Welch PSD;
                        standard in schizophrenia EEG research (Olejarczyk & Jernajczyk, 2017)
  - 100 µV rejection:   correct clinical threshold for AWAKE resting-state EEG.
                        Values > 100 µV almost always indicate eye blinks or EMG.
                        (NOTE: sleep/PSG EEG uses 150-500 µV — this pipeline does NOT support PSG)
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

    # ── Channel Selection ─────────────────────────────────────────────────────
    # Strategy 1: pick channels officially typed as EEG
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude="bads")

    if len(eeg_picks) > 0:
        raw.pick(eeg_picks)
        logger.info(f"Picked {len(eeg_picks)} typed-EEG channels.")
    else:
        # Strategy 2: pick by name prefix (Sleep-EDF, TUAR, etc. label EEG channels as 'EEG Fpz-Cz' etc.)
        eeg_by_name = [ch for ch in raw.ch_names
                       if any(k in ch.upper() for k in ("EEG", "FP", "FZ", "CZ", "OZ", "PZ", "AF", "FC", "CP", "PO"))]
        if eeg_by_name:
            raw.pick(eeg_by_name)
            logger.info(f"Picked {len(eeg_by_name)} name-matched EEG channels: {eeg_by_name}")
        else:
            # Strategy 3: use everything — best effort
            logger.warning(f"No EEG channels identified. Using all {len(raw.ch_names)} channels as fallback.")

    logger.info(
        f"Loaded EEG: {filepath} | "
        f"Channels: {len(raw.ch_names)} | "
        f"Duration: {raw.times[-1]:.1f}s | "
        f"Sampling rate: {raw.info['sfreq']} Hz"
    )
    return raw


def preprocess(raw: mne.io.Raw, notch_freq: float = 50.0) -> mne.io.Raw:
    """
    Apply standard clinical EEG preprocessing for RESTING-STATE recordings.

    Bandpass: 1–40 Hz
      - 1 Hz high-pass: removes DC offset and slow drift artifacts
      - 40 Hz low-pass: removes EMG (facial muscle) and line noise harmonics
      - This is the clinical standard for resting-state schizophrenia EEG
        (Olejarczyk & Jernajczyk, 2017; Borisov et al., 2005)

    Args:
        raw:        Loaded MNE Raw object.
        notch_freq: Power line interference frequency. 50 Hz (India/EU/Poland), 60 Hz (US).

    Returns:
        Preprocessed MNE Raw object.
    """
    nyquist = raw.info["sfreq"] / 2.0

    # Resting-state standard: 1–40 Hz
    # Dynamically cap h_freq below Nyquist for low-sfreq files
    h_freq = min(40.0, nyquist - 1.0)
    l_freq = 1.0
    logger.info(f"Applying bandpass filter ({l_freq}–{h_freq} Hz) [Nyquist={nyquist} Hz]...")
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)

    # Only apply notch filter if notch frequency is safely below Nyquist
    if notch_freq < nyquist:
        logger.info(f"Applying notch filter ({notch_freq} Hz)...")
        raw.notch_filter(freqs=[notch_freq], verbose=False)
    else:
        logger.warning(f"Skipping notch filter: {notch_freq} Hz >= Nyquist {nyquist} Hz")

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


def reject_artifacts(epochs: mne.Epochs, peak_to_peak_uv: float = 100.0) -> mne.Epochs:
    """
    Reject epochs containing artifacts based on peak-to-peak amplitude.

    Threshold: 100 µV — correct for AWAKE resting-state EEG.
      - Eyes-closed resting EEG amplitude: typically 10–80 µV
      - Eye blinks artifact: 200–500 µV → correctly rejected at 100 µV
      - EMG artifact: often >100 µV → correctly rejected
      - Why NOT 150 µV or 500 µV: those are for sleep EEG which has large slow waves.
        For awake resting-state, 100 µV is the well-established clinical threshold.
        Reference: Kleiner et al., 2007; standard in EEGLAB/MNE pipelines.

    Args:
        epochs:           MNE Epochs object.
        peak_to_peak_uv:  Rejection threshold in microvolts (default 100 µV).

    Returns:
        Clean MNE Epochs object with bad epochs removed.
    """
    threshold = peak_to_peak_uv * 1e-6  # convert µV → V (MNE internal unit)

    # Build reject dict using actual EEG channel types present
    ch_types_present = set(epochs.get_channel_types())
    if "eeg" in ch_types_present:
        reject_criteria = {"eeg": threshold}
    else:
        # Unlabeled channels: skip rejection rather than reject everything
        logger.warning("No typed-EEG channels found; skipping artifact rejection.")
        reject_criteria = None

    n_before = len(epochs)
    if reject_criteria:
        epochs.drop_bad(reject=reject_criteria)
    n_after = len(epochs)
    pct = (n_before - n_after) / n_before * 100 if n_before > 0 else 0
    logger.info(
        f"Artifact rejection: {n_before - n_after}/{n_before} epochs rejected "
        f"({pct:.1f}% removed). {n_after} clean epochs remaining."
    )
    return epochs


def run_preprocessing(
    filepath: str,
    notch_freq: float = 50.0,
    epoch_duration: float = 2.0,
    peak_to_peak_uv: float = 100.0,
) -> mne.Epochs:
    """
    Full preprocessing pipeline for RESTING-STATE EEG: Load → Filter → Epoch → Reject.

    If the default threshold rejects all epochs, progressively relax it
    to handle EEG files from different sources and amplitude scales.

    Args:
        filepath:         Path to the raw EEG file (.edf or .fif).
        notch_freq:       Power line frequency in Hz (50 for India/EU/Poland, 60 for US).
        epoch_duration:   Duration of each epoch in seconds (2.0 is standard).
        peak_to_peak_uv:  Artifact rejection threshold in µV (100 µV for awake resting-state).

    Returns:
        Clean MNE Epochs object ready for feature extraction.
    """
    raw = load_raw_eeg(filepath)
    raw = preprocess(raw, notch_freq=notch_freq)
    epochs = epoch_data(raw, epoch_duration=epoch_duration)

    # Progressive threshold relaxation: try increasingly lenient thresholds
    thresholds = [peak_to_peak_uv, 200.0, 500.0, 1000.0]
    for thresh in thresholds:
        test_epochs = epochs.copy()
        test_epochs = reject_artifacts(test_epochs, peak_to_peak_uv=thresh)
        if len(test_epochs) > 0:
            if thresh > peak_to_peak_uv:
                logger.warning(
                    f"Default threshold ({peak_to_peak_uv} µV) rejected all epochs. "
                    f"Using relaxed threshold: {thresh} µV → {len(test_epochs)} clean epochs."
                )
            return test_epochs

    # If all thresholds reject everything, skip rejection entirely
    logger.warning("All thresholds rejected all epochs. Skipping artifact rejection entirely.")
    return epochs

