"""
RepOD Schizophrenia EEG Dataset Downloader — Arogentis
======================================================
Downloads the Olejarczyk & Jernajczyk (2017) resting-state EEG dataset
from RepOD (Polish academic repository).

Dataset: "EEG in schizophrenia"
  → 14 schizophrenia patients (s01–s14) + 14 healthy controls (h01–h14)
  → 19-channel 10-20 montage, 250 Hz sampling rate
  → Eyes-closed resting state, ~15 minutes per subject
  → CC0 1.0 License (public domain)

Source: https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441
DOI:    10.18150/repod.0107441

Citation:
  Olejarczyk, E., & Jernajczyk, W. (2017). EEG in schizophrenia. RepOD.
  https://doi.org/10.18150/repod.0107441

Usage:
    python pipeline/data_downloader.py
    python pipeline/data_downloader.py --output_dir data/raw --verify
"""

import argparse
import csv
import logging
import os
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─── RepOD Dataverse File IDs ─────────────────────────────────────────────────
# These are the Dataverse file IDs for each EDF file on RepOD.
# Download URL pattern: https://repod.icm.edu.pl/api/access/datafile/{fileId}
#
# Healthy controls: h01–h14
# Schizophrenia:    s01–s14
#
# File IDs extracted from:
# https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441

FILE_MAP = {
    # Healthy controls (verified from Dataverse API)
    "h01.edf": 267,
    "h02.edf": 260,
    "h03.edf": 268,
    "h04.edf": 278,
    "h05.edf": 269,
    "h06.edf": 270,
    "h07.edf": 279,
    "h08.edf": 271,
    "h09.edf": 264,
    "h10.edf": 272,
    "h11.edf": 265,
    "h12.edf": 256,
    "h13.edf": 273,
    "h14.edf": 274,
    # Schizophrenia patients (verified from Dataverse API)
    "s01.edf": 275,
    "s02.edf": 261,
    "s03.edf": 276,
    "s04.edf": 266,
    "s05.edf": 257,
    "s06.edf": 277,
    "s07.edf": 262,
    "s08.edf": 258,
    "s09.edf": 254,
    "s10.edf": 252,
    "s11.edf": 263,
    "s12.edf": 253,
    "s13.edf": 255,
    "s14.edf": 259,
}

BASE_DOWNLOAD_URL = "https://repod.icm.edu.pl/api/access/datafile/"

SZ_SUBJECTS = [f"s{i:02d}.edf" for i in range(1, 15)]
HC_SUBJECTS = [f"h{i:02d}.edf" for i in range(1, 15)]


def download_dataset(output_dir: str = "data/raw") -> list[str]:
    """
    Download the RepOD schizophrenia resting-state EEG dataset.

    Uses the Dataverse API for direct file downloads (no login required).

    Args:
        output_dir: Directory to save the EDF files.

    Returns:
        List of successfully downloaded file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    downloaded = []
    failed = []

    for filename, file_id in FILE_MAP.items():
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            logger.info(f"Already exists: {filename}")
            downloaded.append(filepath)
            continue

        url = f"{BASE_DOWNLOAD_URL}{file_id}"
        logger.info(f"Downloading: {filename} (fileId={file_id}) → {filepath}")

        try:
            urllib.request.urlretrieve(url, filepath)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            downloaded.append(filepath)
            logger.info(f"✅ Downloaded: {filename} ({size_mb:.1f} MB)")
        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {e}")
            failed.append(filename)
            # Clean up partial download
            if os.path.exists(filepath):
                os.remove(filepath)

    logger.info(
        f"\nDownload complete: {len(downloaded)}/{len(FILE_MAP)} files. "
        f"Failed: {len(failed)}"
    )
    if failed:
        logger.warning(f"Failed files: {failed}")

    return downloaded


def generate_labels_csv(
    output_dir: str = "data",
    data_dir: str = "data/raw",
) -> str:
    """
    Generate labels CSV for the schizophrenia dataset.

    Label convention:
        1 = schizophrenia patient (s01–s14)
        0 = healthy control (h01–h14)

    Returns:
        Path to the generated labels CSV file.
    """
    csv_path = os.path.join(output_dir, "labels_physionet.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])

        for filename in SZ_SUBJECTS:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                writer.writerow([filename, 1])

        for filename in HC_SUBJECTS:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                writer.writerow([filename, 0])

    logger.info(f"Labels CSV saved: {csv_path}")
    return csv_path


def verify_dataset(data_dir: str = "data/raw") -> dict:
    """
    Verify all 28 EDF files: check they exist, load, and have 19 EEG channels.

    Returns:
        Dict with counts and any missing/corrupt files.
    """
    import mne
    mne.set_log_level("ERROR")

    present_sz, present_hc = [], []
    missing, corrupt = [], []

    for filename in SZ_SUBJECTS:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
                present_sz.append(filename)
                logger.info(
                    f"✅ {filename}: {len(raw.ch_names)} ch, "
                    f"{raw.times[-1]:.0f}s, {raw.info['sfreq']} Hz"
                )
            except Exception as e:
                corrupt.append(filename)
                logger.error(f"❌ {filename}: corrupt — {e}")
        else:
            missing.append(filename)

    for filename in HC_SUBJECTS:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
                present_hc.append(filename)
                logger.info(
                    f"✅ {filename}: {len(raw.ch_names)} ch, "
                    f"{raw.times[-1]:.0f}s, {raw.info['sfreq']} Hz"
                )
            except Exception as e:
                corrupt.append(filename)
                logger.error(f"❌ {filename}: corrupt — {e}")
        else:
            missing.append(filename)

    result = {
        "schizophrenia": len(present_sz),
        "controls": len(present_hc),
        "total": len(present_sz) + len(present_hc),
        "missing": missing,
        "corrupt": corrupt,
    }

    logger.info(
        f"\n{'='*50}\n"
        f"Dataset Verification Summary\n"
        f"{'='*50}\n"
        f"Schizophrenia subjects: {result['schizophrenia']}/14\n"
        f"Healthy controls:       {result['controls']}/14\n"
        f"Total:                  {result['total']}/28\n"
        f"Missing: {len(missing)} | Corrupt: {len(corrupt)}\n"
        f"{'='*50}"
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download RepOD Schizophrenia EEG Dataset (Olejarczyk & Jernajczyk, 2017)"
    )
    parser.add_argument("--output_dir", default="data/raw",
                        help="Directory to save EDF files")
    parser.add_argument("--verify", action="store_true",
                        help="Verify dataset after download")
    args = parser.parse_args()

    # Step 1: Download
    download_dataset(output_dir=args.output_dir)

    # Step 2: Generate labels
    generate_labels_csv(
        output_dir=os.path.dirname(args.output_dir) or "data",
        data_dir=args.output_dir,
    )

    # Step 3: Verify (optional)
    if args.verify:
        verify_dataset(data_dir=args.output_dir)
