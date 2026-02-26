"""
Page 1: Upload — Arogentis Dashboard
======================================
EEG file upload with format validation, preprocessing summary, and session state storage.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import compat  # noqa: F401 — patch NumPy 2.x before MNE loads

import streamlit as st
import numpy as np
import tempfile, shutil

from pipeline.preprocessing import run_preprocessing
from pipeline.feature_extraction import extract_all_features
from backend.services.model_service import ModelService

st.set_page_config(page_title="Upload EEG | Arogentis", page_icon="📁", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .upload-zone { background:#1e2130; border:2px dashed #6c63ff; border-radius:14px; padding:30px; text-align:center; }
    .info-badge { background:rgba(108,99,255,0.12); border:1px solid rgba(108,99,255,0.3); border-radius:8px; padding:10px 14px; color:#a78bfa; font-size:0.88rem; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📁 Upload EEG Recording")
st.caption("Supported formats: `.edf` (EDF/EDF+) and `.fif` (MNE/Neuromag)")

# ─── Model status warning ──────────────────────────────────────────────────────
ModelService.load()
if not ModelService.is_loaded():
    st.warning(
        "⚠️ **Model not trained yet.** Run `python train.py` first to generate model artifacts. "
        "You can still preview preprocessing results.",
        icon="⚠️"
    )

st.markdown('<div class="upload-zone">🧠 Drag and drop your EEG file here<br><span style="color:#64748b;font-size:0.85rem">.edf or .fif format | max 200 MB</span></div>', unsafe_allow_html=True)
st.markdown("")

uploaded_file = st.file_uploader(
    "Choose EEG file",
    type=["edf", "fif"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    st.success(f"✅ File received: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

    # ─── PSG / Sleep File Detection ────────────────────────────────────────────
    fname_upper = uploaded_file.name.upper()
    psg_indicators = ("PSG", "SLEEP", "SC4", "ST7", "SN", "HYP")
    if any(ind in fname_upper for ind in psg_indicators):
        st.warning(
            "⚠️ **This file appears to be a PSG / Sleep study recording.** "
            "This system is designed for **resting-state EEG only** (awake subjects, eyes closed). "
            "Sleep EEG has different physiology and is NOT suitable for schizophrenia detection. "
            "Results may be inaccurate or preprocessing may fail.",
            icon="⚠️"
        )

    with st.spinner("🔄 Running preprocessing pipeline..."):
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(uploaded_file, tmp)
            tmp_path = tmp.name

        try:
            epochs = run_preprocessing(tmp_path)
            n_epochs = len(epochs)
            sfreq = epochs.info["sfreq"]
            n_channels = len(epochs.ch_names)
            duration = (epochs.times[-1] * n_epochs) if n_epochs > 0 else 0.0

            st.markdown("### ✅ Preprocessing Complete")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Clean Epochs", n_epochs)
            c2.metric("Channels", n_channels)
            c3.metric("Sampling Rate", f"{sfreq:.0f} Hz")
            c4.metric("Total Duration", f"{duration:.1f} s")

            if n_epochs == 0:
                st.error(
                    "❌ **No clean epochs found.** All epochs were rejected during artifact removal. "
                    "This may happen if:\n"
                    "- The EEG file is a sleep/PSG recording with non-standard channels\n"
                    "- The signal amplitude is very large (>300µV)\n"
                    "- The recording is too short (<2 seconds)\n\n"
                    "Please try a standard resting-state EEG file (.edf or .fif)."
                )
            else:
                with st.spinner("🧮 Extracting biomarkers..."):
                    X, feature_names = extract_all_features(epochs, aggregate=True)

                # Store in session state for Analysis page
                st.session_state["features"] = X[0]
                st.session_state["feature_names"] = feature_names
                st.session_state["epochs_info"] = {
                    "n_epochs": n_epochs,
                    "n_channels": n_channels,
                    "sfreq": sfreq,
                    "channel_names": epochs.ch_names,
                    "filename": uploaded_file.name,
                }

                st.success(f"✅ **{len(feature_names)} biomarker features** extracted successfully!")
                st.info("👉 Navigate to **2. Analysis** in the sidebar to see the risk score.")

        except Exception as e:
            st.error(f"❌ Preprocessing failed: {e}")
        finally:
            os.unlink(tmp_path)

elif "features" in st.session_state:
    st.info(f"📋 Using previously uploaded: **{st.session_state['epochs_info']['filename']}**. Upload a new file to replace it.")
