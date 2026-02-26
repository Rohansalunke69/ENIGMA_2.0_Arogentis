"""
Page 3: Explainability — Arogentis Dashboard
==============================================
SHAP waterfall plot + MNE topomap of per-channel biomarker importance.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import compat  # noqa: F401 — patch NumPy 2.x before MNE loads

import streamlit as st
import numpy as np
from PIL import Image
import io

from backend.services.model_service import ModelService
from explainability.topomap_viz import render_topomap

st.set_page_config(page_title="Explainability | Arogentis", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .shap-legend { background:#1e2130; border-radius:10px; padding:14px 18px; color:#94a3b8; font-size:0.85rem; border:1px solid #2d3748; }
    .feature-row { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e2130; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🔬 Explainability — What Drove This Prediction?")

# ─── Guard ────────────────────────────────────────────────────────────────────
if "features" not in st.session_state:
    st.warning("⚠️ No EEG data loaded. Please go to **1. Upload** first.")
    st.stop()

features      = st.session_state["features"]
feature_names = st.session_state["feature_names"]
info          = st.session_state["epochs_info"]

ModelService.load()
if not ModelService.is_loaded():
    st.error("❌ Model not loaded. Run `python train.py` first.")
    st.stop()

# ─── SHAP Explanation ─────────────────────────────────────────────────────────
with st.spinner("🧮 Computing SHAP values..."):
    shap_result = ModelService.explain(features)

if "error" in shap_result and not shap_result.get("top_features"):
    st.error(f"SHAP unavailable: {shap_result['error']}")
    st.stop()

top_features  = shap_result.get("top_features", [])
shap_values   = np.array(shap_result.get("shap_values", []))

col_shap, col_topo = st.columns([1.3, 1])

# ─── SHAP Waterfall ───────────────────────────────────────────────────────────
with col_shap:
    st.markdown("### 📉 SHAP Waterfall — Top Biomarker Drivers")
    st.markdown('<div class="shap-legend">🔴 Red bars push risk <b>higher</b> (schizophrenia-like patterns)<br>🔵 Blue bars push risk <b>lower</b> (healthy pattern evidence)</div>', unsafe_allow_html=True)
    st.markdown("")

    waterfall_bytes = ModelService.shap_waterfall_bytes(features)
    if waterfall_bytes:
        st.image(Image.open(io.BytesIO(waterfall_bytes)), use_container_width=True)
    else:
        st.info("SHAP waterfall plot not available. Showing top features table.")

    # ─── Top features table ───────────────────────────────────────────────────
    st.markdown("#### Top 10 Contributing Biomarkers")
    if top_features:
        for feat in top_features[:10]:
            direction = "🔴 ↑" if feat["shap_value"] > 0 else "🔵 ↓"
            bar_width = min(abs(feat["shap_value"]) * 300, 100)
            color = "#ef4444" if feat["shap_value"] > 0 else "#3b82f6"
            st.markdown(
                f'<div class="feature-row" style="background:#1e2130;border-radius:6px;padding:8px 12px;margin-bottom:4px;">'
                f'<span style="color:#e2e8f0;font-size:0.85rem">{feat["feature"]}</span>'
                f'<span style="color:{color};font-weight:600">{direction} {feat["shap_value"]:+.4f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

# ─── Topomap ─────────────────────────────────────────────────────────────────
with col_topo:
    st.markdown("### 🧠 Brain Topomap — Spatial Importance")
    st.markdown("Colour intensity shows which scalp regions drove the prediction most strongly.")

    channel_names = info.get("channel_names", [])

    if len(shap_values) > 0 and len(channel_names) > 0:
        with st.spinner("Rendering topomap..."):
            topo_bytes = render_topomap(
                shap_values=shap_values,
                feature_names=feature_names,
                channel_names=channel_names,
                sfreq=info.get("sfreq", 250.0),
            )
        if topo_bytes:
            st.image(Image.open(io.BytesIO(topo_bytes)), use_container_width=True)
        else:
            st.info("Topomap rendering failed. Ensure channels follow standard 10-20 naming.")
    else:
        st.info("Topomap requires standard 10-20 channel names (e.g. Fp1, F3, Cz...).")

    st.markdown("")
    st.markdown("#### Biomarker Interpretation Guide")
    st.markdown("""
| Biomarker | High Value → | Clinical Meaning |
|-----------|-------------|-----------------|
| `abs_gamma` ↓ | Low gamma | NMDA hypofunction, psychosis marker |
| `abs_alpha` ↓ | Low alpha | Cortical hyperexcitability |
| `spectral_entropy` ↑ | High entropy | Disorganised neural activity |
| `alpha_theta_ratio` ↓ | Reduced ratio | Working memory impairment |
| `abs_delta` ↑ | High delta | Prefrontal slow-wave pathology |
""")

st.divider()
st.caption("⚠️ SHAP values are model attribution scores, not direct physiological measurements. Interpretation should be done by a qualified clinical neurophysiologist.")
