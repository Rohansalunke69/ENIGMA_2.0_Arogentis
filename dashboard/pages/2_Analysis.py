"""
Page 2: Analysis — Arogentis Dashboard
========================================
Displays risk score, risk tier gauge, band power breakdown, and epoch summary.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import compat  # noqa: F401 — patch NumPy 2.x before MNE loads

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from backend.services.model_service import ModelService

st.set_page_config(page_title="Analysis | Arogentis", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .risk-card { border-radius: 14px; padding: 28px; text-align: center; margin-bottom: 16px; }
    .interpretation-box { background:#1e2130; border-radius:12px; padding:18px 22px; border-left:4px solid #6c63ff; color:#e2e8f0; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📊 Risk Analysis Dashboard")

# ─── Check session state ──────────────────────────────────────────────────────
if "features" not in st.session_state:
    st.warning("⚠️ No EEG data loaded. Please go to **1. Upload** and upload an EEG file first.")
    st.stop()

features = st.session_state["features"]
feature_names = st.session_state["feature_names"]
info = st.session_state["epochs_info"]

ModelService.load()
if not ModelService.is_loaded():
    st.error("❌ Model not loaded. Run `python train.py` to train the model.")
    st.stop()

# ─── Run scoring ──────────────────────────────────────────────────────────────
with st.spinner("🔮 Computing risk score..."):
    result = ModelService.predict(features)

# ─── Risk Score Gauge ─────────────────────────────────────────────────────────
prob = result.risk_probability
color = result.tier_color

col_gauge, col_info = st.columns([1, 1.5])

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        title={"text": "Schizophrenia Risk Score", "font": {"size": 18, "color": "#e2e8f0"}},
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "#1e2130",
            "borderwidth": 2,
            "bordercolor": "#2d3748",
            "steps": [
                {"range": [0,  30], "color": "rgba(46,204,113,0.15)"},
                {"range": [30, 55], "color": "rgba(243,156,18,0.15)"},
                {"range": [55, 75], "color": "rgba(231,76,60,0.15)"},
                {"range": [75, 100], "color": "rgba(142,0,0,0.15)"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.8, "value": prob * 100},
        },
        domain={"x": [0, 1], "y": [0, 1]}
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#0f1117", font_color="#e2e8f0",
        height=320, margin=dict(t=30, b=10, l=20, r=20)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_info:
    st.markdown(f"### {result.risk_tier}")
    st.markdown(f'<div class="interpretation-box">{result.interpretation}</div>', unsafe_allow_html=True)
    st.markdown("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Epochs Analyzed", info["n_epochs"])
    c2.metric("Channels", info["n_channels"])
    c3.metric("Sampling Rate", f"{info['sfreq']:.0f} Hz")
    st.caption(f"📄 File: `{info['filename']}`")

st.divider()

# ─── Band Power Breakdown ─────────────────────────────────────────────────────
st.markdown("### 🌊 Frequency Band Power Summary")
st.caption("Mean absolute band power averaged across all channels (µV² /Hz)")

bands = ["delta", "theta", "alpha", "beta", "gamma"]
band_labels = ["δ Delta\n(0.5–4 Hz)", "θ Theta\n(4–8 Hz)", "α Alpha\n(8–13 Hz)", "β Beta\n(13–30 Hz)", "γ Gamma\n(30–45 Hz)"]
band_colors = ["#60a5fa", "#a78bfa", "#34d399", "#fbbf24", "#f87171"]

band_values = []
for band in bands:
    key = f"abs_{band}"
    idxs = [i for i, n in enumerate(feature_names) if n.startswith(key)]
    val = float(features[idxs].mean()) if idxs else 0.0
    band_values.append(val * 1e12)  # convert V²/Hz → µV²/Hz for readability

fig_bar = go.Figure()
for i, (label, val, color) in enumerate(zip(band_labels, band_values, band_colors)):
    fig_bar.add_trace(go.Bar(
        name=label.split("\n")[0],
        x=[label],
        y=[val],
        marker_color=color,
        marker_line_color="rgba(255,255,255,0.1)",
        marker_line_width=1,
    ))

fig_bar.update_layout(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#1e2130",
    font_color="#e2e8f0",
    showlegend=False,
    height=350,
    yaxis_title="Power (µV²/Hz)",
    xaxis_title="Frequency Band",
    margin=dict(t=20, b=20, l=40, r=20),
)
st.plotly_chart(fig_bar, use_container_width=True)

# ─── Key Derived Biomarkers ────────────────────────────────────────────────────
st.markdown("### 🔬 Derived Biomarker Ratios")
m1, m2, m3 = st.columns(3)

def get_ratio(feat_prefix):
    idxs = [i for i, n in enumerate(feature_names) if n.startswith(feat_prefix)]
    return float(features[idxs].mean()) if idxs else 0.0

at_ratio  = get_ratio("alpha_theta_ratio")
gt_ratio  = get_ratio("gamma_theta_ratio")
se_mean   = get_ratio("spectral_entropy")

m1.metric("Alpha/Theta Ratio", f"{at_ratio:.3f}", help="Reduced (<1.0) indicates cognitive disruption")
m2.metric("Gamma/Theta Ratio", f"{gt_ratio:.3f}", help="Reduced reflects NMDA hypofunction")
m3.metric("Mean Spectral Entropy", f"{se_mean:.3f}", help="Elevated (>2.5) indicates disorganised brain activity")
