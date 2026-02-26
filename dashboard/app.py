"""
Arogentis — EEG Schizophrenia Screening Dashboard
====================================================
Main Streamlit entrypoint.

Run with:
    streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compat  # noqa: F401 — patch NumPy 2.x before MNE loads

import streamlit as st

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Arogentis | EEG Schizophrenia Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6c63ff 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    .disclaimer-box {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 10px;
        padding: 14px 18px;
        color: #fca5a5;
        font-size: 0.85rem;
    }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d3748;
        text-align: center;
    }
    .nav-step {
        background: #1e2130;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 3px solid #6c63ff;
        margin-bottom: 8px;
        color: #e2e8f0;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Arogentis")
    st.caption("AI-Powered EEG Schizophrenia Screening")
    st.divider()
    st.markdown("### 🗺️ Navigation")
    st.markdown('<div class="nav-step">📁 <b>1. Upload</b> — Upload your EEG file</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-step">📊 <b>2. Analysis</b> — View risk score + band powers</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-step">🔬 <b>3. Explainability</b> — SHAP + Topomap</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**Dataset:** PhysioNet Schizophrenia EEG")
    st.markdown("**Model:** RandomForest + XGBoost")
    st.markdown("**XAI:** SHAP TreeExplainer")
    st.divider()
    st.markdown('<div class="disclaimer-box">⚠️ <b>Research use only.</b><br>Not a medical diagnostic tool. Results must be interpreted by a qualified clinician.</div>', unsafe_allow_html=True)

# ─── Hero ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 Arogentis</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Early Schizophrenia Risk Detection via Explainable EEG Analysis</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><h3 style="color:#6c63ff">5</h3><p style="color:#94a3b8;margin:0">Biomarker Types</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h3 style="color:#6c63ff">19</h3><p style="color:#94a3b8;margin:0">EEG Channels</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h3 style="color:#6c63ff">SHAP</h3><p style="color:#94a3b8;margin:0">Explainability</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><h3 style="color:#6c63ff">XGBoost</h3><p style="color:#94a3b8;margin:0">Core Model</p></div>', unsafe_allow_html=True)

st.divider()

st.markdown("""
### How It Works

| Step | Action | Technology |
|------|--------|------------|
| 1️⃣  | Upload raw EEG recording | `.edf` / `.fif` |
| 2️⃣  | Preprocessing | MNE: filter + epoch + artifact rejection |
| 3️⃣  | Feature extraction | Band power, spectral entropy, ratios |
| 4️⃣  | Risk scoring | RandomForest / XGBoost (probability 0–1) |
| 5️⃣  | Explainability | SHAP waterfall + MNE topomap |

**👈 Use the sidebar pages to begin analysis.**
""")
