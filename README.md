# 🧠 ML — EEG-Based Schizophrenia Screening System

> Train ML schizophrenia risk detection using EEG frequency-domain biomarkers,
> explainable machine learning, and a Streamlit dashboard.
> **For research use only. Not a medical diagnostic tool.**

---

## 🏗️ Architecture

```
Raw EEG (.edf/.fif)
    ↓
Preprocessing (MNE)       filter → epoch → artifact rejection
    ↓
Feature Extraction         band power · entropy · alpha/theta · gamma/theta ratios
    ↓
ML Model                  RandomForest baseline → XGBoost advanced
    ↓
Risk Scorer               probability [0–1] + clinical tier
    ↓
SHAP Explainability       waterfall + topomap
    ↓
FastAPI backend  ←──────→ Streamlit Dashboard
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
cd ENIGMA_2.0_Arogentis
pip install -r requirements.txt
```

### 2. Train model (synthetic data — no real EEG needed)
```bash
python train.py --synthetic
```

### 3A. Launch Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

### 3B. Launch FastAPI Backend (alternative)
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

---

## 📊 EEG Biomarkers Extracted

| Biomarker | Frequency Range | Schizophrenia Relevance |
|-----------|----------------|------------------------|
| Delta power | 0.5–4 Hz | Elevated in prefrontal regions |
| Theta power | 4–8 Hz | Working memory coherence disrupted |
| **Alpha power** | 8–13 Hz | **Reduced = cortical hyperexcitability** |
| Beta power | 13–30 Hz | Desynchronisation pattern |
| **Gamma power** | 30–45 Hz | **Most significant — NMDA hypofunction** |
| Spectral Entropy | — | Higher = disorganised activity |
| Alpha/Theta ratio | — | Reduced = cognitive biomarker |
| Gamma/Theta ratio | — | NMDA receptor proxy |

---

## 📁 Project Structure

```
ENIGMA_2.0_Arogentis/
├── train.py                       # CLI training script
├── requirements.txt
├── data/                          # Raw EEG + processed features
├── pipeline/
│   ├── preprocessing.py           # MNE filter + epoch + rejection
│   ├── feature_extraction.py      # Band power, entropy, ratios
│   └── dataset_builder.py         # Raw → feature matrix
├── models/
│   ├── baseline_model.py          # RandomForest + SVM
│   ├── advanced_model.py          # XGBoost + tuning
│   ├── risk_scorer.py             # Probability scoring + tiers
│   └── evaluation.py              # ROC-AUC, CM, report
├── explainability/
│   ├── shap_explainer.py          # SHAP waterfall + bar plots
│   └── topomap_viz.py             # MNE brain topomap
├── backend/
│   ├── main.py                    # FastAPI app
│   ├── schemas.py                 # Pydantic models
│   ├── routers/eeg_router.py      # POST /analyze, GET /health
│   └── services/
│       ├── eeg_service.py         # Full pipeline orchestrator
│       └── model_service.py       # Singleton model loader
├── dashboard/
│   ├── app.py                     # Streamlit home
│   └── pages/
│       ├── 1_Upload.py            # EEG upload + preprocessing
│       ├── 2_Analysis.py          # Risk gauge + band powers
│       └── 3_Explainability.py    # SHAP + topomap
├── artifacts/                     # Saved model .pkl files
└── tests/
    └── test_preprocessing.py
```

---

## 🧬 Dataset (PhysioNet)

**Schizophrenia EEG Dataset** (Olejarczyk & Jernajczyk, 2017)
- URL: https://physionet.org/content/eeg-schizophrenia/1.0.0/
- 14 schizophrenia patients + 14 healthy controls
- 19-channel 10-20 system, 250 Hz sampling rate, `.edf` format

### Label CSV format
```csv
filename,label
subject01.edf,1
subject02.edf,0
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🔬 Risk Tiers

| Score | Tier | Action |
|-------|------|--------|
| 0.00 – 0.30 | 🟢 Low Risk | No action required |
| 0.30 – 0.55 | 🟡 Moderate Risk | Follow-up EEG in 3 months |
| 0.55 – 0.75 | 🔴 High Risk | Refer to psychiatrist |
| 0.75 – 1.00 | ⛔ Critical Risk | Immediate clinical assessment |

---

## ⚠️ Disclaimer

This system is a **research prototype** intended for academic and exploratory use.
It does not meet the regulatory requirements for clinical diagnosis in any jurisdiction.
All outputs should be reviewed by qualified neurologists and psychiatrists.

---

