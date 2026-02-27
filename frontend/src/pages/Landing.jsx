import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { checkHealth } from "../api";
import Brain3D from "../components/Brain3D";
import "./Landing.css";

export default function Landing() {
    const [health, setHealth] = useState(null);
    const [error, setError] = useState(false);

    useEffect(() => {
        checkHealth()
            .then((data) => setHealth(data))
            .catch(() => setError(true));
    }, []);

    return (
        <div className="landing">
            {/* ── Dark Hero Section ── */}
            <div className="hero-section">
                {/* EEG waveform background pattern */}
                <svg className="eeg-bg" viewBox="0 0 1200 200" preserveAspectRatio="none">
                    <path d="M0 100 Q50 60 100 100 T200 100 T300 100 T400 100 T500 100 T600 100 T700 100 T800 100 T900 100 T1000 100 T1100 100 T1200 100" fill="none" stroke="rgba(56,189,248,0.08)" strokeWidth="2" />
                    <path d="M0 120 Q75 50 150 120 T300 120 T450 120 T600 120 T750 120 T900 120 T1050 120 T1200 120" fill="none" stroke="rgba(14,165,233,0.06)" strokeWidth="1.5" />
                    <path d="M0 80 Q60 130 120 80 T240 80 T360 80 T480 80 T600 80 T720 80 T840 80 T960 80 T1080 80 T1200 80" fill="none" stroke="rgba(56,189,248,0.05)" strokeWidth="1" />
                </svg>

                {/* Navbar */}
                <nav className="navbar">
                    <div className="logo">
                        <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#0ea5e9" strokeWidth="2">
                            <path d="M12 2a7 7 0 0 1 7 7c0 2.5-1.5 4.5-3 6s-2 3-2 5h-4c0-2-0.5-3.5-2-5s-3-3.5-3-6a7 7 0 0 1 7-7z" />
                            <path d="M9 20h6" /><path d="M10 22h4" />
                        </svg>
                        <span>NeuroScan</span>
                    </div>
                    <div className={`status-badge ${health ? "online" : error ? "offline" : "loading"}`}>
                        <span className="status-dot"></span>
                        System Status: {health ? "ONLINE" : error ? "OFFLINE" : "..."}
                        {health && (
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
                            </svg>
                        )}
                    </div>
                </nav>

                {/* Hero Content */}
                <main className="hero">
                    <div className="hero-content">
                        <h1>AI-Powered EEG<br />Schizophrenia Screening</h1>
                        <p>
                            Instant, AI-driven risk assessments using ML biomarkers
                            and SHAP explainability. Optimized for .edf and .fif formats.
                        </p>
                        <div className="hero-actions">
                            <Link to="/dashboard" className="btn btn-primary">
                                Open Dashboard →
                            </Link>
                            <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="btn btn-outline">
                                API Docs
                            </a>
                        </div>
                        <p className="disclaimer">⚠ For research use only. Not a medical diagnostic tool.</p>
                    </div>

                    <div className="hero-visual">
                        <Brain3D />
                    </div>
                </main>
            </div>

            {/* ── Features Section ── */}
            <section className="features" id="features">
                <div className="feature-card">
                    <div className="feature-icon-wrap blue">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M12 20V10 M6 20V4 M18 20v-6" />
                        </svg>
                    </div>
                    <h3>EEG Analysis</h3>
                    <p>Upload .edf or .fif files for automated preprocessing and feature extraction.</p>
                </div>
                <div className="feature-card">
                    <div className="feature-icon-wrap pink">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M12 2a7 7 0 0 1 7 7c0 2.5-1.5 4.5-3 6s-2 3-2 5h-4c0-2-0.5-3.5-2-5s-3-3.5-3-6a7 7 0 0 1 7-7z" />
                            <path d="M9 20h6" /><path d="M10 22h4" />
                        </svg>
                    </div>
                    <h3>Risk Scoring</h3>
                    <p>ML-powered risk probability with Low, Moderate, High, and Critical tiers.</p>
                </div>
                <div className="feature-card">
                    <div className="feature-icon-wrap cyan">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
                        </svg>
                    </div>
                    <h3>Explainability</h3>
                    <p>SHAP-based biomarker attributions show which features drive the risk score.</p>
                </div>
            </section>
        </div>
    );
}
