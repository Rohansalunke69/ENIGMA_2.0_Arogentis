import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { checkHealth } from "../api";
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
            {/* Animated background */}
            <div className="bg-circles">
                <span></span><span></span><span></span><span></span><span></span>
            </div>

            <nav className="navbar">
                <div className="logo">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#0ea5e9" strokeWidth="2">
                        <path d="M12 2a7 7 0 0 1 7 7c0 2.5-1.5 4.5-3 6s-2 3-2 5h-4c0-2-0.5-3.5-2-5s-3-3.5-3-6a7 7 0 0 1 7-7z" />
                        <path d="M9 20h6" /><path d="M10 22h4" />
                    </svg>
                    <span>NeuroScan</span>
                </div>
                <div className="status-badge">
                    <span className={`dot ${health ? "online" : error ? "offline" : "loading"}`}></span>
                    {health ? "Backend Online" : error ? "Backend Offline" : "Connecting..."}
                </div>
            </nav>

            <main className="hero">
                <div className="hero-content">
                    <h1>AI-Powered EEG<br />Schizophrenia Screening</h1>
                    <p>
                        Upload EEG recordings and receive instant AI-driven risk assessments
                        powered by machine learning biomarkers and SHAP explainability.
                    </p>
                    <div className="hero-actions">
                        <Link to="/dashboard" className="btn btn-primary">
                            Open Dashboard →
                        </Link>
                        <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="btn btn-outline">
                            API Docs
                        </a>
                    </div>
                    <p className="disclaimer">⚠️ For research use only. Not a medical diagnostic tool.</p>
                </div>

                <div className="hero-visual">
                    <div className="brain-icon">
                        <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="50" cy="50" r="45" stroke="#0ea5e9" strokeWidth="1.5" opacity="0.3" />
                            <circle cx="50" cy="50" r="35" stroke="#0ea5e9" strokeWidth="1" opacity="0.2" />
                            <circle cx="50" cy="50" r="25" stroke="#0ea5e9" strokeWidth="0.8" opacity="0.15" />
                            <path d="M50 10 C30 25, 20 40, 25 55 C28 65, 35 70, 40 75 C42 78, 45 82, 50 85" stroke="#0ea5e9" strokeWidth="2" opacity="0.6" />
                            <path d="M50 10 C70 25, 80 40, 75 55 C72 65, 65 70, 60 75 C58 78, 55 82, 50 85" stroke="#38bdf8" strokeWidth="2" opacity="0.6" />
                            <circle cx="50" cy="30" r="3" fill="#0ea5e9" opacity="0.8" />
                            <circle cx="35" cy="45" r="2.5" fill="#38bdf8" opacity="0.7" />
                            <circle cx="65" cy="45" r="2.5" fill="#38bdf8" opacity="0.7" />
                            <circle cx="40" cy="60" r="2" fill="#0ea5e9" opacity="0.6" />
                            <circle cx="60" cy="60" r="2" fill="#0ea5e9" opacity="0.6" />
                            <circle cx="50" cy="72" r="2" fill="#38bdf8" opacity="0.5" />
                            <line x1="50" y1="30" x2="35" y2="45" stroke="#0ea5e9" strokeWidth="0.8" opacity="0.4" />
                            <line x1="50" y1="30" x2="65" y2="45" stroke="#0ea5e9" strokeWidth="0.8" opacity="0.4" />
                            <line x1="35" y1="45" x2="40" y2="60" stroke="#38bdf8" strokeWidth="0.8" opacity="0.3" />
                            <line x1="65" y1="45" x2="60" y2="60" stroke="#38bdf8" strokeWidth="0.8" opacity="0.3" />
                            <line x1="40" y1="60" x2="50" y2="72" stroke="#0ea5e9" strokeWidth="0.8" opacity="0.3" />
                            <line x1="60" y1="60" x2="50" y2="72" stroke="#0ea5e9" strokeWidth="0.8" opacity="0.3" />
                        </svg>
                    </div>
                </div>
            </main>

            <section className="features">
                <div className="feature-card">
                    <div className="feature-icon">📊</div>
                    <h3>EEG Analysis</h3>
                    <p>Upload .edf or .fif files for automated preprocessing and feature extraction.</p>
                </div>
                <div className="feature-card">
                    <div className="feature-icon">🧠</div>
                    <h3>Risk Scoring</h3>
                    <p>ML-powered risk probability with Low, Moderate, High, and Critical tiers.</p>
                </div>
                <div className="feature-card">
                    <div className="feature-icon">🔍</div>
                    <h3>Explainability</h3>
                    <p>SHAP-based biomarker attributions show which features drive the risk score.</p>
                </div>
            </section>
        </div>
    );
}
