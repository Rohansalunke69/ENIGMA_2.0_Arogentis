import { useState, useRef, useCallback } from "react";
import { Link } from "react-router-dom";
import { analyzeEEG, checkHealth } from "../api";
import "./Dashboard.css";

export default function Dashboard() {
    const [file, setFile] = useState(null);
    const [dragging, setDragging] = useState(false);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const inputRef = useRef();

    const handleFile = (f) => {
        if (f) {
            setFile(f);
            setResult(null);
            setError(null);
        }
    };

    const onDrop = useCallback((e) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files[0];
        handleFile(f);
    }, []);

    const onDragOver = (e) => { e.preventDefault(); setDragging(true); };
    const onDragLeave = () => setDragging(false);

    const handleAnalyze = async () => {
        if (!file) return;
        setLoading(true);
        setError(null);
        try {
            const data = await analyzeEEG(file);
            setResult(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const riskPercent = result ? Math.round(result.risk_probability * 100) : 0;

    return (
        <div className="dashboard">
            <nav className="dash-nav">
                <Link to="/" className="back-link">← Back to Home</Link>
                <h2>NeuroScan Dashboard</h2>
                <div style={{ width: 120 }}></div>
            </nav>

            {/* Upload Section */}
            <section className="upload-section">
                <div
                    className={`drop-zone ${dragging ? "drag-over" : ""} ${file ? "has-file" : ""}`}
                    onDrop={onDrop}
                    onDragOver={onDragOver}
                    onDragLeave={onDragLeave}
                    onClick={() => inputRef.current?.click()}
                >
                    <input
                        ref={inputRef}
                        type="file"
                        accept=".edf,.fif"
                        hidden
                        onChange={(e) => handleFile(e.target.files[0])}
                    />
                    {file ? (
                        <div className="file-info">
                            <span className="file-icon">📄</span>
                            <span className="file-name">{file.name}</span>
                            <span className="file-size">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                        </div>
                    ) : (
                        <>
                            <div className="upload-icon">
                                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#0ea5e9" strokeWidth="1.5">
                                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                                    <polyline points="17 8 12 3 7 8" />
                                    <line x1="12" y1="3" x2="12" y2="15" />
                                </svg>
                            </div>
                            <p className="drop-text">Drag & drop your EEG file here</p>
                            <p className="drop-hint">or click to browse (.edf, .fif)</p>
                        </>
                    )}
                </div>

                <button
                    className="analyze-btn"
                    onClick={handleAnalyze}
                    disabled={!file || loading}
                >
                    {loading ? (
                        <span className="spinner"></span>
                    ) : (
                        "Analyze EEG"
                    )}
                </button>
            </section>

            {/* Error */}
            {error && (
                <div className="error-box">
                    <strong>⚠️ Error:</strong> {error}
                </div>
            )}

            {/* Results */}
            {result && (
                <section className="results" key={Date.now()}>
                    {/* Risk Score Card */}
                    <div className="result-card risk-card">
                        <h3>Risk Assessment</h3>
                        <div className="gauge-container">
                            <svg viewBox="0 0 120 120" className="gauge">
                                <circle cx="60" cy="60" r="52" fill="none" stroke="#e0f2fe" strokeWidth="10" />
                                <circle
                                    cx="60" cy="60" r="52"
                                    fill="none"
                                    stroke={result.tier_color}
                                    strokeWidth="10"
                                    strokeDasharray={`${riskPercent * 3.27} 327`}
                                    strokeDashoffset="0"
                                    strokeLinecap="round"
                                    transform="rotate(-90 60 60)"
                                    className="gauge-fill"
                                />
                            </svg>
                            <div className="gauge-text">
                                <span className="gauge-pct">{riskPercent}%</span>
                                <span className="gauge-label">Risk</span>
                            </div>
                        </div>
                        <div className="risk-tier" style={{ background: result.tier_color + "20", color: result.tier_color, borderColor: result.tier_color }}>
                            {result.risk_tier}
                        </div>
                        <p className="interpretation">{result.interpretation}</p>
                    </div>

                    {/* SHAP Top Features */}
                    <div className="result-card">
                        <h3>Top Biomarkers (SHAP)</h3>
                        {result.top_features && result.top_features.length > 0 ? (
                            <div className="features-list">
                                {result.top_features.map((f, i) => (
                                    <div key={i} className="feature-row">
                                        <span className="feature-name" title={f.feature}>
                                            {f.feature.length > 25 ? f.feature.slice(0, 25) + "…" : f.feature}
                                        </span>
                                        <div className="feature-bar-container">
                                            <div
                                                className={`feature-bar ${f.direction === "increases_risk" ? "bar-risk" : "bar-safe"}`}
                                                style={{ width: `${Math.min(Math.abs(f.shap_value) * 300, 100)}%` }}
                                            ></div>
                                        </div>
                                        <span className={`feature-dir ${f.direction === "increases_risk" ? "dir-risk" : "dir-safe"}`}>
                                            {f.direction === "increases_risk" ? "↑" : "↓"}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="no-data">No SHAP data available</p>
                        )}
                    </div>

                    {/* Band Powers */}
                    <div className="result-card">
                        <h3>Band Powers</h3>
                        {result.band_powers_summary && Object.keys(result.band_powers_summary).length > 0 ? (
                            <div className="band-chart">
                                {Object.entries(result.band_powers_summary).map(([band, value]) => (
                                    <div key={band} className="band-bar-group">
                                        <div className="band-bar-wrapper">
                                            <div
                                                className="band-bar"
                                                style={{ height: `${Math.min(value * 100, 100)}%` }}
                                            ></div>
                                        </div>
                                        <span className="band-label">{band}</span>
                                        <span className="band-val">{typeof value === "number" ? value.toFixed(3) : value}</span>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="no-data">No band power data</p>
                        )}
                    </div>

                    {/* EEG Metadata */}
                    <div className="result-card meta-card">
                        <h3>EEG Recording Info</h3>
                        <div className="meta-grid">
                            <div className="meta-item">
                                <span className="meta-val">{result.n_channels}</span>
                                <span className="meta-label">Channels</span>
                            </div>
                            <div className="meta-item">
                                <span className="meta-val">{result.n_epochs_analyzed}</span>
                                <span className="meta-label">Epochs</span>
                            </div>
                            <div className="meta-item">
                                <span className="meta-val">{result.sampling_rate} Hz</span>
                                <span className="meta-label">Sampling Rate</span>
                            </div>
                        </div>
                        {result.channel_names && (
                            <div className="channel-list">
                                <strong>Channels:</strong>{" "}
                                <span>{result.channel_names.join(", ")}</span>
                            </div>
                        )}
                    </div>
                </section>
            )}
        </div>
    );
}
