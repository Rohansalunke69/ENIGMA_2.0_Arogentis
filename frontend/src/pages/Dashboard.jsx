import { useState, useRef, useCallback, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { analyzeEEG, checkHealth } from "../api";
import Sidebar from "../components/Sidebar";
import "./Dashboard.css";

/* ── Simple SVG-based mini charts (no external deps) ── */
function DonutChart({ value, color, size = 80 }) {
    const r = 32, circ = 2 * Math.PI * r;
    const filled = circ * value;
    return (
        <svg width={size} height={size} viewBox="0 0 80 80">
            <circle cx="40" cy="40" r={r} fill="none" stroke="#f1f5f9" strokeWidth="8" />
            <circle cx="40" cy="40" r={r} fill="none" stroke={color} strokeWidth="8"
                strokeDasharray={`${filled} ${circ}`} strokeLinecap="round"
                transform="rotate(-90 40 40)" className="donut-anim" />
            <text x="40" y="44" textAnchor="middle" fontSize="14" fontWeight="700" fill="#1e293b">
                {Math.round(value * 100)}%
            </text>
        </svg>
    );
}

function BarChart({ data, color }) {
    const max = Math.max(...Object.values(data), 0.001);
    return (
        <div className="mini-bar-chart">
            {Object.entries(data).map(([label, val]) => (
                <div key={label} className="bar-col">
                    <div className="bar-track">
                        <div className="bar-fill" style={{ height: `${(val / max) * 100}%`, background: color }} />
                    </div>
                    <span className="bar-lbl">{label}</span>
                </div>
            ))}
        </div>
    );
}



export default function Dashboard() {
    const [file, setFile] = useState(null);
    const [dragging, setDragging] = useState(false);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [health, setHealth] = useState(null);
    const inputRef = useRef();
    const navigate = useNavigate();

    useEffect(() => {
        checkHealth().then(setHealth).catch(() => { });
    }, []);

    const handleFile = (f) => { if (f) { setFile(f); setResult(null); setError(null); } };

    const onDrop = useCallback((e) => {
        e.preventDefault(); setDragging(false);
        handleFile(e.dataTransfer.files[0]);
    }, []);

    const handleAnalyze = async () => {
        if (!file) return;
        setLoading(true); setError(null);
        try {
            const data = await analyzeEEG(file);
            setResult(data);
        } catch (err) { setError(err.message); }
        finally { setLoading(false); }
    };

    const riskPct = result ? Math.round(result.risk_probability * 100) : 0;

    // Color based on percentage: >75% = RED, 50-75% = orange, 25-50% = amber, <25% = green
    const getRiskColor = (pct) => {
        if (pct >= 75) return "#ef4444"; // RED
        if (pct >= 50) return "#f97316"; // Orange
        if (pct >= 25) return "#f59e0b"; // Amber
        return "#22c55e";                // Green
    };
    const riskColor = result ? getRiskColor(riskPct) : "#0ea5e9";

    return (
        <div className="dash-layout">
            {/* ── Sidebar ── */}
            <Sidebar />

            {/* ── Main Content ── */}
            <main className={`dash-main ${loading ? "is-loading" : ""}`}>
                {/* AI Loading Overlay */}
                {loading && (
                    <div className="ai-loading-overlay">
                        <div className="ai-loading-content">
                            <div className="ai-brain-pulse">🧠</div>
                            <h2>AI is Analyzing...</h2>
                            <p>Please wait while the machine learning model extracts features and calculates SHAP values.</p>
                            <div className="ai-progress-bar">
                                <div className="ai-progress-fill"></div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Header */}
                <header className="dash-header">
                    <div>
                        <h1>EEG Analytics</h1>
                        <p className="header-sub">NeuroScan Schizophrenia Screening Dashboard</p>
                    </div>
                    <div className="header-right">
                        <div className={`status-pill ${health?.model_loaded ? "model-on" : "model-off"}`}>
                            <span className="status-dot"></span>
                            {health?.model_loaded ? "Model Ready" : "Model Not Loaded"}
                        </div>
                    </div>
                </header>

                {/* Stat Cards Row — show when we have results */}
                {result && (
                    <div className="stat-cards">
                        <div className="stat-card">
                            <div className="stat-icon" style={{ background: "#eff6ff" }}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
                            </div>
                            <div className="stat-info">
                                <span className="stat-label">Epochs Analyzed</span>
                                <span className="stat-value">{result.n_epochs_analyzed}</span>
                            </div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-icon" style={{ background: "#f0fdf4" }}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2"><circle cx="12" cy="12" r="10" /><path d="M12 6v6l4 2" /></svg>
                            </div>
                            <div className="stat-info">
                                <span className="stat-label">Channels</span>
                                <span className="stat-value">{result.n_channels}</span>
                            </div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-icon" style={{ background: "#fef3c7" }}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2"><path d="M12 20V10 M6 20V4 M18 20v-6" /></svg>
                            </div>
                            <div className="stat-info">
                                <span className="stat-label">Sampling Rate</span>
                                <span className="stat-value">{result.sampling_rate} Hz</span>
                            </div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-icon" style={{ background: riskColor + "18" }}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={riskColor} strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z M12 9v4 M12 17h.01" /></svg>
                            </div>
                            <div className="stat-info">
                                <span className="stat-label">Risk Level</span>
                                <span className="stat-value" style={{ color: riskColor }}>{result.risk_tier}</span>
                            </div>
                        </div>
                    </div>
                )}

                {/* Navigation Buttons */}
                {result && (
                    <div className="nav-buttons-row">
                        <button className="view-analysis-btn" onClick={() => navigate("/analyze", { state: { result } })}>
                            🧠 View Brain Analysis →
                        </button>
                        <button className="view-report-btn" onClick={() => navigate("/report", { state: { result } })}>
                            📋 Generate Final Report →
                        </button>
                    </div>
                )}

                {/* Upload + Charts Grid */}
                <div className={`content-grid ${result ? "has-results" : ""}`}>

                    {/* Upload Card */}
                    <div className="card upload-card">
                        <div className="card-head">
                            <h3>Upload EEG Recording</h3>
                            <span className="badge">.edf / .fif</span>
                        </div>
                        <div
                            className={`drop-zone ${dragging ? "drag-over" : ""} ${file ? "has-file" : ""}`}
                            onDrop={onDrop}
                            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                            onDragLeave={() => setDragging(false)}
                            onClick={() => inputRef.current?.click()}
                        >
                            <input ref={inputRef} type="file" accept=".edf,.fif" hidden onChange={(e) => handleFile(e.target.files[0])} />
                            {file ? (
                                <div className="file-info">
                                    <div className="file-thumb">📄</div>
                                    <div>
                                        <p className="file-name">{file.name}</p>
                                        <p className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                    </div>
                                </div>
                            ) : (
                                <div className="upload-placeholder">
                                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.5">
                                        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                                        <polyline points="17 8 12 3 7 8" />
                                        <line x1="12" y1="3" x2="12" y2="15" />
                                    </svg>
                                    <p>Drop your EEG file here</p>
                                    <span>or click to browse</span>
                                </div>
                            )}
                        </div>
                        <button className="analyze-btn" onClick={handleAnalyze} disabled={!file || loading}>
                            {loading ? <span className="spinner"></span> : "Analyze EEG →"}
                        </button>
                        {error && <div className="inline-error">⚠️ {error}</div>}
                    </div>

                    {/* Risk Donut Chart */}
                    {result && (
                        <div className="card risk-chart-card">
                            <div className="card-head">
                                <h3>Risk Score</h3>
                                <span className="tier-badge" style={{ background: riskColor + "18", color: riskColor }}>{result.risk_tier}</span>
                            </div>
                            <div className="donut-center">
                                <DonutChart value={result.risk_probability} color={riskColor} size={140} />
                            </div>
                            <p className="interpretation">{result.interpretation}</p>
                        </div>
                    )}

                    {/* Band Powers Bar Chart */}
                    {result && result.band_powers_summary && Object.keys(result.band_powers_summary).length > 0 && (
                        <div className="card band-card">
                            <div className="card-head">
                                <h3>Spectral Band Powers</h3>
                            </div>
                            <BarChart data={result.band_powers_summary} color="#0ea5e9" />
                        </div>
                    )}

                    {/* SHAP Features */}
                    {result && (
                        <div className="card shap-card">
                            <div className="card-head">
                                <h3>Top Biomarkers</h3>
                                <span className="badge">SHAP</span>
                            </div>
                            {result.top_features && result.top_features.length > 0 ? (
                                <div className="shap-list">
                                    {result.top_features.slice(0, 8).map((f, i) => (
                                        <div key={i} className="shap-row">
                                            <span className="shap-rank">#{i + 1}</span>
                                            <span className="shap-name" title={f.feature}>{f.feature.length > 28 ? f.feature.slice(0, 28) + "…" : f.feature}</span>
                                            <div className="shap-bar-track">
                                                <div
                                                    className={`shap-bar-fill ${f.direction === "increases_risk" ? "risk-up" : "risk-down"}`}
                                                    style={{ width: `${Math.min(Math.abs(f.shap_value) * 400, 100)}%` }}
                                                />
                                            </div>
                                            <span className={`shap-dir ${f.direction === "increases_risk" ? "dir-up" : "dir-down"}`}>
                                                {f.direction === "increases_risk" ? "↑" : "↓"}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <p className="empty-msg">No SHAP data available</p>
                            )}
                        </div>
                    )}

                    {/* Channel Info */}
                    {result && (
                        <div className="card channels-card">
                            <div className="card-head">
                                <h3>EEG Channels</h3>
                                <span className="badge">{result.n_channels} ch</span>
                            </div>
                            <div className="channel-chips">
                                {result.channel_names?.map((ch) => (
                                    <span key={ch} className="chip">{ch}</span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}
