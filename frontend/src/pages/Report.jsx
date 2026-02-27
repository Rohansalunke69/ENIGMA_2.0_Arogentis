import { useLocation, useNavigate } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import "./Report.css";

export default function Report() {
    const location = useLocation();
    const navigate = useNavigate();
    const result = location.state?.result;

    if (!result) {
        return (
            <div className="report-empty">
                <h2>No Report Data</h2>
                <p>Analyze an EEG file first, then generate the report.</p>
                <button onClick={() => navigate("/dashboard")} className="go-dash-btn">Go to Dashboard →</button>
            </div>
        );
    }

    const riskPct = Math.round(result.risk_probability * 100);
    const riskColor = riskPct >= 75 ? "#ef4444" : riskPct >= 50 ? "#f97316" : riskPct >= 25 ? "#f59e0b" : "#22c55e";
    const now = new Date().toLocaleString();

    // Categorize features by direction
    const riskUp = result.top_features?.filter(f => f.direction === "increases_risk") || [];
    const riskDown = result.top_features?.filter(f => f.direction === "decreases_risk") || [];

    return (
        <div className="dash-layout">
            <Sidebar />
            <main className="report-main">
                {/* Report Header */}
                <div className="report-header-card">
                    <div className="report-title-row">
                        <div>
                            <h1>📋 Final Clinical Report</h1>
                            <p className="report-sub">NeuroScan EEG Schizophrenia Screening — Automated Summary</p>
                        </div>
                        <button className="print-btn" onClick={() => window.print()}>
                            🖨️ Print Report
                        </button>
                    </div>
                    <div className="report-meta-row">
                        <span>📅 Generated: {now}</span>
                        <span>📊 Epochs: {result.n_epochs_analyzed}</span>
                        <span>📡 Channels: {result.n_channels}</span>
                        <span>⚡ Sample Rate: {result.sampling_rate} Hz</span>
                    </div>
                </div>

                {/* Risk Assessment */}
                <div className="report-card">
                    <h2>1. Risk Assessment</h2>
                    <div className="risk-summary">
                        <div className="risk-big-score" style={{ background: riskColor + "12", borderColor: riskColor }}>
                            <span className="risk-number" style={{ color: riskColor }}>{riskPct}%</span>
                            <span className="risk-tier" style={{ color: riskColor }}>{result.risk_tier} Risk</span>
                        </div>
                        <div className="risk-text">
                            <p>{result.interpretation}</p>
                        </div>
                    </div>
                </div>

                {/* Key Findings */}
                <div className="report-card">
                    <h2>2. Key Biomarker Findings</h2>
                    <div className="findings-grid">
                        <div className="findings-col">
                            <h3 className="findings-heading risk-heading">⚠️ Risk-Increasing Factors</h3>
                            {riskUp.length > 0 ? riskUp.slice(0, 5).map((f, i) => (
                                <div key={i} className="finding-item risk-item">
                                    <span className="finding-rank">#{i + 1}</span>
                                    <span className="finding-name">{f.feature}</span>
                                    <span className="finding-val">SHAP: {f.shap_value.toFixed(4)}</span>
                                </div>
                            )) : <p className="no-findings">No risk-increasing factors detected.</p>}
                        </div>
                        <div className="findings-col">
                            <h3 className="findings-heading protect-heading">✅ Protective Factors</h3>
                            {riskDown.length > 0 ? riskDown.slice(0, 5).map((f, i) => (
                                <div key={i} className="finding-item protect-item">
                                    <span className="finding-rank">#{i + 1}</span>
                                    <span className="finding-name">{f.feature}</span>
                                    <span className="finding-val">SHAP: {f.shap_value.toFixed(4)}</span>
                                </div>
                            )) : <p className="no-findings">No protective factors detected.</p>}
                        </div>
                    </div>
                </div>

                {/* Spectral Analysis */}
                {result.band_powers_summary && Object.keys(result.band_powers_summary).length > 0 && (
                    <div className="report-card">
                        <h2>3. Spectral Band Power Analysis</h2>
                        <table className="band-table">
                            <thead>
                                <tr>
                                    <th>Band</th>
                                    <th>Power (µV²)</th>
                                    <th>Visual</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Object.entries(result.band_powers_summary).map(([band, val]) => {
                                    const max = Math.max(...Object.values(result.band_powers_summary), 0.001);
                                    return (
                                        <tr key={band}>
                                            <td className="band-name">{band}</td>
                                            <td className="band-val">{typeof val === "number" ? val.toFixed(4) : val}</td>
                                            <td className="band-bar-cell">
                                                <div className="band-bar-track">
                                                    <div className="band-bar-fill" style={{ width: `${(val / max) * 100}%` }} />
                                                </div>
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                )}

                {/* Channel Summary */}
                <div className="report-card">
                    <h2>4. EEG Channel Summary</h2>
                    <div className="channel-grid">
                        {result.channel_names?.map((ch) => (
                            <span key={ch} className="report-chip">{ch}</span>
                        ))}
                    </div>
                </div>

                {/* Disclaimer */}
                <div className="report-card disclaimer-card">
                    <h2>⚠️ Important Disclaimer</h2>
                    <p>
                        This report is generated by the NeuroScan AI system for <strong>research and screening purposes only</strong>.
                        It does not constitute a medical diagnosis. The risk assessment is based on machine learning models trained
                        on EEG biomarkers and should be interpreted by a qualified psychiatrist or neurologist. Clinical evaluation,
                        patient history, and additional diagnostic tests are necessary for any formal diagnosis.
                    </p>
                </div>
            </main>
        </div>
    );
}
