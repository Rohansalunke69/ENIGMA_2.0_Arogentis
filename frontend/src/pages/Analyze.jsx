import { useLocation, Link, useNavigate } from "react-router-dom";
import { useState } from "react";
import Sidebar from "../components/Sidebar";
import "./Analyze.css";

/* ── 10-20 system electrode positions mapped to SVG coordinates ── */
const ELECTRODE_MAP = {
    "Fp1": { x: 155, y: 55, region: "Prefrontal Left", zone: "frontal" },
    "Fp2": { x: 245, y: 55, region: "Prefrontal Right", zone: "frontal" },
    "F7": { x: 85, y: 120, region: "Frontal Left", zone: "frontal" },
    "F3": { x: 150, y: 115, region: "Frontal Left", zone: "frontal" },
    "Fz": { x: 200, y: 105, region: "Frontal Midline", zone: "frontal" },
    "F4": { x: 250, y: 115, region: "Frontal Right", zone: "frontal" },
    "F8": { x: 315, y: 120, region: "Frontal Right", zone: "frontal" },
    "T3": { x: 55, y: 200, region: "Temporal Left", zone: "temporal" },
    "T7": { x: 55, y: 200, region: "Temporal Left", zone: "temporal" },
    "C3": { x: 140, y: 195, region: "Central Left", zone: "central" },
    "Cz": { x: 200, y: 185, region: "Central Midline", zone: "central" },
    "C4": { x: 260, y: 195, region: "Central Right", zone: "central" },
    "T4": { x: 345, y: 200, region: "Temporal Right", zone: "temporal" },
    "T8": { x: 345, y: 200, region: "Temporal Right", zone: "temporal" },
    "T5": { x: 85, y: 280, region: "Posterior Temporal Left", zone: "temporal" },
    "P7": { x: 85, y: 280, region: "Posterior Temporal Left", zone: "temporal" },
    "P3": { x: 150, y: 275, region: "Parietal Left", zone: "parietal" },
    "Pz": { x: 200, y: 265, region: "Parietal Midline", zone: "parietal" },
    "P4": { x: 250, y: 275, region: "Parietal Right", zone: "parietal" },
    "T6": { x: 315, y: 280, region: "Posterior Temporal Right", zone: "temporal" },
    "P8": { x: 315, y: 280, region: "Posterior Temporal Right", zone: "temporal" },
    "O1": { x: 165, y: 340, region: "Occipital Left", zone: "occipital" },
    "O2": { x: 235, y: 340, region: "Occipital Right", zone: "occipital" },
    "Oz": { x: 200, y: 350, region: "Occipital Midline", zone: "occipital" },
    // Also handle EDF-style names like "EEG Fp1" or "EEG Fp1-REF"
};

function resolveChannel(chName) {
    const clean = chName.replace(/^EEG\s*/i, "").replace(/-REF$/i, "").replace(/\./g, "").trim();
    return ELECTRODE_MAP[clean] || null;
}

/* ── Compute per-channel risk from SHAP features ── */
function getChannelRisks(topFeatures, channels) {
    const risks = {};
    if (!channels) return risks;
    channels.forEach(ch => { risks[ch] = 0; });

    if (topFeatures) {
        topFeatures.forEach(f => {
            const featureName = f.feature || "";
            channels.forEach(ch => {
                const clean = ch.replace(/^EEG\s*/i, "").replace(/-REF$/i, "").trim();
                if (featureName.includes(clean) || featureName.includes(ch)) {
                    risks[ch] += Math.abs(f.shap_value) * (f.direction === "increases_risk" ? 1 : 0.3);
                }
            });
        });
    }

    // normalize 0-1
    const max = Math.max(...Object.values(risks), 0.001);
    Object.keys(risks).forEach(ch => { risks[ch] = risks[ch] / max; });
    return risks;
}

function riskToColor(val) {
    if (val > 0.7) return "#ef4444"; // red
    if (val > 0.4) return "#f97316"; // orange
    if (val > 0.15) return "#f59e0b"; // amber
    return "#22c55e"; // green
}

function riskToLabel(val) {
    if (val > 0.7) return "High Risk";
    if (val > 0.4) return "Moderate";
    if (val > 0.15) return "Mild";
    return "Normal";
}

/* ── Brain zones summary ── */
function getZoneSummary(channelRisks, channels) {
    const zones = { frontal: [], temporal: [], central: [], parietal: [], occipital: [] };
    if (!channels) return zones;

    channels.forEach(ch => {
        const pos = resolveChannel(ch);
        if (pos) {
            zones[pos.zone]?.push({ name: ch, risk: channelRisks[ch] || 0 });
        }
    });
    return zones;
}

function zoneAvgRisk(channels) {
    if (!channels.length) return 0;
    return channels.reduce((s, c) => s + c.risk, 0) / channels.length;
}

/* ── Brain SVG Topography ── */
function BrainMap({ channels, channelRisks, onSelect, selected }) {
    return (
        <svg viewBox="0 0 400 400" className="brain-svg">
            {/* Head outline */}
            <ellipse cx="200" cy="200" rx="185" ry="190" fill="#f8fbff" stroke="#cbd5e1" strokeWidth="2" />
            {/* Nose indicator */}
            <path d="M200 8 L192 30 L208 30 Z" fill="#e2e8f0" />
            {/* Ear indicators */}
            <ellipse cx="10" cy="200" rx="8" ry="18" fill="none" stroke="#cbd5e1" strokeWidth="1.5" />
            <ellipse cx="390" cy="200" rx="8" ry="18" fill="none" stroke="#cbd5e1" strokeWidth="1.5" />

            {/* Zone regions (subtle background) */}
            <ellipse cx="200" cy="90" rx="90" ry="55" fill="#dbeafe" opacity="0.2" />
            <ellipse cx="200" cy="190" rx="75" ry="45" fill="#d1fae5" opacity="0.15" />
            <ellipse cx="200" cy="280" rx="80" ry="45" fill="#fef3c7" opacity="0.15" />
            <ellipse cx="200" cy="345" rx="50" ry="30" fill="#fce7f3" opacity="0.15" />

            {/* Zone labels */}
            <text x="200" y="70" textAnchor="middle" fontSize="10" fill="#94a3b8" fontWeight="600">FRONTAL</text>
            <text x="55" y="200" textAnchor="middle" fontSize="9" fill="#94a3b8" fontWeight="600" transform="rotate(-90 55 200)">TEMPORAL</text>
            <text x="200" y="188" textAnchor="middle" fontSize="10" fill="#94a3b8" fontWeight="600">CENTRAL</text>
            <text x="200" y="260" textAnchor="middle" fontSize="10" fill="#94a3b8" fontWeight="600">PARIETAL</text>
            <text x="200" y="360" textAnchor="middle" fontSize="10" fill="#94a3b8" fontWeight="600">OCCIPITAL</text>

            {/* Channels */}
            {channels?.map(ch => {
                const pos = resolveChannel(ch);
                if (!pos) return null;
                const risk = channelRisks[ch] || 0;
                const color = riskToColor(risk);
                const isSelected = selected === ch;
                const r = isSelected ? 16 : 12;
                return (
                    <g key={ch} onClick={() => onSelect(ch)} style={{ cursor: "pointer" }}>
                        {/* Glow */}
                        {risk > 0.4 && (
                            <circle cx={pos.x} cy={pos.y} r={r + 6} fill={color} opacity={0.15} />
                        )}
                        {/* Dot */}
                        <circle cx={pos.x} cy={pos.y} r={r} fill={color} opacity={0.85}
                            stroke={isSelected ? "#0f172a" : "#fff"} strokeWidth={isSelected ? 3 : 2}
                            className="electrode-dot" />
                        {/* Label */}
                        <text x={pos.x} y={pos.y + 4} textAnchor="middle" fontSize="8" fill="#fff" fontWeight="700" style={{ pointerEvents: "none" }}>
                            {ch.replace(/^EEG\s*/i, "").replace(/-REF$/i, "").slice(0, 3)}
                        </text>
                    </g>
                );
            })}
        </svg>
    );
}

/* ── Zone Card ── */
function ZoneCard({ name, channels, icon }) {
    const avg = zoneAvgRisk(channels);
    const color = riskToColor(avg);
    const label = riskToLabel(avg);

    return (
        <div className="zone-card">
            <div className="zone-header">
                <span className="zone-icon">{icon}</span>
                <div>
                    <h4>{name}</h4>
                    <span className="zone-status" style={{ color }}>{label}</span>
                </div>
            </div>
            <div className="zone-bar-track">
                <div className="zone-bar-fill" style={{ width: `${avg * 100}%`, background: color }} />
            </div>
            {channels.length > 0 ? (
                <div className="zone-channels">
                    {channels.map(c => (
                        <span key={c.name} className="zone-ch" style={{ borderColor: riskToColor(c.risk) }}>
                            {c.name.replace(/^EEG\s*/i, "").replace(/-REF$/i, "")}
                        </span>
                    ))}
                </div>
            ) : (
                <p className="zone-empty">No channels in this zone</p>
            )}
        </div>
    );
}

/* ── Main Analyze Page ── */
export default function Analyze() {
    const location = useLocation();
    const navigate = useNavigate();
    const result = location.state?.result;
    const [selectedCh, setSelectedCh] = useState(null);

    if (!result) {
        return (
            <div className="analyze-empty">
                <h2>No Analysis Data</h2>
                <p>Upload and analyze an EEG file from the Dashboard first.</p>
                <button onClick={() => navigate("/dashboard")} className="go-dash-btn">Go to Dashboard →</button>
            </div>
        );
    }

    const channelRisks = getChannelRisks(result.top_features, result.channel_names);
    const zoneSummary = getZoneSummary(channelRisks, result.channel_names);
    const selectedPos = selectedCh ? resolveChannel(selectedCh) : null;
    const riskPct = Math.round(result.risk_probability * 100);

    return (
        <div className="dash-layout">
            <Sidebar />
            <div className="analyze-page">
                {/* Header */}
                <header className="analyze-header">
                    <Link to="/dashboard" className="back-link">← Dashboard</Link>
                    <h1>Brain Region Analysis</h1>
                    <div className="analyze-risk-pill" style={{
                        background: riskPct >= 75 ? "#fef2f2" : riskPct >= 50 ? "#fff7ed" : riskPct >= 25 ? "#fffbeb" : "#f0fdf4",
                        color: riskPct >= 75 ? "#dc2626" : riskPct >= 50 ? "#ea580c" : riskPct >= 25 ? "#d97706" : "#16a34a",
                    }}>
                        Overall Risk: {riskPct}% — {result.risk_tier}
                    </div>
                </header>

                <div className="analyze-grid">
                    {/* Brain Map */}
                    <div className="analyze-card brain-map-card">
                        <div className="card-head">
                            <h3>🧠 Brain Topography</h3>
                            <p className="card-sub">Click any electrode to see details</p>
                        </div>
                        <BrainMap
                            channels={result.channel_names}
                            channelRisks={channelRisks}
                            onSelect={setSelectedCh}
                            selected={selectedCh}
                        />
                        {/* Legend */}
                        <div className="brain-legend">
                            <span><i style={{ background: "#22c55e" }}></i> Normal</span>
                            <span><i style={{ background: "#f59e0b" }}></i> Mild</span>
                            <span><i style={{ background: "#f97316" }}></i> Moderate</span>
                            <span><i style={{ background: "#ef4444" }}></i> High Risk</span>
                        </div>
                    </div>

                    {/* Selected Channel Detail */}
                    <div className="analyze-card detail-card">
                        {selectedCh && selectedPos ? (
                            <>
                                <div className="card-head">
                                    <h3>Channel: {selectedCh.replace(/^EEG\s*/i, "").replace(/-REF$/i, "")}</h3>
                                    <span className="detail-badge" style={{
                                        background: riskToColor(channelRisks[selectedCh]) + "18",
                                        color: riskToColor(channelRisks[selectedCh])
                                    }}>
                                        {riskToLabel(channelRisks[selectedCh])}
                                    </span>
                                </div>
                                <div className="detail-info">
                                    <div className="detail-row">
                                        <span className="detail-label">Brain Region</span>
                                        <span className="detail-val">{selectedPos.region}</span>
                                    </div>
                                    <div className="detail-row">
                                        <span className="detail-label">Zone</span>
                                        <span className="detail-val capitalize">{selectedPos.zone}</span>
                                    </div>
                                    <div className="detail-row">
                                        <span className="detail-label">Risk Score</span>
                                        <span className="detail-val" style={{ color: riskToColor(channelRisks[selectedCh]), fontWeight: 700 }}>
                                            {Math.round(channelRisks[selectedCh] * 100)}%
                                        </span>
                                    </div>
                                </div>
                                <div className="detail-section">
                                    <h4>Related Biomarkers</h4>
                                    <div className="related-features">
                                        {result.top_features?.filter(f => {
                                            const clean = selectedCh.replace(/^EEG\s*/i, "").replace(/-REF$/i, "").trim();
                                            return f.feature.includes(clean) || f.feature.includes(selectedCh);
                                        }).map((f, i) => (
                                            <div key={i} className="related-feature">
                                                <span className={`rf-dir ${f.direction === "increases_risk" ? "rf-up" : "rf-down"}`}>
                                                    {f.direction === "increases_risk" ? "↑" : "↓"}
                                                </span>
                                                <span className="rf-name">{f.feature}</span>
                                                <span className="rf-val">{f.shap_value.toFixed(4)}</span>
                                            </div>
                                        )) || <p className="zone-empty">No specific biomarkers for this channel</p>}
                                    </div>
                                </div>
                            </>
                        ) : (
                            <div className="detail-empty">
                                <div className="detail-empty-icon">👆</div>
                                <h3>Select a Channel</h3>
                                <p>Click on any electrode on the brain map to view detailed analysis for that brain region.</p>
                            </div>
                        )}
                    </div>

                    {/* Zone Summary Cards */}
                    <div className="analyze-card zones-card">
                        <div className="card-head">
                            <h3>Region Analysis</h3>
                        </div>
                        <div className="zones-grid">
                            <ZoneCard name="Frontal Lobe" channels={zoneSummary.frontal} icon="🔵" />
                            <ZoneCard name="Temporal Lobe" channels={zoneSummary.temporal} icon="🟡" />
                            <ZoneCard name="Central Region" channels={zoneSummary.central} icon="🟢" />
                            <ZoneCard name="Parietal Lobe" channels={zoneSummary.parietal} icon="🟠" />
                            <ZoneCard name="Occipital Lobe" channels={zoneSummary.occipital} icon="🟣" />
                        </div>
                    </div>

                    {/* Clinical Interpretation */}
                    <div className="analyze-card interpretation-card">
                        <div className="card-head">
                            <h3>📋 Clinical Interpretation</h3>
                        </div>
                        <p className="interp-text">{result.interpretation}</p>
                        <div className="interp-meta">
                            <span>📊 {result.n_epochs_analyzed} epochs</span>
                            <span>📡 {result.n_channels} channels</span>
                            <span>⚡ {result.sampling_rate} Hz</span>
                        </div>
                        <p className="interp-disclaimer">⚠️ This analysis is for research purposes only and should not be used as a medical diagnosis.</p>
                    </div>
                </div>
            </div>
        </div>
    );
}
