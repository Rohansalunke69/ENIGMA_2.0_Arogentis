import { Link } from "react-router-dom";
import "./Landing.css";

export default function Landing() {
    return (
        <div className="landing-wrapper">
            {/* Top Left Logo */}
            <div className="landing-logo">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000" strokeWidth="2">
                    <path d="M12 2a7 7 0 0 1 7 7c0 2.5-1.5 4.5-3 6s-2 3-2 5h-4c0-2-0.5-3.5-2-5s-3-3.5-3-6a7 7 0 0 1 7-7z" />
                    <path d="M9 20h6" /><path d="M10 22h4" />
                </svg>
                <span>NeuroScan</span>
            </div>

            {/* Floating Top Nav (Center) */}
            <nav className="landing-nav">
                <Link to="/" className="active">Home</Link>
                <Link to="#">Conditions</Link>
                <Link to="#">Move Support</Link>
                <div className="nav-divider"></div>
                <button className="icon-btn">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                        <circle cx="12" cy="7" r="4"></circle>
                    </svg>
                </button>
                <Link to="/dashboard" className="signin-btn">SIGN IN</Link>
            </nav>

            {/* Main Center Content */}
            <div className="landing-hero">
                <h1 className="hero-bg-text">NEUROSCAN</h1>
                <div className="girl-container">
                    <img src="/20678a8a57b988929b2a568138552a3d.webp" alt="NeuroScan Silhouette" className="hero-girl-img" />
                </div>
            </div>

            {/* Mid-Left Vertical Nav */}
            <div className="vertical-nav">
                <div className="v-nav-line"></div>
                <button className="v-nav-item">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>
                </button>
                <button className="v-nav-item active">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59-3.41A2 2 0 1 0 14 8h2m-5.41 3.41A2 2 0 1 1 13 14H2m8.59-3.41A2 2 0 1 0 16 14h2M6.59 17.59A2 2 0 1 1 8 21H2m12.59-3.41A2 2 0 1 0 18 21h2" /></svg>
                </button>
                <button className="v-nav-item">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                </button>
                <button className="v-nav-item">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
                </button>
            </div>

            {/* Bottom-Left Value Prop */}
            <div className="bottom-left-prop">
                <p>TRUSTED GUIDANCE FOR<br />SAFER DAILY MOVEMENT</p>
            </div>

            {/* Bottom-Right Stat Card */}
            <div className="stat-card-overlay">
                <h2>93%</h2>
                <p>Reported greater confidence in daily activities.</p>
                <div className="stat-author">
                    <span>Marie Doe</span> | 10+ Years Experience
                </div>
            </div>

            {/* Scroll Down */}
            <div className="scroll-down">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="5" y="2" width="14" height="20" rx="7" ry="7"></rect>
                    <line x1="12" y1="6" x2="12" y2="10"></line>
                </svg>
                <span>Scroll Down</span>
            </div>
        </div>
    );
}
