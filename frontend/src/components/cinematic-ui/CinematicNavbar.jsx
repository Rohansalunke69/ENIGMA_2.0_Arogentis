import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { navbarReveal } from '../../design-system/animations';

/**
 * CinematicNavbar — Floating glassmorphism navbar
 * Matches the BioTrack video: logo + links + social icons + CTA
 */
export default function CinematicNavbar() {
  return (
    <motion.nav
      className="cin-navbar"
      variants={navbarReveal}
      initial="hidden"
      animate="visible"
    >
      {/* Logo */}
      <Link to="/" className="cin-navbar__logo">
        <div className="cin-navbar__logo-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2a7 7 0 0 1 7 7c0 2.5-1.5 4.5-3 6s-2 3-2 5h-4c0-2-0.5-3.5-2-5s-3-3.5-3-6a7 7 0 0 1 7-7z" />
            <path d="M9 20h6" /><path d="M10 22h4" />
          </svg>
        </div>
        NeuroScan
      </Link>

      {/* Navigation Links */}
      <ul className="cin-navbar__links">
        <li><Link to="/" className="cin-navbar__link cin-navbar__link--active">Home</Link></li>
        <li><a href="#features" className="cin-navbar__link">Features</a></li>
        <li><a href="#how-it-works" className="cin-navbar__link">How it Works</a></li>
        <li><Link to="/dashboard" className="cin-navbar__link">Dashboard</Link></li>
      </ul>

      {/* Right side: social + CTA */}
      <div className="cin-navbar__actions">
        <div className="cin-navbar__social">
          <button className="cin-navbar__social-icon" aria-label="GitHub">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
            </svg>
          </button>
          <button className="cin-navbar__social-icon" aria-label="Info">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>
            </svg>
          </button>
        </div>
        <Link to="/dashboard" className="cin-navbar__cta">
          Get Started
        </Link>
      </div>
    </motion.nav>
  );
}
