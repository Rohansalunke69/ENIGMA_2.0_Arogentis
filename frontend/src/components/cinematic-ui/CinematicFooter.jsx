import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { fadeInUp } from '../../design-system/animations';

/**
 * CinematicFooter — Footer CTA with tagline
 * Matches video: centered tagline + CTA button + logo + links
 */
export default function CinematicFooter() {
  return (
    <footer className="cin-footer">
      {/* Main CTA Copy */}
      <motion.h2
        className="cin-footer__tagline"
        variants={fadeInUp}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        custom={0}
      >
        Your Brain Data,
        <br />
        Decoded by AI
      </motion.h2>

      <motion.p
        className="cin-footer__description"
        variants={fadeInUp}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        custom={0.15}
      >
        NeuroScan helps you upload, analyze, and understand
        EEG recordings — with smart AI-powered insights and
        clean dashboards.
      </motion.p>

      {/* Footer CTA */}
      <motion.div
        variants={fadeInUp}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        custom={0.3}
      >
        <Link to="/dashboard" className="cin-footer__cta">
          See Dashboard
          <span className="cin-footer__cta-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="7" y1="17" x2="17" y2="7" />
              <polyline points="7 7 17 7 17 17" />
            </svg>
          </span>
        </Link>
      </motion.div>

      {/* Bottom Row */}
      <motion.div
        className="cin-footer__bottom"
        variants={fadeInUp}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        custom={0.4}
      >
        {/* Brand */}
        <div>
          <div className="cin-footer__brand">
            <div className="cin-footer__brand-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2a7 7 0 0 1 7 7c0 2.5-1.5 4.5-3 6s-2 3-2 5h-4c0-2-0.5-3.5-2-5s-3-3.5-3-6a7 7 0 0 1 7-7z" />
                <path d="M9 20h6" /><path d="M10 22h4" />
              </svg>
            </div>
            NeuroScan
          </div>
          <p className="cin-footer__copyright">
            © NeuroScan 2025 &nbsp; All rights reserved &nbsp;&nbsp; Privacy policy
          </p>
        </div>

        {/* Links */}
        <div className="cin-footer__links">
          <span className="cin-footer__links-title">Links</span>
          <div className="cin-footer__links-grid">
            <Link to="/" className="cin-footer__link">Home</Link>
            <Link to="/dashboard" className="cin-footer__link">Dashboard</Link>
            <Link to="/analyze" className="cin-footer__link">Analyze</Link>
            <Link to="/report" className="cin-footer__link">Report</Link>
          </div>
        </div>
      </motion.div>
    </footer>
  );
}
