import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { fadeInUp, slideInRight } from '../../design-system/animations';
import CinematicBrain3D from './CinematicBrain3D';

/**
 * CinematicHero — Full-screen hero section
 * Left: cinematic title + CTA
 * Right: Brain3D model with glow backdrop
 */
export default function CinematicHero() {
  return (
    <section className="cin-hero">
      {/* Left Content */}
      <div className="cin-hero__content">
        {/* Title */}
        <motion.h1
          className="cin-hero__title"
          variants={fadeInUp}
          initial="hidden"
          animate="visible"
          custom={0.4}
        >
          Next-Gen EEG
          <br />
          Screening with AI
          <br />
          Precision
        </motion.h1>

        {/* Description */}
        <motion.p
          className="cin-hero__description"
          variants={fadeInUp}
          initial="hidden"
          animate="visible"
          custom={0.6}
        >
          NeuroScan helps you upload, analyze, and understand
          EEG recordings — with smart AI-powered schizophrenia
          risk assessment and clean dashboards.
        </motion.p>

        {/* CTA Button (pill-style with arrow icon) */}
        <motion.div
          variants={fadeInUp}
          initial="hidden"
          animate="visible"
          custom={0.8}
        >
          <Link to="/dashboard" className="cin-hero__cta">
            See Dashboard
            <span className="cin-hero__cta-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="7" y1="17" x2="17" y2="7" />
                <polyline points="7 7 17 7 17 17" />
              </svg>
            </span>
          </Link>
        </motion.div>
      </div>

      {/* Right Visual — Brain3D with glow backdrop */}
      <motion.div
        className="cin-hero__visual"
        variants={slideInRight}
        initial="hidden"
        animate="visible"
        custom={0.3}
      >
        <div className="cin-hero__glow" />
        <div className="cin-hero__brain-container">
          <CinematicBrain3D />
        </div>
      </motion.div>
    </section>
  );
}
