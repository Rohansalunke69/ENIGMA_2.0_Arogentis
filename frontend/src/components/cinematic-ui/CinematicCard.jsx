import { motion } from 'framer-motion';
import { scaleIn, hoverScale, tapScale } from '../../design-system/animations';

/**
 * CinematicCard — Reusable glassmorphism card with hover effects
 * Pure presentational, no business logic
 */
export default function CinematicCard({ children, delay = 0, className = '' }) {
  return (
    <motion.div
      className={`cin-card ${className}`}
      variants={scaleIn}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-50px' }}
      custom={delay}
      whileHover={hoverScale}
      whileTap={tapScale}
    >
      {children}
    </motion.div>
  );
}
