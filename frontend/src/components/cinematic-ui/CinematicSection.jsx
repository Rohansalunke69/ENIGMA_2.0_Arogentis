import { motion } from 'framer-motion';
import { fadeInUp } from '../../design-system/animations';

/**
 * CinematicSection — Scroll-triggered reveal wrapper
 * Wraps any content and animates it into view on scroll
 */
export default function CinematicSection({ children, className = '', delay = 0 }) {
  return (
    <motion.section
      className={`cin-section ${className}`}
      variants={fadeInUp}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-80px' }}
      custom={delay}
    >
      {children}
    </motion.section>
  );
}
