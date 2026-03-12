import { motion } from 'framer-motion';
import { hoverScale, tapScale } from '../../design-system/animations';

/**
 * CinematicButton — Pill-shaped button with hover glow
 * Variants: 'primary' (gradient) and 'ghost' (outlined)
 */
export default function CinematicButton({
  children,
  variant = 'primary',
  href,
  onClick,
  className = '',
  ...props
}) {
  const Tag = href ? motion.a : motion.button;

  return (
    <Tag
      className={`cin-btn cin-btn--${variant} ${className}`}
      href={href}
      onClick={onClick}
      whileHover={hoverScale}
      whileTap={tapScale}
      {...props}
    >
      {children}
    </Tag>
  );
}
