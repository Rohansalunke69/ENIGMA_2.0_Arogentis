/* ═══════════════════════════════════════════
   Cinematic Design System — Framer Motion Variants
   ═══════════════════════════════════════════ */

/* ── Fade In Up — default hero entrance ── */
export const fadeInUp = {
  hidden: { opacity: 0, y: 40 },
  visible: (delay = 0) => ({
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.8,
      delay,
      ease: [0.16, 1, 0.3, 1],
    },
  }),
};

/* ── Fade In — simple opacity ── */
export const fadeIn = {
  hidden: { opacity: 0 },
  visible: (delay = 0) => ({
    opacity: 1,
    transition: { duration: 0.6, delay, ease: 'easeOut' },
  }),
};

/* ── Scale In — card entrance ── */
export const scaleIn = {
  hidden: { opacity: 0, scale: 0.92 },
  visible: (delay = 0) => ({
    opacity: 1,
    scale: 1,
    transition: {
      duration: 0.6,
      delay,
      ease: [0.16, 1, 0.3, 1],
    },
  }),
};

/* ── Slide In From Left ── */
export const slideInLeft = {
  hidden: { opacity: 0, x: -60 },
  visible: (delay = 0) => ({
    opacity: 1,
    x: 0,
    transition: {
      duration: 0.7,
      delay,
      ease: [0.16, 1, 0.3, 1],
    },
  }),
};

/* ── Slide In From Right ── */
export const slideInRight = {
  hidden: { opacity: 0, x: 60 },
  visible: (delay = 0) => ({
    opacity: 1,
    x: 0,
    transition: {
      duration: 0.7,
      delay,
      ease: [0.16, 1, 0.3, 1],
    },
  }),
};

/* ── Stagger Children Container ── */
export const staggerContainer = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.12,
      delayChildren: 0.3,
    },
  },
};

/* ── Stagger Child ── */
export const staggerChild = {
  hidden: { opacity: 0, y: 24 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: [0.16, 1, 0.3, 1],
    },
  },
};

/* ── Navbar ── */
export const navbarReveal = {
  hidden: { opacity: 0, y: -20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.6,
      delay: 0.1,
      ease: [0.16, 1, 0.3, 1],
    },
  },
};

/* ── Hover Scale (for buttons, cards) ── */
export const hoverScale = {
  scale: 1.03,
  transition: { duration: 0.2, ease: 'easeOut' },
};

export const tapScale = {
  scale: 0.97,
};

/* ── Float / Breathe (for background orbs) ── */
export const floatAnimation = {
  y: [0, -15, 0],
  transition: {
    duration: 6,
    repeat: Infinity,
    ease: 'easeInOut',
  },
};

export const breatheAnimation = {
  scale: [1, 1.05, 1],
  opacity: [0.5, 0.8, 0.5],
  transition: {
    duration: 4,
    repeat: Infinity,
    ease: 'easeInOut',
  },
};
