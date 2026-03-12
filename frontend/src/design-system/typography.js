/* ═══════════════════════════════════════════
   Cinematic Design System — Typography
   ═══════════════════════════════════════════ */

export const typography = {
  fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",

  /* ── Headings ── */
  heroTitle: {
    fontSize: 'clamp(2.8rem, 5vw, 4.5rem)',
    fontWeight: 300,
    letterSpacing: '-0.03em',
    lineHeight: 1.1,
  },
  heroSubtitle: {
    fontSize: 'clamp(0.95rem, 1.2vw, 1.1rem)',
    fontWeight: 400,
    lineHeight: 1.7,
  },
  sectionTitle: {
    fontSize: 'clamp(2rem, 3.5vw, 3rem)',
    fontWeight: 300,
    letterSpacing: '-0.02em',
    lineHeight: 1.2,
  },

  /* ── Nav ── */
  navLink: {
    fontSize: '0.88rem',
    fontWeight: 500,
    letterSpacing: '0.01em',
  },
  navCta: {
    fontSize: '0.85rem',
    fontWeight: 600,
    letterSpacing: '0.02em',
  },

  /* ── Body ── */
  body: {
    fontSize: '0.95rem',
    fontWeight: 400,
    lineHeight: 1.6,
  },
  caption: {
    fontSize: '0.8rem',
    fontWeight: 500,
    letterSpacing: '0.04em',
    textTransform: 'uppercase',
  },

  /* ── Footer ── */
  footerTitle: {
    fontSize: 'clamp(1.8rem, 3vw, 2.5rem)',
    fontWeight: 300,
    letterSpacing: '-0.02em',
    lineHeight: 1.3,
    fontStyle: 'italic',
  },
};

export default typography;
