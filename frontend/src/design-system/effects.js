/* ═══════════════════════════════════════════
   Cinematic Design System — Effect Presets
   CSS-in-JS style objects for glassmorphism, glow, gradients
   ═══════════════════════════════════════════ */

export const effects = {
  /* ── Glassmorphism Card ── */
  glassCard: {
    background: 'rgba(240, 245, 250, 0.88)',
    backdropFilter: 'blur(24px)',
    WebkitBackdropFilter: 'blur(24px)',
    border: '1px solid rgba(255, 255, 255, 0.5)',
    borderRadius: '24px',
    boxShadow: '0 8px 32px rgba(0, 30, 60, 0.1)',
  },

  /* ── Glass Navbar ── */
  glassNavbar: {
    background: 'rgba(255, 255, 255, 0.75)',
    backdropFilter: 'blur(20px)',
    WebkitBackdropFilter: 'blur(20px)',
    border: '1px solid rgba(255, 255, 255, 0.4)',
    borderRadius: '999px',
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.06)',
  },

  /* ── Glow Behind Hero Image ── */
  heroGlow: {
    position: 'absolute',
    width: '120%',
    height: '120%',
    background: 'radial-gradient(ellipse at center, rgba(78, 205, 196, 0.3) 0%, rgba(56, 189, 248, 0.15) 40%, transparent 70%)',
    filter: 'blur(40px)',
    pointerEvents: 'none',
    zIndex: 0,
  },

  /* ── Gradient Overlay ── */
  gradientOverlay: {
    background: 'linear-gradient(135deg, rgba(78, 205, 196, 0.08) 0%, rgba(56, 189, 248, 0.05) 50%, transparent 100%)',
  },

  /* ── Inner Shadow for depth ── */
  innerShadow: {
    boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.04), 0 8px 32px rgba(0, 30, 60, 0.08)',
  },
};

export default effects;
