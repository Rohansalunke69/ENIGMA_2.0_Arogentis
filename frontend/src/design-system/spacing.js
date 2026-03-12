/* ═══════════════════════════════════════════
   Cinematic Design System — Spacing & Layout
   ═══════════════════════════════════════════ */

export const spacing = {
  xs: '0.25rem',
  sm: '0.5rem',
  md: '1rem',
  lg: '1.5rem',
  xl: '2rem',
  '2xl': '3rem',
  '3xl': '4rem',
  '4xl': '6rem',

  /* Page padding */
  pagePadding: 'clamp(1rem, 3vw, 3rem)',
  sectionGap: 'clamp(3rem, 6vw, 6rem)',
};

export const breakpoints = {
  mobile: '480px',
  tablet: '768px',
  desktop: '1024px',
  wide: '1280px',
};

export const layout = {
  maxContentWidth: '1200px',
  navHeight: '64px',
  cardRadius: '24px',
  buttonRadius: '999px',
};

export default { spacing, breakpoints, layout };
