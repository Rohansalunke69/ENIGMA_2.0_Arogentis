import '../../styles/cinematic-effects.css';

/**
 * CinematicLayout — Root wrapper component
 * Provides the outer gradient glow background + inner glass container
 * Usage: <CinematicLayout>{children}</CinematicLayout>
 */
export default function CinematicLayout({ children }) {
  return (
    <div className="cinematic-outer">
      {/* Animated glow orbs */}
      <div className="cinematic-glow-orb cinematic-glow-orb--1" />
      <div className="cinematic-glow-orb cinematic-glow-orb--2" />
      <div className="cinematic-glow-orb cinematic-glow-orb--3" />

      {/* Main content card */}
      <div className="cinematic-inner">
        {children}
      </div>
    </div>
  );
}
