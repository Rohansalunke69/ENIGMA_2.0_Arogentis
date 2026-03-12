import { useMemo } from 'react';

/**
 * CinematicParticles — CSS-based floating ambient particles
 * Pure visual, no interaction, fixed position overlay
 */
export default function CinematicParticles({ count = 30 }) {
  const particles = useMemo(() => {
    return Array.from({ length: count }, (_, i) => ({
      id: i,
      left: `${Math.random() * 100}%`,
      size: 2 + Math.random() * 4,
      duration: 8 + Math.random() * 12,
      delay: Math.random() * 10,
      opacity: 0.2 + Math.random() * 0.4,
    }));
  }, [count]);

  return (
    <div className="cin-particles">
      {particles.map((p) => (
        <div
          key={p.id}
          className="cin-particle"
          style={{
            left: p.left,
            width: `${p.size}px`,
            height: `${p.size}px`,
            animationDuration: `${p.duration}s`,
            animationDelay: `${p.delay}s`,
            opacity: p.opacity,
          }}
        />
      ))}
    </div>
  );
}
