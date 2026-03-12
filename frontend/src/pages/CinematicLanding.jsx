import CinematicLayout from '../components/cinematic-ui/CinematicLayout';
import CinematicHero from '../components/cinematic-ui/CinematicHero';
import CinematicParticles from '../components/cinematic-ui/CinematicParticles';

/**
 * CinematicLanding — Premium cinematic landing page
 * Wraps existing Brain3D component in a cinematic layout
 * Does NOT modify or import any existing business logic
 */
export default function CinematicLanding() {
  return (
    <CinematicLayout>
      <CinematicParticles count={25} />
      <CinematicHero />
    </CinematicLayout>
  );
}
