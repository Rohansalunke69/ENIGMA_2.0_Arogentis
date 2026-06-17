import { Suspense, useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF, Stage, Html, Environment } from '@react-three/drei';
import * as THREE from 'three';

/**
 * CinematicBrain3D — Realistic textured 3D brain for the hero section.
 *
 * Loads the solid, UV-mapped /brain.glb (cerebrum, cerebellum, brain stem)
 * and renders it auto-rotating inside the circular transparent hero container.
 *
 * The old brain.obj point-cloud mesh and cyan neural particles were removed.
 *
 * NOTE: the glb materials ship with metalness=1.0, which renders the brain
 * dark/metallic. We force metalness to 0.0 per mesh (do not remove that line)
 * while keeping the embedded textures.
 */

useGLTF.preload('/brain.glb');

/* ── Realistic textured anatomical brain mesh ── */
function BrainModel() {
  const groupRef = useRef();
  const { scene } = useGLTF('/brain.glb');

  const model = useMemo(() => {
    const clone = scene.clone(true);
    clone.traverse((child) => {
      if (!child.isMesh) return;
      child.castShadow = true;
      child.receiveShadow = true;
      if (child.material) {
        child.material.roughness = Math.min(child.material.roughness ?? 0.6, 0.6);
        child.material.metalness = 0.0; // REQUIRED: model ships metalness=1.0 (renders dark). Do not remove.
        child.material.envMapIntensity = 0.8;
      }
    });
    return clone;
  }, [scene]);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.6) * 0.04;
    }
  });

  return (
    <group ref={groupRef}>
      <primitive object={model} />
    </group>
  );
}

/* ── Suspense fallback shown while the 8MB glb loads ── */
function Loader() {
  return (
    <Html center>
      <div style={{ color: '#9fb3d1', fontSize: 14, fontFamily: 'sans-serif' }}>Loading brain…</div>
    </Html>
  );
}

/* ── Main Component ── */
export default function CinematicBrain3D() {
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        dpr={[1, 2]}
        gl={{ alpha: true, antialias: true }}
        style={{ background: 'transparent' }}
        camera={{ position: [0, 0, 4], fov: 40 }}
      >
        <Suspense fallback={<Loader />}>
          <Stage adjustCamera={1.15} intensity={0.7} environment="studio" preset="rembrandt" shadows={false}>
            <BrainModel />
          </Stage>
          <Environment preset="studio" />
        </Suspense>
        <OrbitControls
          enableZoom={false}
          enablePan={false}
          autoRotate
          autoRotateSpeed={0.55}
          enableDamping
          dampingFactor={0.05}
          minPolarAngle={Math.PI / 3}
          maxPolarAngle={Math.PI / 1.5}
        />
      </Canvas>
    </div>
  );
}
