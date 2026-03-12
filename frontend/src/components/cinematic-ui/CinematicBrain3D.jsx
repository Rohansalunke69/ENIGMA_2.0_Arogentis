import { useRef, useMemo } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls, PointMaterial, Points } from '@react-three/drei';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import * as THREE from 'three';
import React from 'react'; // Added React import for React.Suspense

/**
 * CinematicBrain3D — Enhanced 3D brain visualization for the hero section
 * Uses the same brain_points.json data but with:
 *   - Larger point sizes for visibility in the circular viewport
 *   - Cyan/teal color scheme matching the cinematic design
 *   - More neural particles with glow
 *   - Optimized camera for the dark sphere container
 *
 * This is a NEW component — Brain3D.jsx is NOT modified.
 */

/* ── Highly Realistic Anatomical 3D Brain Mesh ── */
function CinematicBrainMesh() {
  const groupRef = useRef();
  
  // Load the mathematically true anatomical biological brain (fsaverage)
  const obj = useLoader(OBJLoader, '/brain.obj');

  // Apply a custom, highly cinematic medical shader to the mesh
  useMemo(() => {
    obj.traverse((child) => {
      if (child.isMesh) {
        // Compute vertex normals for smooth shading
        child.geometry.computeVertexNormals();
        
        // Fleshy biological tone infused with the cinematic scanner aesthetic
        child.material = new THREE.MeshStandardMaterial({
          color: '#e2aeb3', // Pinkish fleshy realistic base
          emissive: '#082f49', // Dark cyan core
          emissiveIntensity: 0.2,
          roughness: 0.4,
          metalness: 0.1,
          transparent: true,
          opacity: 0.95,
        });
      }
    });
  }, [obj]);

  useFrame((state) => {
    if (!groupRef.current) return;
    // Start from lateral (side) view and slowly rotate
    groupRef.current.rotation.y = (Math.PI / 2) + state.clock.elapsedTime * 0.15;
    groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.8) * 1.5;
  });

  return (
    <group ref={groupRef} scale={0.7}>
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <primitive object={obj} />
      </group>
      
      {/* Subtle Outer Glow mimicking brain energy */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[75, 32, 32]} />
        <meshBasicMaterial 
          color="#38bdf8" 
          transparent 
          opacity={0.03} 
          blending={THREE.AdditiveBlending} 
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

/* ── Neural Sparkle Particles ── */
function CinematicNeuralParticles() {
  const pointsRef = useRef();
  const particleCount = 100;

  const [positions] = useMemo(() => { // Changed useState to useMemo for positions
    const pos = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);
      // Neural shell around the 40-unit brain
      const r = 45 + Math.random() * 25;
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);
    }
    return [pos]; // Return as an array for useMemo
  }, [particleCount]); // Dependency array for useMemo

  useFrame((state, delta) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y -= delta * 0.08;
      pointsRef.current.rotation.z += delta * 0.03;
    }
  });

  return (
    <Points ref={pointsRef} positions={positions} stride={3}>
      <PointMaterial
        transparent
        color="#67e8f9"
        size={1.8}
        sizeAttenuation={true}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        opacity={0.7}
      />
    </Points>
  );
}

/* ── Main Component ── */
export default function CinematicBrain3D() {
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        camera={{ position: [0, 0, 150], fov: 50 }}
        gl={{ alpha: true, antialias: true }}
        style={{ background: 'transparent' }}
      >
        <ambientLight intensity={0.9} />
        {/* Core cinematic lighting for biological textures */}
        <directionalLight position={[100, 100, 50]} intensity={2.5} color="#ffffff" />
        <directionalLight position={[-100, -100, -50]} intensity={1.5} color="#f59e0b" />
        <spotLight position={[0, 100, 100]} intensity={2} color="#38bdf8" penumbra={1} />

        {/* Load the anatomical mesh component */}
        <React.Suspense fallback={null}>
          <CinematicBrainMesh />
        </React.Suspense>
        
        <CinematicNeuralParticles />

        <OrbitControls
          enableZoom={false}
          enablePan={false}
          autoRotate={true}
          autoRotateSpeed={0.6}
          maxPolarAngle={Math.PI / 1.5}
          minPolarAngle={Math.PI / 3}
        />
      </Canvas>
    </div>
  );
}
