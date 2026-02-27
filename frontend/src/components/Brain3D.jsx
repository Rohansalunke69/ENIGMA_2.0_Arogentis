import { useRef, useState, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Environment, Points, PointMaterial } from "@react-three/drei";
import * as THREE from "three";

// A stylized real brain using the extracted MNE fsaverage coordinates
function RealBrainPoints() {
    const pointsRef = useRef();
    const [positions, setPositions] = useState(null);

    useEffect(() => {
        // Load the subsampled real brain coordinates (fsaverage)
        fetch('/brain_points.json')
            .then(res => res.json())
            .then(data => {
                const positionsArray = new Float32Array(data);
                setPositions(positionsArray);
            })
            .catch(err => console.error("Error loading brain points:", err));
    }, []);

    useFrame((state) => {
        if (!pointsRef.current) return;
        // Gentle rotation over time
        pointsRef.current.rotation.y = state.clock.elapsedTime * 0.15;
        // Subtle hover effect
        pointsRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.8) * 0.5;
    });

    if (!positions) {
        return (
            <mesh>
                <sphereGeometry args={[10, 16, 16]} />
                <meshBasicMaterial color="#f59e0b" wireframe opacity={0.2} transparent />
            </mesh>
        );
    }

    return (
        <group>
            {/* The main point cloud - using real brain topology */}
            <Points ref={pointsRef} positions={positions} stride={3}>
                <PointMaterial
                    transparent
                    color="#fbbf24"
                    size={0.15}
                    sizeAttenuation={true}
                    depthWrite={false}
                    blending={THREE.AdditiveBlending}
                />
            </Points>
            {/* A subtle glowing core for depth */}
            <mesh position={[0, 0, 0]}>
                <sphereGeometry args={[45, 32, 32]} />
                <meshBasicMaterial color="#f59e0b" transparent opacity={0.05} blending={THREE.AdditiveBlending} />
            </mesh>
        </group>
    );
}

// 60 Floating neural sparks around the brain
function NeuralParticles() {
    const pointsRef = useRef();
    const particleCount = 60;

    // Create random particles around the brain scale (fsaverage bounds are roughly +/- 70)
    const [positions] = useState(() => {
        const pos = new Float32Array(particleCount * 3);
        for (let i = 0; i < particleCount; i++) {
            // Random point in a spherical shell around the brain
            const u = Math.random();
            const v = Math.random();
            const theta = u * 2.0 * Math.PI;
            const phi = Math.acos(2.0 * v - 1.0);
            const r = 60 + Math.random() * 40; // Shell radius
            pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
            pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            pos[i * 3 + 2] = r * Math.cos(phi);
        }
        return pos;
    });

    useFrame((state, delta) => {
        if (pointsRef.current) {
            pointsRef.current.rotation.y -= delta * 0.05;
            pointsRef.current.rotation.z += delta * 0.02;
        }
    });

    return (
        <Points ref={pointsRef} positions={positions} stride={3}>
            <PointMaterial
                transparent
                color="#4ade80"
                size={1.2}
                sizeAttenuation={true}
                depthWrite={false}
                blending={THREE.AdditiveBlending}
                opacity={0.8}
            />
        </Points>
    );
}

export default function Brain3D() {
    return (
        <div style={{ width: "100%", height: "100%", position: "relative" }}>
            <Canvas camera={{ position: [0, 0, 160], fov: 45 }}>
                <ambientLight intensity={0.5} />
                <directionalLight position={[100, 100, 50]} intensity={1.5} color="#ffffff" />
                <directionalLight position={[-100, -100, -50]} intensity={0.5} color="#f59e0b" />

                <RealBrainPoints />
                <NeuralParticles />

                <OrbitControls
                    enableZoom={false}
                    enablePan={false}
                    autoRotate={true}
                    autoRotateSpeed={0.5}
                    maxPolarAngle={Math.PI / 1.5}
                    minPolarAngle={Math.PI / 3}
                />
            </Canvas>
        </div>
    );
}
