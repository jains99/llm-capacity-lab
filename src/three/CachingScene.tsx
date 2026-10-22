import { Edges, Line } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { Metrics } from "../lib/formulas";
import { C, CameraRig, PulseRing, SceneLabel, mulberry32 } from "./primitives";

const POS: [number, number, number] = [-0.8, 6.6, 12.4];
const TARGET: [number, number, number] = [-0.8, 1.1, 0];
const BRANCHES = [-1.3, 0, 1.3];
const JUNCTION = 0.15;
const LEAF = 4.1;

function CacheDots({ m }: { m: Metrics }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const dots = useMemo(() => {
    const rand = mulberry32(29);
    return Array.from({ length: 18 }, () => ({
      offset: rand(),
      branch: BRANCHES[Math.floor(rand() * BRANCHES.length)],
      hit: rand() < m.prefixHit,
      speed: 0.8 + rand() * 0.5,
    }));
  }, [m.prefixHit]);

  useEffect(() => {
    if (!ref.current) return;
    const c = new THREE.Color();
    dots.forEach((d, i) => {
      ref.current?.setColorAt(i, c.set(d.hit ? C.lime : "#9ad9ef"));
    });
    if (ref.current.instanceColor) ref.current.instanceColor.needsUpdate = true;
  }, [dots]);

  useFrame(({ clock }) => {
    if (!ref.current) return;
    const time = clock.getElapsedTime();
    dots.forEach((d, i) => {
      const p = (time * 0.11 * d.speed + d.offset) % 1;
      let x: number;
      let y = 1.05;
      let z = 0;
      if (p < 0.5) {
        x = -6.05 + (p / 0.5) * 6.2;
      } else {
        const q = (p - 0.5) / 0.5;
        x = JUNCTION + q * (LEAF - JUNCTION);
        z = d.branch * q;
        y = 1.05 + Math.sin(q * Math.PI) * 0.16;
      }
      dummy.position.set(x, y, z);
      dummy.scale.setScalar(0.055 + (d.hit ? 0.012 : 0));
      dummy.updateMatrix();
      ref.current?.setMatrixAt(i, dummy.matrix);
    });
    ref.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={ref} args={[undefined, undefined, 18]} frustumCulled={false}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

export function CachingScene({ m }: { m: Metrics }) {
  const { size } = useThree();
  const narrow = size.width < 800;
  const enabled = m.prefixHit > 0;
  const blocks = [-5.55, -4.55, -3.55, -2.55, -1.55, -0.55];
  const spineY = 1.05;

  return (
    <group>
      <CameraRig pos={POS} target={TARGET} mobileScale={1.9} />

      <SceneLabel position={[-0.2, 3.05, 0]} tone={enabled ? "hot" : "warn"}>
        {enabled
          ? `prefill skipped · ${Math.round(m.prefillSavedFrac * 100)}%`
          : "prefix cache off · no reuse"}
      </SceneLabel>

      <SceneLabel position={[-3.3, 1.95, 0]} tone={enabled ? "hot" : "warn"}>
        {enabled
          ? narrow
            ? `system prompt · ${Math.round(m.sharedSaving * 100)}% of KV`
            : `system prompt · stored once · ${Math.round(m.sharedSaving * 100)}% of KV`
          : "every request pays for its own prefix"}
      </SceneLabel>

      {narrow ? null : <SceneLabel position={[2.35, 2.35, 0]}>private suffix</SceneLabel>}

      {narrow ? null : <SceneLabel position={[-5.45, 0.15, 0]}>requests</SceneLabel>}

      <mesh position={[-6.05, spineY, 0]}>
        <sphereGeometry args={[0.2, 20, 20]} />
        <meshStandardMaterial
          color="#0c4d5e"
          emissive={C.cyan}
          emissiveIntensity={0.5}
          roughness={0.35}
        />
      </mesh>
      <mesh position={[-6.25, spineY, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.34, 0.008, 6, 36]} />
        <meshBasicMaterial color={C.cyan} transparent opacity={0.35} toneMapped={false} />
      </mesh>

      <Line
        points={[
          [-5.9, spineY, 0],
          [JUNCTION, spineY, 0],
        ]}
        color="#2c6f80"
        lineWidth={1.2}
        transparent
        opacity={0.8}
      />

      {blocks.map((x) => (
        <mesh key={x} position={[x, spineY, 0]}>
          <boxGeometry args={[0.54, 0.54, 0.54]} />
          <meshStandardMaterial
            color={enabled ? "#0b3c4a" : "#202b36"}
            emissive={C.cyan}
            emissiveIntensity={enabled ? 0.34 : 0.03}
            metalness={0.4}
            roughness={0.38}
          />
          <Edges scale={1.002} color={enabled ? "#2e7d90" : "#2c3947"} />
        </mesh>
      ))}

      {BRANCHES.map((z, i) => (
        <group key={z}>
          <Line
            points={[
              [JUNCTION, spineY, 0],
              [2.0, spineY + 0.08, z * 0.45],
              [LEAF, spineY, z],
            ]}
            color="#8a6326"
            lineWidth={1.4}
            transparent
            opacity={0.55}
          />
          {[0.45, 0.72].map((q) => (
            <mesh
              key={q}
              position={[
                JUNCTION + q * (LEAF - JUNCTION),
                spineY + Math.sin(q * Math.PI) * 0.16,
                z * q,
              ]}
            >
              <boxGeometry args={[0.3, 0.3, 0.3]} />
              <meshStandardMaterial
                color="#4a3414"
                emissive={C.amber}
                emissiveIntensity={0.22}
                metalness={0.3}
                roughness={0.42}
              />
            </mesh>
          ))}
          <mesh position={[LEAF, spineY, z]}>
            <boxGeometry args={[0.62, 0.62, 0.62]} />
            <meshStandardMaterial
              color="#4a3414"
              emissive={C.amber}
              emissiveIntensity={0.26}
              metalness={0.3}
              roughness={0.4}
            />
            <Edges scale={1.002} color="#916a2c" />
          </mesh>
          <SceneLabel
            position={narrow ? [LEAF + 0.2, 1.1 + i * 0.85, 0] : [LEAF, 1.65, z]}
          >
            {narrow ? `u${i + 1}` : `user ${i + 1}`}
          </SceneLabel>
        </group>
      ))}

      <PulseRing
        position={[JUNCTION, spineY + 0.02, 0]}
        color={enabled ? C.lime : C.magenta}
        radius={1.0}
        period={2.2}
      />

      <CacheDots m={m} />

      <pointLight position={[-3.5, 2.2, 2.2]} intensity={7} color={C.cyan} distance={7} />
      <pointLight position={[3.6, 2.0, -2]} intensity={6} color={C.amber} distance={7} />
    </group>
  );
}
