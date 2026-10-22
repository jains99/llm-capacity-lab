import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import type { Metrics, TopologyAdvice } from "../lib/formulas";
import { C, CameraRig, GpuBox, PulseRing, SceneLabel } from "./primitives";

const POS: [number, number, number] = [0, 5.4, 12.2];
const TARGET: [number, number, number] = [0, 0.85, 0];
const LINK_Y = 1.05;

function LinkFlow({ from, to }: { from: number; to: number }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  useFrame(({ clock }) => {
    if (!ref.current) return;
    const t = clock.getElapsedTime();
    for (let i = 0; i < 4; i++) {
      const p = (t * 0.5 + i / 4) % 1;
      const dir = i % 2 === 0 ? p : 1 - p;
      dummy.position.set(from + (to - from) * dir, LINK_Y, 0);
      dummy.scale.setScalar(0.045);
      dummy.updateMatrix();
      ref.current.setMatrixAt(i, dummy.matrix);
    }
    ref.current.instanceMatrix.needsUpdate = true;
  });
  return (
    <instancedMesh ref={ref} args={[undefined, undefined, 4]} frustumCulled={false}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshBasicMaterial color={C.cyan} toneMapped={false} />
    </instancedMesh>
  );
}

export function TopologyScene({ m, advice }: { m: Metrics; advice: TopologyAdvice }) {
  const show = Math.min(8, Math.max(m.n, advice.recommended));
  const spacing = 2.15;
  const xs = useMemo(
    () => Array.from({ length: show }, (_, i) => (i - (show - 1) / 2) * spacing),
    [show],
  );
  const isTp = advice.mode === "tensor-parallel" || advice.mode === "multi-node";
  const isReplica = advice.mode === "replicas";

  return (
    <group>
      <CameraRig pos={POS} target={TARGET} />

      <SceneLabel position={[0, 3.05, 0]} tone="hot">
        {advice.mode === "tensor-parallel"
          ? `tensor parallel × ${advice.recommended}`
          : advice.mode === "replicas"
            ? `replicas × ${advice.recommended}`
            : advice.mode === "multi-node"
              ? `multi-node × ${advice.recommended}`
              : "single GPU"}
      </SceneLabel>

      <SceneLabel position={[0, -1.15, 0]}>
        {isTp
          ? `${m.gpu.linkName} · ${m.gpu.linkGBs} GB/s all-reduce · ${Math.round(
              advice.efficiency * 100,
            )}% scaling`
          : isReplica
            ? "independent copies · no interconnect traffic"
            : "weights + KV on one device"}
      </SceneLabel>

      {show > m.n ? (
        <SceneLabel position={[xs[Math.min(show - 1, m.n)] ?? 0, 2.1, 0]} tone="warn">
          +{advice.recommended - m.n} needed
        </SceneLabel>
      ) : null}

      {xs.map((x, i) => {
        const active = i < m.n;
        const needed = i >= m.n && i < advice.recommended;
        return (
          <group key={i}>
            <GpuBox
              position={[x, 0.32, 0]}
              accent={needed ? C.amber : i < advice.recommended ? C.lime : "#37506a"}
              dim={!active}
              bright={active ? 0.26 : 0.02}
            />
            {i < advice.recommended ? (
              <PulseRing
                position={[x, 0.02, 0]}
                color={i < m.n ? C.lime : C.amber}
                radius={1.0}
                period={3.0}
                phase={i * 0.4}
              />
            ) : null}
            {isReplica && active ? (
              <mesh position={[x, 1.8, 0]}>
                <cylinderGeometry args={[0.03, 0.34, 2.5, 16, 1, true]} />
                <meshBasicMaterial
                  color={C.lime}
                  transparent
                  opacity={0.09}
                  side={THREE.DoubleSide}
                  depthWrite={false}
                />
              </mesh>
            ) : null}
            <SceneLabel position={[x, -0.35, 0]}>
              {active ? `G${i + 1}` : "needed"}
            </SceneLabel>
          </group>
        );
      })}

      {isTp ? (
        <group>
          <mesh position={[0, LINK_Y, 0]} rotation={[0, 0, Math.PI / 2]}>
            <cylinderGeometry args={[0.022, 0.022, xs[xs.length - 1] - xs[0] + 1.2, 8]} />
            <meshBasicMaterial color="#2d5f78" />
          </mesh>
          {xs.map((x) => (
            <mesh key={`s${x}`} position={[x, LINK_Y - 0.26, 0]}>
              <cylinderGeometry args={[0.014, 0.014, 0.5, 6]} />
              <meshBasicMaterial color="#2d5f78" />
            </mesh>
          ))}
        </group>
      ) : null}

      {isTp
        ? xs.slice(0, -1).map((x, i) => {
            const next = xs[i + 1];
            return (
              <group key={`l${i}`}>
                {i < Math.min(m.n, advice.recommended) - 1 ? (
                  <LinkFlow from={x} to={next} />
                ) : null}
              </group>
            );
          })
        : null}

      <SceneLabel position={[0, 1.32, 0]}>
        {m.n} installed · {advice.recommended} recommended
      </SceneLabel>
    </group>
  );
}
