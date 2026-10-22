import { Edges } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import { useMemo } from "react";
import type { Metrics } from "../lib/formulas";
import { C, CameraRig, PulseRing, SceneLabel } from "./primitives";

const POS: [number, number, number] = [-1.85, 6.6, 15.4];
const TARGET: [number, number, number] = [-1.85, 2.2, 0];

export function MemoryScene({ m }: { m: Metrics }) {
  const { size } = useThree();
  const narrow = size.width < 800;
  const tiny = size.width < 620;
  const show = Math.min(m.n, 8);
  const spacing = show > 5 ? 1.85 : 2.1;
  const xs = useMemo(
    () => Array.from({ length: show }, (_, i) => (i - (show - 1) / 2) * spacing),
    [show, spacing],
  );

  const unit = 5.2 / m.gpu.vramGB;
  const usable = m.gpu.vramGB * 0.92;
  const wPer = (m.mode === "tensor-parallel" ? m.weightsGB / m.n : m.weightsGB) * unit;
  const kvPer = (m.kvTotalGB / m.n) * unit;
  const ovPer = (m.overheadGB / m.n) * unit;
  const usedPer = wPer + kvPer + ovPer;
  const cap = m.gpu.vramGB * unit;
  const capLine = usable * unit;
  const width = 1.38;

  return (
    <group>
      <CameraRig pos={POS} target={TARGET} />

      <SceneLabel position={[0, cap + 1.05, 0]} tone="hot">
        {m.n} × {m.gpu.short} · {m.mode === "tensor-parallel" ? "weights sharded" : "weights copied"}
        {m.n > show ? ` · showing ${show}` : ""}
      </SceneLabel>

      {xs.map((x, i) => {
        const overflow = Math.max(0, usedPer - cap);
        const overflowVis = Math.min(overflow, cap * 0.2);
        return (
          <group key={i}>
            <mesh position={[x, cap / 2, 0]}>
              <boxGeometry args={[width + 0.22, cap, width + 0.22]} />
              <meshStandardMaterial
                color="#0a1219"
                transparent
                opacity={0.28}
                metalness={0.35}
                roughness={0.55}
              />
              <Edges scale={1.001} color="#244052" />
            </mesh>

            <mesh position={[x, wPer / 2 + 0.01, 0]}>
              <boxGeometry args={[width, Math.max(wPer, 0.02), width]} />
              <meshStandardMaterial
                color="#4a3414"
                emissive={C.amber}
                emissiveIntensity={0.13}
                metalness={0.45}
                roughness={0.42}
              />
              <Edges scale={1.001} color="#8a6326" />
            </mesh>

            <mesh position={[x, wPer + kvPer / 2 + 0.01, 0]}>
              <boxGeometry args={[width, Math.max(kvPer, 0.02), width]} />
              <meshStandardMaterial
                color="#0c3f4d"
                emissive={C.cyan}
                emissiveIntensity={0.2}
                metalness={0.25}
                roughness={0.38}
              />
              <Edges scale={1.001} color="#2c7c92" />
            </mesh>

            <mesh position={[x, wPer + kvPer + ovPer / 2 + 0.01, 0]}>
              <boxGeometry args={[width * 0.86, Math.max(ovPer, 0.02), width * 0.86]} />
              <meshStandardMaterial color="#242e3a" metalness={0.55} roughness={0.5} />
            </mesh>

            {overflow > 0 ? (
              <mesh position={[x, cap + overflowVis / 2, 0]}>
                <boxGeometry args={[width, overflowVis, width]} />
                <meshStandardMaterial
                  color="#5c1717"
                  emissive={C.red}
                  emissiveIntensity={0.35}
                  transparent
                  opacity={0.85}
                />
              </mesh>
            ) : null}

            <mesh position={[x, capLine, 0]}>
              <boxGeometry args={[width + 0.34, 0.018, width + 0.34]} />
              <meshBasicMaterial color="#dce9f6" transparent opacity={0.3} />
            </mesh>

            <PulseRing
              position={[x, Math.min(usedPer, cap) + 0.03, 0]}
              color={overflow > 0 ? C.red : C.cyan}
              radius={1.0}
              period={3.8}
              phase={i * 0.45}
            />

            <SceneLabel position={[x, -0.5, 0]} tone={overflow > 0 ? "warn" : undefined}>
              {tiny
                ? `${m.mode === "single" ? "GPU" : `G${i + 1}`} ${Math.round(usedPer / unit)}`
                : narrow
                  ? `${m.mode === "single" ? "GPU" : `G${i + 1}`} ${Math.round(usedPer / unit)}/${m.gpu.vramGB}`
                  : `${m.mode === "single" ? "GPU" : `G${i + 1}`} · ${Math.round(usedPer / unit)}/${m.gpu.vramGB} GB`}
            </SceneLabel>
          </group>
        );
      })}

    </group>
  );
}
