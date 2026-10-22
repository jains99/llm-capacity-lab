import { useMemo } from "react";
import type { Metrics } from "../lib/formulas";
import { C, CameraRig, FlowBeads, GpuBox, PulseRing, SceneLabel, fillColor } from "./primitives";

const POS: [number, number, number] = [0, 5.6, 11.6];
const TARGET: [number, number, number] = [0, 0.85, 0];

export function OverviewScene({ m }: { m: Metrics }) {
  const show = Math.min(m.n, 8);
  const spacing = show > 6 ? 1.85 : 2.25;
  const xs = useMemo(
    () => Array.from({ length: show }, (_, i) => (i - (show - 1) / 2) * spacing),
    [show, spacing],
  );

  const fill = Math.min(1, m.usedGB / Math.max(1, m.capacityGB));
  const accent = fillColor(fill);
  const overloaded = m.utilization > 0.92 || !m.fits;
  const batchFrac = Math.min(1, m.batch / 32);
  const lit = Math.max(1, Math.round(batchFrac * 6));

  return (
    <group>
      <CameraRig pos={POS} target={TARGET} />

      <SceneLabel position={[0, 2.05, 0.4]} tone="hot">
        {`${m.n} × ${m.gpu.short}${m.n > show ? ` (+${m.n - show} more)` : ""} · ${
          m.mode === "tensor-parallel" ? "tensor parallel" : m.mode
        }`}
      </SceneLabel>

      <SceneLabel position={[0, -0.5, 2.1]}>
        {m.model.short} · {m.quant.label} · {m.usedGB.toFixed(0)} / {m.capacityGB.toFixed(0)} GB
      </SceneLabel>

      {overloaded ? (
        <SceneLabel position={[-6.9, 1.8, 0.4]} tone="warn">
          queue · {Math.round(m.queueLength)} waiting
        </SceneLabel>
      ) : null}

      <mesh position={[-4.4, 0.015, 0]}>
        <boxGeometry args={[8.6, 0.02, 0.05]} />
        <meshBasicMaterial color={overloaded ? C.magenta : "#2b4a5e"} transparent opacity={0.55} />
      </mesh>

      {xs.map((x, i) => (
        <group key={i}>
          <GpuBox position={[x, 0.26, 0]} accent={accent} bright={0.24} />
          <PulseRing
            position={[x, 0.01, 0]}
            color={i % 2 === 0 ? C.cyan : C.violet}
            radius={0.9}
            period={3.4}
            phase={i * 0.6}
          />
          {Array.from({ length: 6 }, (_, s) => {
            const on = s < lit;
            return (
              <mesh
                key={s}
                position={[
                  x + ((s % 3) - 1) * 0.36,
                  0.55 + Math.floor(s / 3) * 0.1,
                  0.3,
                ]}
              >
                <boxGeometry args={[0.2, 0.05, 0.2]} />
                <meshBasicMaterial
                  color={on ? (s % 2 === 0 ? C.cyan : C.lime) : "#16232f"}
                  transparent
                  opacity={on ? 0.7 : 0.55}
                  toneMapped={false}
                />
              </mesh>
            );
          })}
          <SceneLabel position={[x, 1.02, 0]}>
            G{i + 1} · {Math.round(m.usedGB / m.n)}GB
          </SceneLabel>
        </group>
      ))}

      <FlowBeads
        count={34}
        targets={xs}
        stalled={overloaded}
        color="#8fdcf2"
        accent={C.lime}
        speed={0.14}
        spread={2.8}
        size={0.055}
      />

      <SceneLabel position={[-3.4, 2.35, 1.4]}>
        {overloaded ? "admission queue" : "traffic"}
      </SceneLabel>

      <pointLight position={[0, 3.2, 3.6]} intensity={10} color={accent} distance={9} />
    </group>
  );
}
