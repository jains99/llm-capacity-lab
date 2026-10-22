import { useThree } from "@react-three/fiber";
import type { Metrics } from "../lib/formulas";
import { C, CameraRig, FlowBeads, GpuBox, PulseRing, SceneLabel } from "./primitives";

const POS: [number, number, number] = [-0.7, 4.6, 11.8];
const TARGET: [number, number, number] = [-0.7, 0.95, 0];
const GPU_X = 2.3;

export function LatencyScene({ m }: { m: Metrics }) {
  const { size } = useThree();
  const narrow = size.width < 680;
  const overloaded = m.utilization > 0.92 || !m.fits || m.maxUsers === 0;
  const queueCount = Math.min(14, Math.ceil(m.queueLength));
  const slots = 18;
  const active = Math.max(1, Math.min(slots, Math.ceil(m.batch / 2)));

  return (
    <group>
      <CameraRig pos={POS} target={TARGET} mobileScale={1.7} />

      <SceneLabel position={[GPU_X, 2.05, 0]} tone="hot">
        {narrow ? `batch · ${Math.round(m.batch)}` : `continuous batch · ${Math.round(m.batch)}`}
      </SceneLabel>

      <SceneLabel position={[GPU_X, -0.58, 0]}>
        {narrow
          ? `14 ms/tok · ${m.tokPerSecUser.toFixed(1)} tok/s`
          : `decode ${Math.round(m.tpotMs)} ms/token · ${m.tokPerSecUser.toFixed(1)} tok/s`}
      </SceneLabel>

      <mesh position={[-2.2, 0.3, 0]}>
        <boxGeometry args={[7.6, 0.05, 0.34]} />
        <meshStandardMaterial color="#101b24" metalness={0.7} roughness={0.4} />
      </mesh>
      <mesh position={[-2.2, 0.33, 0]}>
        <boxGeometry args={[7.6, 0.008, 0.05]} />
        <meshBasicMaterial color="#2b4f64" toneMapped={false} />
      </mesh>

      <mesh position={[-6.4, 0.31, 0]}>
        <boxGeometry args={[1.0, 0.05, 2.7]} />
        <meshBasicMaterial color={overloaded ? C.magenta : "#1e3444"} transparent opacity={0.35} />
      </mesh>

      {Array.from({ length: queueCount }, (_, i) => (
        <mesh
          key={i}
          position={[-6.25 + (i % 2) * 0.24, 0.48 + i * 0.2, ((i % 3) - 1) * 0.18]}
        >
          <boxGeometry args={[0.15, 0.12, 0.15]} />
          <meshBasicMaterial color={C.magenta} transparent opacity={0.7} toneMapped={false} />
        </mesh>
      ))}

      {overloaded ? (
        <SceneLabel position={[GPU_X, 2.9, 0]} tone="warn">
          queue · {Math.round(m.queueLength)} waiting
        </SceneLabel>
      ) : null}

      <FlowBeads
        count={30}
        targets={[1.3]}
        stalled={overloaded}
        color="#8fdcf2"
        accent={C.lime}
        speed={0.2}
        spread={2.3}
        y={0.62}
        size={0.062}
      />

      <GpuBox
        position={[GPU_X, 0.4, 0]}
        width={2.5}
        height={0.7}
        depth={1.9}
        accent={C.cyan}
        bright={0.38}
      />

      {Array.from({ length: slots }, (_, i) => {
        const on = i < active;
        const col = i % 6;
        const row = Math.floor(i / 6);
        return (
          <mesh
            key={i}
            position={[GPU_X + (col - 2.5) * 0.36, 0.82, -0.45 + row * 0.45]}
          >
            <boxGeometry args={[0.27, 0.07, 0.3]} />
            <meshBasicMaterial
              color={on ? (row === 0 ? C.cyan : row === 1 ? C.lime : C.violet) : "#131f29"}
              transparent
              opacity={on ? 0.85 : 0.5}
              toneMapped={false}
            />
          </mesh>
        );
      })}

      <PulseRing position={[GPU_X, 0.78, 0]} color={C.cyan} radius={1.15} period={2.4} />
      <PulseRing position={[GPU_X, 0.78, 0]} color={C.lime} radius={0.62} period={2.4} phase={1.2} />

      <pointLight position={[3.0, 2.8, 3.5]} intensity={20} color={C.cyan} distance={9} />
    </group>
  );
}
