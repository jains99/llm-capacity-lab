import { RoundedBox, Edges } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import {
  Children,
  useEffect,
  useId,
  useMemo,
  useRef,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import * as THREE from "three";

export const C = {
  bg: "#05070a",
  cyan: "#4de1ff",
  amber: "#ffb347",
  magenta: "#ff5c8a",
  lime: "#b6f36b",
  red: "#ff5f5f",
  violet: "#a78bfa",
  steel: "#16212d",
  steelDark: "#0b131b",
  ink: "#e8eef6",
};

export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function CameraRig({
  pos,
  target,
  mobileScale = 1.45,
}: {
  pos: [number, number, number];
  target: [number, number, number];
  mobileScale?: number;
}) {
  const { camera, size } = useThree();
  const from = useRef(camera.position.clone());
  const look = useMemo(() => new THREE.Vector3(...target), [target]);
  // Narrow stages have a narrower horizontal field of view. Pull the camera
  // back so a full GPU row or cache tree fits instead of clipping at the ends.
  const aspect = size.width / Math.max(1, size.height);
  const scale =
    aspect < 0.95 ? mobileScale : aspect < 1.32 ? Math.min(1.4, mobileScale) : 1;
  const to = useMemo(() => {
    const end = new THREE.Vector3(...pos);
    if (scale === 1) return end;
    return end.sub(look).multiplyScalar(scale).add(look);
  }, [pos, look, scale]);
  const t = useRef(0);

  useEffect(() => {
    from.current.copy(camera.position);
    t.current = 0;
  }, [to, camera]);

  useFrame((_, dt) => {
    t.current = Math.min(1, t.current + dt * 1.8);
    const e = 1 - Math.pow(1 - t.current, 3);
    camera.position.lerpVectors(from.current, to, e);
    camera.lookAt(look);
  });

  return null;
}

/* ---------------------------------------------------------------- labels */
type LabelRecord = {
  id: string;
  position: [number, number, number];
  text: string;
  tone?: "hot" | "warn";
  el: HTMLDivElement | null;
};

const labelMap = new Map<string, LabelRecord>();
let labelVersion = 0;
const labelListeners = new Set<() => void>();

const emitLabels = () => {
  labelVersion += 1;
  labelListeners.forEach((fn) => fn());
};

const subscribeLabels = (fn: () => void) => {
  labelListeners.add(fn);
  return () => {
    labelListeners.delete(fn);
  };
};

const labelSnapshot = () => labelVersion;

export function getLabels() {
  return Array.from(labelMap.values());
}

function upsertLabel(
  id: string,
  position: [number, number, number],
  text: string,
  tone?: "hot" | "warn",
) {
  const prev = labelMap.get(id);
  if (
    prev &&
    prev.text === text &&
    prev.tone === tone &&
    prev.position[0] === position[0] &&
    prev.position[1] === position[1] &&
    prev.position[2] === position[2]
  ) {
    return;
  }
  labelMap.set(id, {
    id,
    position,
    text,
    tone,
    el: prev?.el ?? null,
  });
  emitLabels();
}

function removeLabel(id: string) {
  if (labelMap.delete(id)) emitLabels();
}

function setLabelEl(id: string, el: HTMLDivElement | null) {
  const rec = labelMap.get(id);
  if (rec) rec.el = el;
}

/**
 * Labels live in the DOM overlay outside the Canvas. This component only
 * registers their text/world position; it renders nothing in the 3D tree.
 */
export function SceneLabel({
  position,
  children,
  tone,
}: {
  position: [number, number, number];
  children: ReactNode;
  tone?: "hot" | "warn";
}) {
  const id = useId();
  const [x, y, z] = position;
  const text = Children.toArray(children).join("");

  useEffect(() => {
    upsertLabel(id, [x, y, z], text, tone);
  }, [id, x, y, z, text, tone]);

  useEffect(() => {
    return () => removeLabel(id);
  }, [id]);

  return null;
}

/** Per-frame projection of the registered labels onto the canvas. */
export function LabelProjector() {
  const { camera, size } = useThree();
  const v = useMemo(() => new THREE.Vector3(), []);

  useFrame(() => {
    const halfW = size.width / 2;
    const halfH = size.height / 2;
    for (const label of getLabels()) {
      const el = label.el;
      if (!el) continue;
      v.set(label.position[0], label.position[1], label.position[2]).project(camera);
      if (v.z > 1) {
        el.style.opacity = "0";
        continue;
      }
      el.style.opacity = "1";
      el.style.transform = `translate(-50%, -50%) translate(${(v.x + 1) * halfW}px, ${
        (1 - v.y) * halfH
      }px)`;
    }
  });

  return null;
}

/** DOM overlay that renders every registered scene label. */
export function LabelLayer() {
  useSyncExternalStore(subscribeLabels, labelSnapshot, labelSnapshot);
  const labels = getLabels();
  return (
    <div className="label-layer" aria-hidden>
      {labels.map((l) => (
        <div
          key={l.id}
          ref={(el) => setLabelEl(l.id, el)}
          className={`scene-label ${l.tone ?? ""}`}
        >
          {l.text}
        </div>
      ))}
    </div>
  );
}

export function GpuBox({
  position,
  accent = C.cyan,
  dim = false,
  width = 1.7,
  height = 0.5,
  depth = 2.3,
  bright = 0.18,
}: {
  position: [number, number, number];
  accent?: string;
  dim?: boolean;
  width?: number;
  height?: number;
  depth?: number;
  bright?: number;
}) {
  const vents = [-2, -1, 0, 1, 2];
  return (
    <group position={position}>
      <RoundedBox args={[width, height, depth]} radius={0.07} smoothness={4}>
        <meshStandardMaterial
          color="#0b131a"
          metalness={0.85}
          roughness={0.34}
          emissive={accent}
          emissiveIntensity={dim ? 0.004 : bright * 0.16}
        />
        <Edges scale={1.002} threshold={25} color={dim ? "#16212c" : "#2b4356"} />
      </RoundedBox>

      <mesh position={[0, height / 2 + 0.003, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[width * 0.82, depth * 0.82]} />
        <meshStandardMaterial color="#091016" metalness={0.7} roughness={0.52} />
      </mesh>

      {vents.map((i) => (
        <mesh key={i} position={[0, height / 2 + 0.009, i * depth * 0.12]}>
          <boxGeometry args={[width * 0.52, 0.006, 0.016]} />
          <meshBasicMaterial
            color={accent}
            transparent
            opacity={dim ? 0.05 : 0.4}
            toneMapped={false}
          />
        </mesh>
      ))}

      <mesh position={[0, -height * 0.14, depth / 2 + 0.004]}>
        <boxGeometry args={[width * 0.68, 0.014, 0.008]} />
        <meshBasicMaterial
          color={accent}
          transparent
          opacity={dim ? 0.08 : 0.9}
          toneMapped={false}
        />
      </mesh>

      <mesh position={[width / 2 + 0.004, 0, 0]}>
        <boxGeometry args={[0.008, height * 0.45, depth * 0.3]} />
        <meshBasicMaterial
          color={accent}
          transparent
          opacity={dim ? 0.05 : 0.32}
          toneMapped={false}
        />
      </mesh>
    </group>
  );
}

type Bead = {
  offset: number;
  lane: number;
  target: number;
  hit: boolean;
  speed: number;
};

export function FlowBeads({
  count,
  targets,
  color = C.cyan,
  accent = C.lime,
  speed = 0.13,
  seed = 11,
  stalled = false,
  y = 0.6,
  spread = 1.5,
  size = 0.065,
}: {
  count: number;
  targets: number[];
  color?: string;
  accent?: string;
  speed?: number;
  seed?: number;
  stalled?: boolean;
  y?: number;
  spread?: number;
  size?: number;
}) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  const beads = useMemo<Bead[]>(() => {
    const rand = mulberry32(seed);
    return Array.from({ length: count }, () => ({
      offset: rand(),
      lane: (rand() - 0.5) * spread,
      target: targets[Math.floor(rand() * targets.length)] ?? 0,
      hit: rand() < 0.32,
      speed: 0.75 + rand() * 0.5,
    }));
  }, [count, targets, seed, spread]);

  useEffect(() => {
    if (!ref.current) return;
    const col = new THREE.Color();
    beads.forEach((b, i) => {
      ref.current?.setColorAt(i, col.set(b.hit ? accent : color));
    });
    if (ref.current.instanceColor) ref.current.instanceColor.needsUpdate = true;
  }, [beads, color, accent]);

  useFrame(({ clock }) => {
    if (!ref.current) return;
    const time = clock.getElapsedTime();
    beads.forEach((b, i) => {
      let px: number;
      let py: number;
      let pz = b.lane;
      if (stalled) {
        const cycle = (time * 0.06 * b.speed + b.offset) % 1;
        px = -9.2 + cycle * 2.4;
        py = y + Math.floor(cycle * 3) * 0.22 + Math.sin(time * 2 + i) * 0.03;
        pz = b.lane * 0.5;
      } else {
        const p = (time * speed * b.speed + b.offset) % 1;
        px = -9.4 + p * (b.target + 9.4);
        py = y + Math.sin(p * Math.PI) * 1.25;
      }
      dummy.position.set(px, py, pz);
      const s = size * (1 + (b.hit ? 0.35 : 0));
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      ref.current?.setMatrixAt(i, dummy.matrix);
    });
    ref.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={ref} args={[undefined, undefined, count]} frustumCulled={false}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

export function PulseRing({
  position,
  color = C.cyan,
  radius = 0.5,
  period = 2.4,
  phase = 0,
}: {
  position: [number, number, number];
  color?: string;
  radius?: number;
  period?: number;
  phase?: number;
}) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (!ref.current) return;
    const t = ((clock.getElapsedTime() + phase) % period) / period;
    const s = 0.2 + t * radius;
    ref.current.scale.setScalar(s);
    const mat = ref.current.material as THREE.MeshBasicMaterial;
    mat.opacity = (1 - t) * 0.6;
  });
  return (
    <mesh ref={ref} position={position} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[0.85, 1, 40]} />
      <meshBasicMaterial color={color} transparent opacity={0.5} side={THREE.DoubleSide} toneMapped={false} />
    </mesh>
  );
}

export function fillColor(frac: number): string {
  if (frac > 0.98) return C.red;
  if (frac > 0.86) return C.amber;
  return C.cyan;
}
