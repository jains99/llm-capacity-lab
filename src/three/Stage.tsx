import { Grid } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Suspense, useMemo } from "react";
import * as THREE from "three";
import type { Metrics, TopologyAdvice } from "../lib/formulas";
import type { Lens } from "../state/store";
import { CachingScene } from "./CachingScene";
import { LatencyScene } from "./LatencyScene";
import { MemoryScene } from "./MemoryScene";
import { OverviewScene } from "./OverviewScene";
import { TopologyScene } from "./TopologyScene";
import { LabelLayer, LabelProjector } from "./primitives";

function hasWebGL() {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(
      canvas.getContext("webgl2") ||
        canvas.getContext("webgl") ||
        canvas.getContext("experimental-webgl"),
    );
  } catch {
    return false;
  }
}

export function Stage({
  m,
  advice,
  lens,
}: {
  m: Metrics;
  advice: TopologyAdvice;
  lens: Lens;
}) {
  const supported = useMemo(hasWebGL, []);

  if (!supported) {
    return (
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "grid",
          placeItems: "center",
          padding: 30,
          textAlign: "center",
          color: "#8496a9",
          fontSize: 12,
          lineHeight: 1.7,
        }}
      >
        <div>
          <div style={{ fontSize: 10, letterSpacing: "0.16em", color: "#5b6a7c" }}>
            3D VIEW UNAVAILABLE
          </div>
          <p style={{ maxWidth: 340 }}>
            This browser did not expose WebGL. Every number, slider and chart still
            works — enable hardware acceleration to see the memory and queue scenes.
          </p>
        </div>
      </div>
    );
  }

  return (
    <>
      <Canvas
        dpr={[1, 1.7]}
        camera={{ position: [0, 6, 12], fov: 38, near: 0.1, far: 120 }}
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
        style={{ background: "transparent" }}
        onCreated={({ gl }) => {
          gl.toneMapping = THREE.ACESFilmicToneMapping;
          gl.toneMappingExposure = 1.05;
        }}
      >
        <fog attach="fog" args={["#05070a", 26, 56]} />

        <ambientLight intensity={0.32} />
        <directionalLight position={[5, 11, 9]} intensity={2.1} color="#dcecff" />
        <directionalLight position={[-8, 6, -4]} intensity={0.5} color="#4de1ff" />
        <pointLight position={[-9, 5, 6]} intensity={38} color="#4de1ff" distance={34} />
        <pointLight position={[9, 4, -7]} intensity={24} color="#ff5c8a" distance={32} />

        <Suspense fallback={null}>
          {lens === "overview" ? <OverviewScene m={m} /> : null}
          {lens === "memory" ? <MemoryScene m={m} /> : null}
          {lens === "latency" ? <LatencyScene m={m} /> : null}
          {lens === "caching" ? <CachingScene m={m} /> : null}
          {lens === "topology" ? <TopologyScene m={m} advice={advice} /> : null}
        </Suspense>

        <LabelProjector />

        <Grid
          position={[0, -0.02, 0]}
          args={[80, 80]}
          cellSize={1}
          cellThickness={0.55}
          cellColor="#1a2c39"
          sectionSize={5}
          sectionThickness={0.9}
          sectionColor="#2a4a5e"
          fadeDistance={34}
          fadeStrength={1.1}
          infiniteGrid
        />
      </Canvas>

      <LabelLayer />
    </>
  );
}
