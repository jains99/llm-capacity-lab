import { create } from "zustand";
import type { EngineConfig } from "../lib/formulas";
import { computeMetrics, type Metrics } from "../lib/formulas";
import { BASE_CONFIG, PRESETS, presetById } from "../lib/presets";
import { clamp } from "../lib/format";

export type Lens = "overview" | "memory" | "latency" | "caching" | "topology";

const ONBOARDING_KEY = "capacity-lab:onboarding-v1";

function hasSeenOnboarding() {
  if (typeof window === "undefined") return true;
  try {
    return window.localStorage.getItem(ONBOARDING_KEY) === "done";
  } catch {
    return false;
  }
}

function rememberOnboarding() {
  try {
    window.localStorage.setItem(ONBOARDING_KEY, "done");
  } catch {
    // Private browsing or storage disabled: the tour will still be replayable.
  }
}

type State = {
  config: EngineConfig;
  lens: Lens;
  showMath: boolean;
  presetId: string;
  explain: string | null;
  onboardingOpen: boolean;
  set: <K extends keyof EngineConfig>(key: K, value: EngineConfig[K]) => void;
  setLens: (lens: Lens) => void;
  setShowMath: (v: boolean) => void;
  setExplain: (topic: string | null) => void;
  openOnboarding: () => void;
  closeOnboarding: () => void;
  applyPreset: (id: string) => void;
};

const NUMERIC: Record<string, [number, number]> = {
  gpuCount: [1, 16],
  context: [256, 262144],
  outputTokens: [16, 4096],
  users: [1, 20000],
  thinkTime: [3, 600],
  mfu: [0.15, 0.8],
  mbu: [0.2, 0.9],
  overheadPerGpuGB: [0.5, 8],
  exactHit: [0, 1],
  semanticHit: [0, 1],
  prefixHit: [0, 1],
  prefixShare: [0, 1],
  targetTtft: [100, 20000],
  targetTpot: [10, 300],
  inputPricePerM: [0, 30],
  outputPricePerM: [0, 100],
  cacheReadMult: [0, 1],
  cacheWriteMult: [0, 3],
};

function fromUrl(): Partial<EngineConfig> {
  if (typeof window === "undefined") return {};
  const params = new URLSearchParams(window.location.search);
  const out: Partial<EngineConfig> = {};
  for (const [key, [lo, hi]] of Object.entries(NUMERIC)) {
    const raw = params.get(key);
    if (raw !== null && !Number.isNaN(Number(raw))) {
      (out as Record<string, number>)[key] = clamp(Number(raw), lo, hi);
    }
  }
  for (const key of ["modelId", "quantId", "kvQuantId", "gpuId"]) {
    const raw = params.get(key);
    if (raw) (out as Record<string, string>)[key] = raw;
  }
  return out;
}

const initial: EngineConfig = { ...BASE_CONFIG, ...fromUrl() };

export const useStore = create<State>((set, get) => ({
  config: initial,
  lens: "overview",
  showMath: false,
  presetId: "enterprise",
  explain: null,
  onboardingOpen: !hasSeenOnboarding(),
  set: (key, value) => {
    const cfg = { ...get().config, [key]: value };
    if (key === "context") {
      cfg.outputTokens = Math.min(cfg.outputTokens, Math.max(32, cfg.context - 64));
    }
    set({ config: cfg, presetId: "custom" });
  },
  setLens: (lens) => set({ lens }),
  setShowMath: (showMath) => set({ showMath }),
  setExplain: (explain) => set({ explain }),
  openOnboarding: () => set({ onboardingOpen: true }),
  closeOnboarding: () => {
    rememberOnboarding();
    set({ onboardingOpen: false });
  },
  applyPreset: (id) =>
    set({ config: { ...presetById(id).config }, presetId: PRESETS.some((p) => p.id === id) ? id : "custom" }),
}));

export function useMetrics(): Metrics {
  const config = useStore((s) => s.config);
  return computeMetrics(config);
}
