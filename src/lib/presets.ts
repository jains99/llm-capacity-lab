import type { EngineConfig } from "./formulas";

export const BASE_CONFIG: EngineConfig = {
  modelId: "llama70b",
  quantId: "fp8",
  kvQuantId: "kv8",
  gpuId: "h100",
  gpuCount: 4,
  context: 8192,
  outputTokens: 300,
  users: 150,
  thinkTime: 30,
  mfu: 0.45,
  mbu: 0.65,
  overheadPerGpuGB: 2.5,
  exactHit: 0.12,
  semanticHit: 0.1,
  prefixHit: 0.75,
  prefixShare: 0.35,
  targetTtft: 1500,
  targetTpot: 50,
  inputPricePerM: 3,
  outputPricePerM: 15,
  cacheReadMult: 0.1,
  cacheWriteMult: 1.25,
};

export type Preset = {
  id: string;
  name: string;
  tagline: string;
  config: EngineConfig;
};

export const PRESETS: Preset[] = [
  {
    id: "enterprise",
    name: "Enterprise RAG",
    tagline: "70B · 8k context · 150 users",
    config: { ...BASE_CONFIG },
  },
  {
    id: "startup",
    name: "Startup chat",
    tagline: "8B · 4k · 2× L40S",
    config: {
      ...BASE_CONFIG,
      modelId: "llama8b",
      quantId: "bf16",
      kvQuantId: "kv16",
      gpuId: "l40s",
      gpuCount: 2,
      context: 4096,
      outputTokens: 250,
      users: 100,
    },
  },
  {
    id: "longdoc",
    name: "Long-doc analyst",
    tagline: "70B · 128k · 8× H100",
    config: {
      ...BASE_CONFIG,
      modelId: "llama70b",
      quantId: "fp8",
      gpuId: "h100",
      gpuCount: 8,
      context: 131072,
      outputTokens: 500,
      users: 8,
      thinkTime: 60,
      prefixShare: 0.08,
      prefixHit: 0.4,
      targetTtft: 30000,
    },
  },
  {
    id: "overloaded",
    name: "Overloaded",
    tagline: "70B · 32k · 3,000 users",
    config: {
      ...BASE_CONFIG,
      gpuCount: 2,
      context: 32768,
      outputTokens: 400,
      users: 3000,
    },
  },
  {
    id: "budget",
    name: "Budget single GPU",
    tagline: "Mistral 7B · INT4 · 1× L4",
    config: {
      ...BASE_CONFIG,
      modelId: "mistral7b",
      quantId: "int4",
      kvQuantId: "kv8",
      gpuId: "l4",
      gpuCount: 1,
      context: 8192,
      outputTokens: 250,
      users: 120,
      prefixShare: 0.3,
    },
  },
  {
    id: "frontier",
    name: "Frontier",
    tagline: "405B · FP8 · 8× B200",
    config: {
      ...BASE_CONFIG,
      modelId: "llama405b",
      quantId: "fp8",
      gpuId: "b200",
      gpuCount: 8,
      context: 32768,
      outputTokens: 400,
      users: 200,
      targetTpot: 40,
    },
  },
];

export const presetById = (id: string) =>
  PRESETS.find((p) => p.id === id) ?? PRESETS[0];
