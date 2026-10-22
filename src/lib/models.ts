export type ModelSpec = {
  id: string;
  name: string;
  short: string;
  paramsB: number;
  activeB: number;
  layers: number;
  hidden: number;
  heads: number;
  kvHeads: number;
  headDim: number;
  /** Layers with a token-wise KV cache. Hybrid models can use fewer. */
  kvLayers?: number;
  /** Full-attention layers used for quadratic prefill FLOPs. */
  attentionLayers?: number;
  maxContext: number;
  kind: "dense" | "moe";
  note: string;
};

/**
 * Architecture figures from the published model cards.
 * paramsB is total parameters; activeB is what each token touches (MoE).
 */
export const MODELS: ModelSpec[] = [
  {
    id: "llama3_2_3b",
    name: "Llama 3.2 3B",
    short: "3B",
    paramsB: 3.21,
    activeB: 3.21,
    layers: 28,
    hidden: 3072,
    heads: 24,
    kvHeads: 8,
    headDim: 128,
    maxContext: 131072,
    kind: "dense",
    note: "A small Llama that fits a 12 GB card in BF16 with room for KV.",
  },
  {
    id: "qwen3_4b",
    name: "Qwen3 4B",
    short: "4B",
    paramsB: 4.0,
    activeB: 4.0,
    layers: 36,
    hidden: 2560,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    maxContext: 131072,
    kind: "dense",
    note: "Qwen's small dense model. FP8 leaves useful headroom on a 12 GB card.",
  },
  {
    id: "mistral7b",
    name: "Mistral 7B",
    short: "7B",
    paramsB: 7.24,
    activeB: 7.24,
    layers: 32,
    hidden: 4096,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    maxContext: 32768,
    kind: "dense",
    note: "Small, fast, cheap. The best case for a single GPU.",
  },
  {
    id: "llama8b",
    name: "Llama 3.1 8B",
    short: "8B",
    paramsB: 8.03,
    activeB: 8.03,
    layers: 32,
    hidden: 4096,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    maxContext: 131072,
    kind: "dense",
    note: "The default workhorse. Fits on one 24 GB card in FP8.",
  },
  {
    id: "qwen3_8b",
    name: "Qwen3 8B",
    short: "8B",
    paramsB: 8.2,
    activeB: 8.2,
    layers: 36,
    hidden: 4096,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    maxContext: 131072,
    kind: "dense",
    note: "The Qwen3 workhorse. FP8 fits a 12 GB card; BF16 wants 24 GB.",
  },
  {
    id: "qwen3_14b",
    name: "Qwen3 14B",
    short: "14B",
    paramsB: 14.8,
    activeB: 14.8,
    layers: 40,
    hidden: 5120,
    heads: 40,
    kvHeads: 8,
    headDim: 128,
    maxContext: 131072,
    kind: "dense",
    note: "A quality step up from 8B. FP8 fits one 24 GB card; BF16 needs 48 GB.",
  },
  {
    id: "qwen3_8_27b",
    name: "Qwen3.8 27B",
    short: "27B",
    paramsB: 27.78,
    activeB: 27.78,
    layers: 64,
    hidden: 5120,
    heads: 24,
    kvHeads: 4,
    headDim: 256,
    kvLayers: 16,
    attentionLayers: 16,
    maxContext: 262144,
    kind: "dense",
    note: "Hybrid attention: 16 full-attention KV layers + 48 linear layers. Native 262k context.",
  },
  {
    id: "qwen3_30b_a3b",
    name: "Qwen3 30B-A3B",
    short: "30B-A3B",
    paramsB: 30.5,
    activeB: 3.3,
    layers: 48,
    hidden: 2048,
    heads: 32,
    kvHeads: 4,
    headDim: 128,
    maxContext: 131072,
    kind: "moe",
    note: "30B of weights, 3B of compute per token. Memory is the constraint, not FLOPs.",
  },
  {
    id: "qwen32b",
    name: "Qwen3 32B",
    short: "32B",
    paramsB: 32.8,
    activeB: 32.8,
    layers: 64,
    hidden: 5120,
    heads: 64,
    kvHeads: 8,
    headDim: 128,
    maxContext: 131072,
    kind: "dense",
    note: "Big quality jump per GPU; needs two 24 GB cards or one 48 GB.",
  },
  {
    id: "mixtral",
    name: "Mixtral 8x7B",
    short: "MoE",
    paramsB: 46.7,
    activeB: 12.9,
    layers: 32,
    hidden: 4096,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    maxContext: 32768,
    kind: "moe",
    note: "46.7B of memory for 12.9B of compute. Memory ≠ FLOPs.",
  },
  {
    id: "llama70b",
    name: "Llama 3.3 70B",
    short: "70B",
    paramsB: 70.6,
    activeB: 70.6,
    layers: 80,
    hidden: 8192,
    heads: 64,
    kvHeads: 8,
    headDim: 128,
    maxContext: 131072,
    kind: "dense",
    note: "141 GB in BF16 — no single 80 GB card holds it. FP8 is the minimum, and leaves little room for KV.",
  },
  {
    id: "llama405b",
    name: "Llama 3.1 405B",
    short: "405B",
    paramsB: 405,
    activeB: 405,
    layers: 126,
    hidden: 16384,
    heads: 128,
    kvHeads: 8,
    headDim: 128,
    maxContext: 131072,
    kind: "dense",
    note: "810 GB in BF16. Multi-node territory even in FP8.",
  },
];

export const modelById = (id: string): ModelSpec =>
  MODELS.find((m) => m.id === id) ?? MODELS.find((m) => m.id === "llama70b") ?? MODELS[0];

export type QuantSpec = {
  id: string;
  label: string;
  weightBytes: number;
  kvBytes: number;
  note: string;
};

export const QUANTS: QuantSpec[] = [
  {
    id: "bf16",
    label: "BF16",
    weightBytes: 2,
    kvBytes: 2,
    note: "Reference quality. 2 bytes per weight and per KV element.",
  },
  {
    id: "fp8",
    label: "FP8",
    weightBytes: 1,
    kvBytes: 1,
    note: "The serving default. Half the memory, near-lossless for most models.",
  },
  {
    id: "int4",
    label: "INT4",
    weightBytes: 0.5,
    kvBytes: 1,
    note: "Smallest weights, measurable quality risk. KV stays FP8.",
  },
];

export const quantById = (id: string): QuantSpec =>
  QUANTS.find((q) => q.id === id) ?? QUANTS[1];

export const KV_QUANTS: { id: string; label: string; bytes: number }[] = [
  { id: "kv16", label: "BF16", bytes: 2 },
  { id: "kv8", label: "FP8", bytes: 1 },
];

export type GpuSpec = {
  id: string;
  name: string;
  short: string;
  vramGB: number;
  bwGBs: number;
  tflops: number;
  linkGBs: number;
  linkName: string;
  pricePerHour: number;
};

/**
 * VRAM in decimal GB, bandwidth in GB/s, dense tensor TFLOPS (vendor peak).
 * Prices are illustrative on-demand rates so cost is comparable, not a quote.
 */
export const GPUS: GpuSpec[] = [
  {
    id: "t4",
    name: "NVIDIA T4",
    short: "T4",
    vramGB: 16,
    bwGBs: 320,
    tflops: 65,
    linkGBs: 32,
    linkName: "PCIe 3.0",
    pricePerHour: 0.35,
  },
  {
    id: "rtx3060",
    name: "RTX 3060",
    short: "3060",
    vramGB: 12,
    bwGBs: 360,
    tflops: 25.6,
    linkGBs: 64,
    linkName: "PCIe 4.0",
    pricePerHour: 0.25,
  },
  {
    id: "rtx3090",
    name: "RTX 3090",
    short: "3090",
    vramGB: 24,
    bwGBs: 936,
    tflops: 142,
    linkGBs: 64,
    linkName: "PCIe 4.0",
    pricePerHour: 0.55,
  },
  {
    id: "rtx4060ti",
    name: "RTX 4060 Ti",
    short: "4060 Ti",
    vramGB: 16,
    bwGBs: 288,
    tflops: 44.2,
    linkGBs: 64,
    linkName: "PCIe 4.0",
    pricePerHour: 0.35,
  },
  {
    id: "rtx4090",
    name: "RTX 4090",
    short: "4090",
    vramGB: 24,
    bwGBs: 1008,
    tflops: 165,
    linkGBs: 64,
    linkName: "PCIe 4.0",
    pricePerHour: 0.7,
  },
  {
    id: "rtx5090",
    name: "RTX 5090",
    short: "5090",
    vramGB: 32,
    bwGBs: 1792,
    tflops: 209.5,
    linkGBs: 64,
    linkName: "PCIe 5.0",
    pricePerHour: 1.1,
  },
  {
    id: "a6000",
    name: "RTX A6000",
    short: "A6000",
    vramGB: 48,
    bwGBs: 768,
    tflops: 155,
    linkGBs: 64,
    linkName: "PCIe 4.0",
    pricePerHour: 1.2,
  },
  {
    id: "l4",
    name: "NVIDIA L4",
    short: "L4",
    vramGB: 24,
    bwGBs: 300,
    tflops: 121,
    linkGBs: 64,
    linkName: "PCIe 4.0",
    pricePerHour: 0.8,
  },
  {
    id: "a10g",
    name: "NVIDIA A10G",
    short: "A10G",
    vramGB: 24,
    bwGBs: 600,
    tflops: 125,
    linkGBs: 64,
    linkName: "PCIe 4.0",
    pricePerHour: 1.0,
  },
  {
    id: "l40s",
    name: "NVIDIA L40S",
    short: "L40S",
    vramGB: 48,
    bwGBs: 864,
    tflops: 183,
    linkGBs: 64,
    linkName: "PCIe 4.0",
    pricePerHour: 1.8,
  },
  {
    id: "a100",
    name: "NVIDIA A100",
    short: "A100",
    vramGB: 80,
    bwGBs: 2039,
    tflops: 312,
    linkGBs: 600,
    linkName: "NVLink 3",
    pricePerHour: 2.5,
  },
  {
    id: "h100",
    name: "NVIDIA H100",
    short: "H100",
    vramGB: 80,
    bwGBs: 3350,
    tflops: 989,
    linkGBs: 900,
    linkName: "NVLink 4",
    pricePerHour: 3.5,
  },
  {
    id: "h200",
    name: "NVIDIA H200",
    short: "H200",
    vramGB: 141,
    bwGBs: 4800,
    tflops: 989,
    linkGBs: 900,
    linkName: "NVLink 4",
    pricePerHour: 5.0,
  },
  {
    id: "b200",
    name: "NVIDIA B200",
    short: "B200",
    vramGB: 192,
    bwGBs: 8000,
    tflops: 2250,
    linkGBs: 1800,
    linkName: "NVLink 5",
    pricePerHour: 7.0,
  },
  {
    id: "mi300x",
    name: "AMD MI300X",
    short: "MI300X",
    vramGB: 192,
    bwGBs: 5300,
    tflops: 1300,
    linkGBs: 896,
    linkName: "Infinity Fabric",
    pricePerHour: 3.0,
  },
];

export const gpuById = (id: string): GpuSpec =>
  GPUS.find((g) => g.id === id) ?? GPUS.find((g) => g.id === "h100") ?? GPUS[0];
