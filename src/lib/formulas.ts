import { clamp } from "./format";
import { gpuById, KV_QUANTS, modelById, quantById } from "./models";

export type EngineConfig = {
  modelId: string;
  quantId: string;
  kvQuantId: string;
  gpuId: string;
  gpuCount: number;
  context: number;
  outputTokens: number;
  users: number;
  thinkTime: number;
  mfu: number;
  mbu: number;
  overheadPerGpuGB: number;
  exactHit: number;
  semanticHit: number;
  prefixHit: number;
  prefixShare: number;
  targetTtft: number;
  targetTpot: number;
  inputPricePerM: number;
  outputPricePerM: number;
  cacheReadMult: number;
  cacheWriteMult: number;
};

export type Bottleneck =
  | "vram"
  | "weights-bandwidth"
  | "kv-bandwidth"
  | "prefill-compute"
  | "queue"
  | "balanced";

export type Metrics = {
  model: ReturnType<typeof modelById>;
  gpu: ReturnType<typeof gpuById>;
  quant: ReturnType<typeof quantById>;
  n: number;
  mode: "single" | "replicas" | "tensor-parallel";
  context: number;
  promptTokens: number;
  outputTokens: number;

  weightsGB: number;
  weightsInstalledGB: number;
  weightsPerGpuGB: number;
  kvPerTokenMB: number;
  kvPerUserGB: number;
  kvTotalGB: number;
  overheadGB: number;
  capacityGB: number;
  usedGB: number;
  freeGB: number;
  usedFrac: number;
  fits: boolean;
  modelFitsSingle: boolean;
  maxUsers: number;
  sharedSaving: number;

  prefillMs: number;
  decodeMs: number;
  tpotMs: number;
  tokPerSecUser: number;
  outputTps: number;
  batch: number;
  batchPerGpu: number;

  arrivalRate: number;
  concurrency: number;
  utilization: number;
  queueWaitP50Ms: number;
  queueWaitP95Ms: number;
  ttftP50Ms: number;
  ttftP95Ms: number;
  queueLength: number;

  responseHitRate: number;
  effectivePromptTokens: number;
  prefillSavedFrac: number;
  cacheSavedPct: number;
  prefixHit: number;
  prefixShare: number;

  costPerHour: number;
  costPerMillionOut: number;
  providerBlendedMult: number;

  bottleneck: Bottleneck;
  bottleneckNote: string;
  attentionShare: number;
  arithmeticIntensity: number;
};

const GB = 1e9;

/** Erlang C: probability an arriving request has to wait, for M/M/c. */
export function erlangC(c: number, offeredLoad: number): number {
  if (c <= 0) return 1;
  if (offeredLoad >= c) return 1;
  if (offeredLoad <= 0) return 0;
  let term = 1;
  let sum = 1;
  for (let k = 1; k < c; k++) {
    term *= offeredLoad / k;
    sum += term;
    if (!isFinite(sum)) return 1;
  }
  const last = (term * offeredLoad) / c;
  const rho = offeredLoad / c;
  const top = last / (1 - rho);
  return top / (sum + top);
}

/** Ideal speedup efficiency for tensor parallelism, decaying with link quality. */
export function scalingEfficiency(n: number, linkGBs: number): number {
  if (n <= 1) return 1;
  const nvlink = linkGBs >= 400;
  const table = nvlink ? [0, 1, 0.94, 0.88, 0.83, 0.79, 0.76, 0.73, 0.7] : [0, 1, 0.86, 0.7, 0.55, 0.42, 0.32, 0.24, 0.18];
  if (n <= 8) {
    const lo = Math.floor(n);
    const hi = Math.min(8, lo + 1);
    const t = n - lo;
    return table[lo] * (1 - t) + table[hi] * t;
  }
  return Math.max(0.08, table[8] * (8 / n) ** 0.55);
}

export function computeMetrics(c: EngineConfig): Metrics {
  const model = modelById(c.modelId);
  const quant = quantById(c.quantId);
  const kvQuant = KV_QUANTS.find((q) => q.id === c.kvQuantId) ?? KV_QUANTS[1];
  const gpu = gpuById(c.gpuId);
  const n = Math.max(1, Math.round(c.gpuCount));
  const context = clamp(Math.round(c.context), 256, model.maxContext);
  const outputTokens = clamp(Math.round(c.outputTokens), 16, Math.max(32, context - 64));
  const promptTokens = Math.max(64, context - outputTokens);
  const users = Math.max(1, Math.round(c.users));
  const thinkTime = Math.max(2, c.thinkTime);

  const weightsGB = model.paramsB * quant.weightBytes;
  const kvLayers = model.kvLayers ?? model.layers;
  const attentionLayers = model.attentionLayers ?? model.layers;
  const kvPerTokenBytes = 2 * kvLayers * model.kvHeads * model.headDim * kvQuant.bytes;
  const kvPerTokenMB = kvPerTokenBytes / 1e6;
  const kvPerUserGB = (kvPerTokenBytes * context) / GB;

  const sharedSaving = clamp(c.prefixShare * c.prefixHit, 0, 0.95);
  const effectiveN = (1 - sharedSaving) * users + sharedSaving;

  const usablePerGpu = gpu.vramGB * 0.92;
  const overheadGB = c.overheadPerGpuGB * n;
  const capacityGB = usablePerGpu * n;

  const modelFitsSingle = weightsGB + c.overheadPerGpuGB <= usablePerGpu;
  // Replicas avoid every all-reduce, but duplicate the weights. Prefer them only
  // when they leave meaningful KV headroom *and* can carry the requested load.
  // Otherwise shard the weights, which keeps one copy of them in total.
  const replicaFreeFrac =
    (usablePerGpu - weightsGB - c.overheadPerGpuGB) / usablePerGpu;
  const sharedKVGB = kvPerUserGB * sharedSaving;
  const privateKVGB = kvPerUserGB * (1 - sharedSaving);
  const maxUsersFor = (layout: "replicas" | "tensor-parallel") => {
    if (layout === "replicas") {
      if (!modelFitsSingle) return 0;
      const perReplicaAvail = usablePerGpu - weightsGB - c.overheadPerGpuGB;
      const perReplica =
        privateKVGB > 0
          ? Math.floor((perReplicaAvail - sharedKVGB) / privateKVGB)
          : Infinity;
      return Math.max(0, perReplica) * n;
    }
    const avail = capacityGB - weightsGB - overheadGB;
    const raw =
      privateKVGB > 0 ? (avail - sharedKVGB) / privateKVGB : Infinity;
    return Math.max(0, Math.floor(raw));
  };
  const replicaMaxUsers = maxUsersFor("replicas");
  const tpMaxUsers = maxUsersFor("tensor-parallel");
  const canReplicate = modelFitsSingle && replicaFreeFrac >= 0.15;
  // If replicas can serve the requested users, take the no-all-reduce route.
  // If they cannot, sharding usually buys the capacity the replicas duplicated
  // away. The capacity comparison is the fallback when neither layout fits.
  const mode: Metrics["mode"] =
    n === 1
      ? "single"
      : canReplicate && (replicaMaxUsers >= users || replicaMaxUsers >= tpMaxUsers)
        ? "replicas"
        : "tensor-parallel";
  const weightsPerGpuGB = mode === "tensor-parallel" ? weightsGB / n : weightsGB;
  const weightsInstalledGB = mode === "replicas" ? weightsGB * n : weightsGB;

  // KV is sharded by attention heads in TP, and split across replicas otherwise.
  // Replicas cannot share blocks with each other, so the shared prefix is
  // duplicated once per replica; TP holds a single shared copy.
  const kvTotalGB =
    mode === "replicas"
      ? kvPerUserGB * ((1 - sharedSaving) * users + sharedSaving * n)
      : kvPerUserGB * effectiveN;
  const usedGB = weightsInstalledGB + kvTotalGB + overheadGB;
  const freeGB = capacityGB - usedGB;
  const fits = freeGB >= 0 && weightsPerGpuGB + c.overheadPerGpuGB <= usablePerGpu;

  let maxUsers =
    mode === "tensor-parallel" ? tpMaxUsers : maxUsersFor("replicas");
  maxUsers = Number.isFinite(maxUsers) ? Math.min(maxUsers, 100_000) : 100_000;

  const eff = mode === "tensor-parallel" ? scalingEfficiency(n, gpu.linkGBs) : 1;

  // --- prefill -----------------------------------------------------------
  const effectivePrompt = promptTokens * (1 - c.prefixShare * c.prefixHit);
  const attnFlops = 4 * attentionLayers * effectivePrompt * effectivePrompt * model.hidden;
  const linearFlops = 2 * model.activeB * 1e9 * effectivePrompt;
  const prefillFlops = attnFlops + linearFlops;
  const computeFactor = mode === "tensor-parallel" ? n * eff : 1;
  const prefillMs =
    (prefillFlops / (computeFactor * gpu.tflops * 1e12 * c.mfu)) * 1000;

  // --- decode ------------------------------------------------------------
  // --- traffic -----------------------------------------------------------
  // Response-cache hits never reach the GPU, so they do not consume KV slots
  // or decode capacity. Prefix reuse is still handled inside prefill below.
  const responseHitRate = 1 - (1 - c.exactHit) * (1 - c.semanticHit);
  const gpuUsers = users * (1 - responseHitRate);
  const lambda = gpuUsers / thinkTime;
  const serviceSecondsAt = (tpotMs: number) => prefillMs / 1000 + (outputTokens * tpotMs) / 1000;

  const decodeMsFor = (batch: number): number => {
    const b = Math.max(1, batch);
    const kvBytes = kvPerTokenBytes * context * b;
    let ms: number;
    if (mode === "tensor-parallel") {
      const perGpuBytes = (weightsGB * GB + kvBytes) / n;
      ms = (perGpuBytes / (gpu.bwGBs * GB * c.mbu * eff)) * 1000;
      const arBytes =
        8 * model.layers * model.hidden * b * ((n - 1) / n);
      ms += (arBytes / (gpu.linkGBs * GB)) * 1000;
    } else {
      // Each replica owns its share of the batch and a full copy of the weights.
      const perGpuKv = kvBytes / n;
      ms = ((weightsGB * GB + perGpuKv) / (gpu.bwGBs * GB * c.mbu)) * 1000;
    }
    return ms;
  };

  const slots = Math.max(0, maxUsers);
  let batch = Math.max(1, Math.min(slots || 1, users));
  let tpotMs = decodeMsFor(batch);
  for (let i = 0; i < 3; i++) {
    const service = serviceSecondsAt(tpotMs);
    const conc = lambda * service;
    const next = Math.max(1, Math.min(slots || 1, conc));
    batch = next;
    tpotMs = decodeMsFor(batch);
  }
  const decodeMs = tpotMs;
  const batchPerGpu = mode === "replicas" ? batch / n : batch;
  const tokPerSecUser = 1000 / tpotMs;
  const outputTps = (batch / tpotMs) * 1000;

  // --- queue -------------------------------------------------------------
  const serviceSeconds = serviceSecondsAt(tpotMs);
  const mu = 1 / Math.max(0.001, serviceSeconds);
  const offered = lambda / mu;
  const util = slots > 0 ? offered / slots : 1;
  const utilization = clamp(util, 0, 1);
  const concurrency = slots > 0 ? Math.min(offered, slots) : 0;
  const C = slots > 0 ? erlangC(slots, Math.min(offered, slots * 0.999999)) : 1;
  const drain = slots * mu - lambda;

  const waitP = (p: number) => {
    if (slots <= 0 || drain <= 0.0001) return 60_000;
    const inside = C / (1 - p);
    if (inside <= 1) return 0;
    return Math.min(60, Math.log(inside) / drain) * 1000;
  };
  const queueWaitP50Ms = waitP(0.5);
  const queueWaitP95Ms = waitP(0.95);
  const ttftP50Ms = prefillMs + queueWaitP50Ms;
  const ttftP95Ms = prefillMs + queueWaitP95Ms;
  const queueLength = lambda * (queueWaitP50Ms / 1000);

  // --- caching -----------------------------------------------------------
  const effectivePromptTokens = promptTokens * (1 - c.prefixShare * c.prefixHit);
  const prefillSavedFrac = clamp(1 - effectivePromptTokens / promptTokens, 0, 1);
  const cacheSavedPct = clamp(responseHitRate + (1 - responseHitRate) * prefillSavedFrac, 0, 0.99);

  // Steady-state hosted-cache blend. The write premium is amortised over ~10
  // reuses; read-heavy traffic approaches cacheReadMult. OpenAI has no write
  // premium (cacheWriteMult = 1), Anthropic charges one on first write.
  const sharedFrac = c.prefixShare * c.prefixHit;
  const providerBlendedMult =
    (1 - sharedFrac) +
    sharedFrac * c.cacheReadMult +
    sharedFrac * (c.cacheWriteMult - 1) * 0.1;

  // --- cost --------------------------------------------------------------
  const costPerHour = gpu.pricePerHour * n;
  const servedTps = outputTps / Math.max(0.01, 1 - responseHitRate);
  const costPerMillionOut =
    servedTps > 0 ? (costPerHour / 3600 / servedTps) * 1e6 : Infinity;

  // --- diagnosis ---------------------------------------------------------
  const kvBytes = kvPerTokenBytes * context * Math.max(1, batch);
  const weightBytes = weightsGB * GB;
  const attentionShare = attnFlops / Math.max(1, prefillFlops);
  const arithmeticIntensity = (2 * model.activeB * 1e9) / Math.max(1, weightBytes + kvBytes);

  let bottleneck: Bottleneck = "balanced";
  let bottleneckNote = "";
  if (!fits || slots === 0) {
    bottleneck = "vram";
    bottleneckNote = !modelFitsSingle
      ? "The weights alone exceed one GPU. You need a sharded model before anything else."
      : "Weights fit, but there is no room left for the KV cache at this context.";
  } else if (utilization > 0.9) {
    bottleneck = "queue";
    bottleneckNote = "Arrivals exceed service capacity. p95 is about to fall off a cliff.";
  } else if (kvBytes > weightBytes * 1.5) {
    bottleneck = "kv-bandwidth";
    bottleneckNote = "Decode is re-reading more KV cache than weights. Long context has made memory bandwidth the wall.";
  } else if (prefillMs > outputTokens * tpotMs) {
    bottleneck = "prefill-compute";
    bottleneckNote = "Prefill dominates. Bigger prompts mean more FLOPs, and attention is quadratic.";
  } else if (utilization > 0.75) {
    bottleneck = "queue";
    bottleneckNote = "You are close to capacity. Small traffic increases cost a lot of tail latency.";
  } else if (kvBytes > weightBytes * 0.4) {
    bottleneck = "kv-bandwidth";
    bottleneckNote = "KV traffic is a meaningful share of every decode step.";
  } else {
    bottleneck = "weights-bandwidth";
    bottleneckNote = "Decode is streaming the weights. Larger batches amortize this almost for free.";
  }

  return {
    model,
    gpu,
    quant,
    n,
    mode,
    context,
    promptTokens,
    outputTokens,
    weightsGB,
    weightsInstalledGB,
    weightsPerGpuGB,
    kvPerTokenMB,
    kvPerUserGB,
    kvTotalGB,
    overheadGB,
    capacityGB,
    usedGB,
    freeGB,
    usedFrac: capacityGB > 0 ? usedGB / capacityGB : 1,
    fits,
    modelFitsSingle,
    maxUsers,
    sharedSaving,
    prefillMs,
    decodeMs,
    tpotMs,
    tokPerSecUser,
    outputTps,
    batch,
    batchPerGpu,
    arrivalRate: lambda,
    concurrency,
    utilization,
    queueWaitP50Ms,
    queueWaitP95Ms,
    ttftP50Ms,
    ttftP95Ms,
    queueLength,
    responseHitRate,
    effectivePromptTokens,
    prefillSavedFrac,
    cacheSavedPct,
    prefixHit: c.prefixHit,
    prefixShare: c.prefixShare,
    costPerHour,
    costPerMillionOut,
    providerBlendedMult,
    bottleneck,
    bottleneckNote,
    attentionShare,
    arithmeticIntensity,
  };
}

export type TopologyAdvice = {
  minWeights: number;
  minCapacity: number;
  minDecode: number;
  minPrefill: number;
  recommended: number;
  mode: "single" | "replicas" | "tensor-parallel" | "multi-node";
  reasons: { label: string; detail: string; gpus: number }[];
  warnings: string[];
  efficiency: number;
  ttftAt: number;
  tpotAt: number;
  maxUsersAt: number;
};

export function adviseTopology(c: EngineConfig): TopologyAdvice {
  const base = computeMetrics({ ...c, gpuCount: 1 });
  const gpu = base.gpu;
  const usable = gpu.vramGB * 0.92;
  const MAX_GPUS = 64;

  const minWeights = Math.max(
    1,
    Math.ceil(base.weightsGB / Math.max(1, usable - c.overheadPerGpuGB)),
  );

  let minCapacity = 1;
  if (base.maxUsers < c.users) {
    minCapacity = MAX_GPUS;
    for (let k = 2; k <= MAX_GPUS; k++) {
      if (computeMetrics({ ...c, gpuCount: k }).maxUsers >= c.users) {
        minCapacity = k;
        break;
      }
    }
  }

  let minDecode = base.tpotMs <= c.targetTpot ? 1 : 0;
  let minPrefill = base.ttftP95Ms <= c.targetTtft && base.fits ? 1 : 0;
  for (let k = 2; k <= MAX_GPUS; k++) {
    const m = computeMetrics({ ...c, gpuCount: k });
    if (minDecode === 0 && m.tpotMs <= c.targetTpot) minDecode = k;
    if (minPrefill === 0 && m.ttftP95Ms <= c.targetTtft && m.fits) minPrefill = k;
    if (minDecode !== 0 && minPrefill !== 0) break;
  }

  const atMax = computeMetrics({ ...c, gpuCount: MAX_GPUS });
  const capacityUnreachable = minCapacity === MAX_GPUS && atMax.maxUsers < c.users;
  const decodeUnreachable = minDecode === 0;
  const prefillUnreachable = minPrefill === 0;

  // If the weights leave real room for KV, more copies beat one sharded copy.
  // A model that only *just* fits leaves no KV room, so replicas are a trap.
  const canReplicate =
    base.modelFitsSingle &&
    (usable - base.weightsGB - c.overheadPerGpuGB) / usable >= 0.15;
  const capacityDriven = Math.max(minWeights, minCapacity);
  let recommended = canReplicate
    ? Math.max(1, capacityDriven, minDecode, minPrefill)
    : Math.max(minWeights, capacityDriven, minDecode, minPrefill);
  recommended = Math.min(MAX_GPUS, recommended);

  let final = base;
  for (let k = 1; k <= MAX_GPUS; k++) {
    if (k >= recommended) {
      final = computeMetrics({ ...c, gpuCount: k });
      break;
    }
  }

  const mode: TopologyAdvice["mode"] =
    recommended === 1
      ? "single"
      : final.mode === "replicas"
        ? "replicas"
        : recommended <= 8
          ? "tensor-parallel"
          : "multi-node";

  const reasons: TopologyAdvice["reasons"] = [];
  if (minWeights > 1) {
    reasons.push({
      label: "Weights don't fit",
      detail: `${base.weightsGB.toFixed(0)} GB of weights + ${c.overheadPerGpuGB.toFixed(1)} GB runtime against ${usable.toFixed(0)} GB usable per ${gpu.short}.`,
      gpus: minWeights,
    });
  }
  if (minCapacity > 1) {
    reasons.push({
      label: capacityUnreachable
        ? "Capacity beyond 64 GPUs"
        : final.mode === "replicas"
          ? "KV capacity"
          : "KV + weights capacity",
      detail: `Serving ${c.users.toLocaleString()} users at ${(base.context / 1024).toFixed(0)}k context needs about ${minCapacity} GPU${minCapacity > 1 ? "s" : ""} of memory.`,
      gpus: minCapacity,
    });
  }
  if (minDecode !== 1) {
    reasons.push({
      label: decodeUnreachable ? "TPOT unreachable" : "Decode too slow",
      detail: `Target TPOT is ${c.targetTpot} ms; one ${gpu.short} gives ${base.tpotMs.toFixed(0)} ms. Sharding spreads the weight read.`,
      gpus: decodeUnreachable ? MAX_GPUS : minDecode,
    });
  }
  if (minPrefill !== 1) {
    reasons.push({
      label: prefillUnreachable ? "TTFT unreachable" : "TTFT over SLO",
      detail: `Target TTFT is ${c.targetTtft} ms; one ${gpu.short} needs ${base.ttftP95Ms.toFixed(0)} ms at p95. Prefill is compute-bound.`,
      gpus: prefillUnreachable ? MAX_GPUS : minPrefill,
    });
  }

  const warnings: string[] = [];
  if (capacityUnreachable) {
    warnings.push(
      `Even ${MAX_GPUS} GPUs only reach ${atMax.maxUsers.toLocaleString()} KV slots at this context. The user count and context length have to come down, or the fleet has to cross nodes.`,
    );
  }
  if (decodeUnreachable) {
    warnings.push(
      final.mode === "replicas"
        ? `Replicas cannot reduce per-token latency: even ${MAX_GPUS} copies miss the ${c.targetTpot} ms TPOT target. Meeting it needs tensor parallelism, which pays an all-reduce.`
        : `Even ${MAX_GPUS}-way sharding misses the ${c.targetTpot} ms TPOT target. The interconnect or the model size is the wall.`,
    );
  }
  if (prefillUnreachable) {
    warnings.push(
      final.mode === "replicas"
        ? `Replicas do not split a single request's prefill: even ${MAX_GPUS} copies miss the ${c.targetTtft} ms TTFT target. Tensor parallelism, a shorter prompt, or a faster node is the fix.`
        : `Even ${MAX_GPUS} GPUs miss the ${c.targetTtft} ms TTFT target. Shorten the prompt, raise MFU, or use a faster node.`,
    );
  }
  if (final.mode === "tensor-parallel" && recommended > 1 && gpu.linkGBs < 100) {
    warnings.push(
      `Tensor parallelism over ${gpu.linkName} (${gpu.linkGBs} GB/s) pays an all-reduce on every layer, every token. Prefer a node with NVLink, or pipeline parallelism.`,
    );
  }
  if (recommended > 8) {
    warnings.push(
      "Past 8-way tensor parallelism you are crossing nodes. Expect pipeline parallelism or a multi-node fabric, and a real scaling penalty.",
    );
  }
  if (final.mode === "tensor-parallel" && base.model.kvHeads < recommended && recommended <= 8) {
    warnings.push(
      `This model has only ${base.model.kvHeads} KV heads, which limits clean tensor-parallel splits to ${base.model.kvHeads}.`,
    );
  }
  if (base.kvPerUserGB > usable) {
    warnings.push(
      `A single ${(base.context / 1024).toFixed(0)}k-token request needs ${base.kvPerUserGB.toFixed(1)} GB of KV — more than one ${gpu.short} can hold. Context itself forces sharding.`,
    );
  }
  if (final.mode === "replicas" && recommended > 1 && !decodeUnreachable) {
    warnings.push(
      prefillUnreachable
        ? "Replicas are the right shape for capacity and throughput, but they cannot fix the TTFT target. Tensor parallelism would only help latency."
        : "Replicas are the right move here: they add capacity and throughput without any all-reduce. Tensor parallelism would only help latency.",
    );
  }

  return {
    minWeights,
    minCapacity,
    minDecode,
    minPrefill,
    recommended,
    mode,
    reasons,
    warnings,
    efficiency:
      mode === "replicas" ? 1 : scalingEfficiency(recommended, gpu.linkGBs),
    ttftAt: final.ttftP95Ms,
    tpotAt: final.tpotMs,
    maxUsersAt: final.maxUsers,
  };
}
