import { describe, expect, it } from "vitest";
import { adviseTopology, computeMetrics, erlangC, scalingEfficiency } from "../src/lib/formulas";
import { GPUS } from "../src/lib/models";
import { BASE_CONFIG } from "../src/lib/presets";

const noCache = { ...BASE_CONFIG, exactHit: 0, semanticHit: 0, prefixHit: 0, prefixShare: 0 };

describe("memory model", () => {
  it("computes 70B weights at BF16", () => {
    const m = computeMetrics({ ...noCache, modelId: "llama70b", quantId: "bf16" });
    expect(m.weightsGB).toBeCloseTo(141.2, 1);
  });

  it("matches the known KV size for Llama 3 70B", () => {
    const bf16 = computeMetrics({ ...noCache, modelId: "llama70b", kvQuantId: "kv16" });
    expect(bf16.kvPerTokenMB).toBeCloseTo(0.32768, 5);
    const fp8 = computeMetrics({ ...noCache, modelId: "llama70b", kvQuantId: "kv8" });
    expect(fp8.kvPerTokenMB).toBeCloseTo(0.16384, 5);
  });

  it("does not fit a 70B BF16 on one 80 GB GPU", () => {
    const m = computeMetrics({ ...noCache, modelId: "llama70b", quantId: "bf16", gpuCount: 1 });
    expect(m.modelFitsSingle).toBe(false);
    expect(m.maxUsers).toBe(0);
    expect(m.fits).toBe(false);
  });

  it("finds roughly 159 users for 70B FP8 on 4xH100 at 8k", () => {
    const m = computeMetrics({ ...noCache, gpuCount: 4, context: 8192 });
    expect(m.maxUsers).toBeGreaterThan(150);
    expect(m.maxUsers).toBeLessThan(170);
  });

  it("models Qwen3.8 27B hybrid attention with 16 KV layers", () => {
    const m = computeMetrics({
      ...noCache,
      modelId: "qwen3_8_27b",
      quantId: "fp8",
      gpuId: "l40s",
      gpuCount: 1,
      context: 8192,
      users: 1,
    });
    expect(m.model.name).toBe("Qwen3.8 27B");
    expect(m.model.maxContext).toBe(262144);
    expect(m.kvPerTokenMB).toBeCloseTo(0.032768, 6);
    expect(m.maxUsers).toBeGreaterThan(30);
    expect(m.fits).toBe(true);
  });

  it("keeps the GPU catalogue at 12 GB or more", () => {
    expect(GPUS.every((gpu) => gpu.vramGB >= 12)).toBe(true);
  });

  it("fits Qwen3 8B FP8 on a 12 GB RTX 3060", () => {
    const m = computeMetrics({
      ...noCache,
      modelId: "qwen3_8b",
      quantId: "fp8",
      gpuId: "rtx3060",
      gpuCount: 1,
      context: 4096,
      users: 1,
    });
    expect(m.model.name).toBe("Qwen3 8B");
    expect(m.gpu.name).toBe("RTX 3060");
    expect(m.gpu.vramGB).toBe(12);
    expect(m.kvPerTokenMB).toBeCloseTo(0.073728, 6);
    expect(m.maxUsers).toBeGreaterThan(0);
    expect(m.fits).toBe(true);
  });

  it("counts a shared prefix once and stays internally consistent", () => {
    const m = computeMetrics({ ...BASE_CONFIG, gpuCount: 4, context: 8192 });
    expect(m.maxUsers).toBeGreaterThan(200);
    expect(m.maxUsers).toBeLessThan(225);
    expect(m.usedGB).toBeLessThanOrEqual(m.capacityGB);
  });

  it("charges every replica for a full copy of the weights", () => {
    const m = computeMetrics({
      ...noCache,
      modelId: "llama8b",
      quantId: "bf16",
      gpuId: "l40s",
      gpuCount: 2,
      context: 4096,
      users: 100,
    });
    expect(m.mode).toBe("replicas");
    expect(m.weightsInstalledGB).toBeCloseTo(m.weightsGB * 2, 1);
    expect(m.usedGB).toBeCloseTo(
      m.weightsInstalledGB + m.kvTotalGB + m.overheadGB,
      1,
    );
  });

  it("uses replicas when the model fits a single card", () => {
    const m = computeMetrics({
      ...noCache,
      modelId: "llama8b",
      quantId: "bf16",
      gpuId: "l40s",
      gpuCount: 2,
      context: 4096,
    });
    expect(m.mode).toBe("replicas");
    expect(m.maxUsers).toBeGreaterThan(80);
  });

  it("splits weights across GPUs in tensor parallel mode", () => {
    const m = computeMetrics({ ...noCache, gpuCount: 4 });
    expect(m.mode).toBe("tensor-parallel");
    expect(m.weightsPerGpuGB).toBeCloseTo(17.65, 1);
  });
});

describe("time model", () => {
  it("needs six H100s just for 405B FP8 weights", () => {
    const advice = adviseTopology({
      ...noCache,
      modelId: "llama405b",
      quantId: "fp8",
      gpuId: "h100",
      gpuCount: 8,
      context: 4096,
    });
    expect(advice.minWeights).toBe(6);
  });

  it("prefill grows faster than linearly with context", () => {
    const a = computeMetrics({ ...noCache, context: 8192, outputTokens: 64 });
    const b = computeMetrics({ ...noCache, context: 16384, outputTokens: 64 });
    expect(b.prefillMs).toBeGreaterThan(a.prefillMs * 2.15);
    expect(b.attentionShare).toBeGreaterThan(a.attentionShare);
  });

  it("longer context reduces concurrency", () => {
    const a = computeMetrics({ ...noCache, context: 4096 });
    const b = computeMetrics({ ...noCache, context: 32768 });
    expect(b.maxUsers).toBeLessThan(a.maxUsers / 6);
  });

  it("decode is bandwidth bound and amortizes with batching", () => {
    const m = computeMetrics({ ...noCache, gpuCount: 4, context: 8192 });
    expect(m.outputTps).toBeGreaterThan(m.tokPerSecUser);
    expect(m.tpotMs).toBeGreaterThan(5);
  });
});

describe("queueing", () => {
  it("matches the M/M/c waiting probability reference", () => {
    expect(erlangC(4, 2)).toBeCloseTo(0.1739, 3);
    expect(erlangC(1, 0.5)).toBeCloseTo(0.5, 5);
  });

  it("is stable at low load and explodes past capacity", () => {
    const calm = computeMetrics({ ...noCache, users: 20 });
    const busy = computeMetrics({ ...noCache, users: 5000 });
    expect(calm.ttftP95Ms).toBeLessThan(4000);
    expect(busy.ttftP95Ms).toBeGreaterThanOrEqual(59000);
    expect(calm.utilization).toBeLessThan(0.6);
  });

  it("removes response-cache hits from GPU load", () => {
    const cached = computeMetrics(BASE_CONFIG);
    const raw = computeMetrics({
      ...BASE_CONFIG,
      exactHit: 0,
      semanticHit: 0,
    });
    expect(cached.arrivalRate).toBeLessThan(raw.arrivalRate);
    expect(cached.batch).toBeLessThanOrEqual(raw.batch);
  });
});

describe("topology", () => {
  it("scaling efficiency declines with GPU count and worse links", () => {
    expect(scalingEfficiency(4, 900)).toBeGreaterThan(scalingEfficiency(8, 900));
    expect(scalingEfficiency(4, 900)).toBeGreaterThan(scalingEfficiency(4, 64));
  });

  it("recommends sharding for a model that cannot fit one card", () => {
    const advice = adviseTopology({ ...noCache, modelId: "llama70b", quantId: "bf16" });
    expect(advice.minWeights).toBe(2);
    expect(advice.mode).toBe("tensor-parallel");
  });

  it("recommends replicas for a small model under load", () => {
    const advice = adviseTopology({
      ...noCache,
      modelId: "llama8b",
      quantId: "bf16",
      gpuId: "l40s",
      gpuCount: 1,
      context: 2048,
      users: 200,
    });
    expect(advice.mode).toBe("replicas");
    expect(advice.recommended).toBeGreaterThan(1);
  });

  it("flags a user count that 64 GPUs still cannot hold", () => {
    const advice = adviseTopology({
      ...noCache,
      context: 32768,
      gpuCount: 2,
      users: 3000,
    });
    expect(advice.warnings.some((w) => w.includes("Even 64 GPUs only reach"))).toBe(
      true,
    );
  });
});
