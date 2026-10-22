import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  adviseTopology,
  computeMetrics,
  type EngineConfig,
  type Metrics,
  type TopologyAdvice,
} from "../src/lib/formulas";
import { GPUS, KV_QUANTS, MODELS, QUANTS } from "../src/lib/models";
import { BASE_CONFIG, PRESETS } from "../src/lib/presets";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPORT_DIR = resolve(ROOT, "reports");
const HTML_PATH = resolve(REPORT_DIR, "inference-matrix.html");
const LOG_PATH = resolve(REPORT_DIR, "inference-matrix.log");
const EPSILON = 1e-6;
const MAX_USERS = 100_000;

type CacheProfile = {
  name: string;
  exactHit: number;
  semanticHit: number;
  prefixHit: number;
  prefixShare: number;
};

type TestCase = {
  id: string;
  suite: string;
  name: string;
  config: EngineConfig;
  withAdvice?: boolean;
  custom?: (metrics: Metrics, advice: TopologyAdvice | undefined) => string[];
};

type TestResult = {
  id: string;
  suite: string;
  name: string;
  status: "PASS" | "FAIL";
  summary: string;
  failures: string[];
  durationMs: number;
};

const NO_CACHE: CacheProfile = {
  name: "no cache",
  exactHit: 0,
  semanticHit: 0,
  prefixHit: 0,
  prefixShare: 0,
};

const CACHE_PROFILES: CacheProfile[] = [
  NO_CACHE,
  {
    name: "default cache",
    exactHit: BASE_CONFIG.exactHit,
    semanticHit: BASE_CONFIG.semanticHit,
    prefixHit: BASE_CONFIG.prefixHit,
    prefixShare: BASE_CONFIG.prefixShare,
  },
  {
    name: "high cache",
    exactHit: 0.4,
    semanticHit: 0.3,
    prefixHit: 0.9,
    prefixShare: 0.6,
  },
  {
    name: "exact-heavy",
    exactHit: 0.6,
    semanticHit: 0.2,
    prefixHit: 0.1,
    prefixShare: 0.1,
  },
  {
    name: "prefix-heavy",
    exactHit: 0,
    semanticHit: 0,
    prefixHit: 0.95,
    prefixShare: 0.9,
  },
];

const cases: TestCase[] = [];
let sequence = 0;

function addCase(
  suite: string,
  name: string,
  config: EngineConfig,
  options: Pick<TestCase, "withAdvice" | "custom"> = {},
) {
  sequence += 1;
  cases.push({
    id: `TC-${String(sequence).padStart(4, "0")}`,
    suite,
    name,
    config: { ...config },
    ...options,
  });
}

function configWith(
  overrides: Partial<EngineConfig>,
  cache: CacheProfile = NO_CACHE,
): EngineConfig {
  return {
    ...BASE_CONFIG,
    ...cache,
    ...overrides,
  };
}

function contextFor(raw: number, modelMax: number): number {
  return Math.min(Math.max(256, Math.round(raw)), modelMax);
}

function contextLabel(context: number): string {
  return context >= 1024 ? `${Math.round(context / 1024)}k ctx` : `${context} ctx`;
}

function close(a: number, b: number, tolerance = EPSILON): boolean {
  return Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(a), Math.abs(b));
}

function expectedFits(metrics: Metrics, config: EngineConfig): boolean {
  const usable = metrics.gpu.vramGB * 0.92;
  return (
    metrics.freeGB >= 0 &&
    metrics.weightsPerGpuGB + config.overheadPerGpuGB <= usable + EPSILON
  );
}

function checkMetrics(metrics: Metrics, config: EngineConfig): string[] {
  const failures: string[] = [];
  const users = Math.max(1, Math.round(config.users));
  const n = Math.max(1, Math.round(config.gpuCount));
  const context = contextFor(config.context, metrics.model.maxContext);
  const maxOutput = Math.max(32, context - 64);
  const outputTokens = Math.min(Math.max(16, Math.round(config.outputTokens)), maxOutput);
  const shared = Math.min(0.95, Math.max(0, config.prefixShare * config.prefixHit));
  const expectedKvTotal =
    metrics.kvPerUserGB *
    (metrics.mode === "replicas"
      ? (1 - shared) * users + shared * n
      : (1 - shared) * users + shared);
  const expectedInstalledWeights =
    metrics.mode === "replicas" ? metrics.weightsGB * n : metrics.weightsGB;

  const fail = (condition: boolean, message: string) => {
    if (condition) failures.push(message);
  };
  const finite = (value: number, label: string) =>
    fail(!Number.isFinite(value), `${label} is not finite (${value})`);
  const positive = (value: number, label: string) => {
    finite(value, label);
    fail(value <= 0, `${label} must be positive (${value})`);
  };

  positive(metrics.weightsGB, "weightsGB");
  positive(metrics.kvPerTokenMB, "kvPerTokenMB");
  positive(metrics.kvPerUserGB, "kvPerUserGB");
  positive(metrics.capacityGB, "capacityGB");
  positive(metrics.outputTps, "outputTps");
  positive(metrics.prefillMs, "prefillMs");
  positive(metrics.tpotMs, "tpotMs");
  positive(metrics.tokPerSecUser, "tokPerSecUser");
  positive(metrics.costPerHour, "costPerHour");
  finite(metrics.usedGB, "usedGB");
  finite(metrics.freeGB, "freeGB");
  finite(metrics.maxUsers, "maxUsers");
  finite(metrics.ttftP50Ms, "ttftP50Ms");
  finite(metrics.ttftP95Ms, "ttftP95Ms");
  finite(metrics.queueWaitP50Ms, "queueWaitP50Ms");
  finite(metrics.queueWaitP95Ms, "queueWaitP95Ms");
  finite(metrics.costPerMillionOut, "costPerMillionOut");
  finite(metrics.arrivalRate, "arrivalRate");
  finite(metrics.concurrency, "concurrency");
  finite(metrics.queueLength, "queueLength");

  fail(metrics.n !== n, `n=${metrics.n}, expected ${n}`);
  fail(metrics.context !== context, `context=${metrics.context}, expected ${context}`);
  fail(
    metrics.outputTokens !== outputTokens,
    `outputTokens=${metrics.outputTokens}, expected ${outputTokens}`,
  );
  fail(metrics.maxUsers < 0, `maxUsers is negative (${metrics.maxUsers})`);
  fail(metrics.maxUsers > MAX_USERS, `maxUsers exceeds ${MAX_USERS} (${metrics.maxUsers})`);
  fail(metrics.usedGB < 0, `usedGB is negative (${metrics.usedGB})`);
  fail(metrics.utilization < -EPSILON || metrics.utilization > 1 + EPSILON, `utilization=${metrics.utilization}`);
  fail(
    metrics.responseHitRate < -EPSILON || metrics.responseHitRate > 1 + EPSILON,
    `responseHitRate=${metrics.responseHitRate}`,
  );
  fail(
    metrics.cacheSavedPct < -EPSILON || metrics.cacheSavedPct > 0.99 + EPSILON,
    `cacheSavedPct=${metrics.cacheSavedPct}`,
  );
  fail(
    metrics.attentionShare < -EPSILON || metrics.attentionShare > 1 + EPSILON,
    `attentionShare=${metrics.attentionShare}`,
  );
  fail(
    metrics.sharedSaving < -EPSILON || metrics.sharedSaving > 0.95 + EPSILON,
    `sharedSaving=${metrics.sharedSaving}`,
  );
  fail(
    metrics.ttftP95Ms < metrics.ttftP50Ms - EPSILON,
    `ttftP95=${metrics.ttftP95Ms} < ttftP50=${metrics.ttftP50Ms}`,
  );
  fail(
    metrics.ttftP95Ms < metrics.prefillMs - EPSILON,
    `ttftP95=${metrics.ttftP95Ms} < prefill=${metrics.prefillMs}`,
  );
  fail(
    metrics.queueWaitP95Ms < metrics.queueWaitP50Ms - EPSILON,
    `queue p95=${metrics.queueWaitP95Ms} < p50=${metrics.queueWaitP50Ms}`,
  );
  fail(metrics.batch < 1, `batch=${metrics.batch} is below 1`);
  fail(metrics.batch > Math.max(1, metrics.maxUsers), `batch=${metrics.batch} exceeds slots=${metrics.maxUsers}`);
  fail(metrics.bottleneckNote.length === 0, "bottleneckNote is empty");
  fail(metrics.modelFitsSingle !== (metrics.weightsGB + config.overheadPerGpuGB <= metrics.gpu.vramGB * 0.92), "modelFitsSingle is inconsistent");

  if (n === 1) {
    fail(metrics.mode !== "single", `one GPU must use single mode, got ${metrics.mode}`);
  } else {
    fail(metrics.mode === "single", `multi-GPU config must not use single mode`);
  }
  fail(metrics.mode === "replicas" && !metrics.modelFitsSingle, "replicas used while weights do not fit one GPU");

  fail(!close(metrics.weightsGB, metrics.model.paramsB * metrics.quant.weightBytes, 1e-9), "weightsGB does not match model × precision");
  fail(!close(metrics.weightsInstalledGB, expectedInstalledWeights), `weightsInstalledGB=${metrics.weightsInstalledGB}, expected ${expectedInstalledWeights}`);
  fail(!close(metrics.weightsPerGpuGB, metrics.mode === "tensor-parallel" ? metrics.weightsGB / n : metrics.weightsGB), "weightsPerGpuGB is inconsistent");
  fail(!close(metrics.overheadGB, config.overheadPerGpuGB * n), `overheadGB=${metrics.overheadGB}`);
  fail(!close(metrics.capacityGB, metrics.gpu.vramGB * 0.92 * n), `capacityGB=${metrics.capacityGB}`);
  fail(!close(metrics.kvTotalGB, expectedKvTotal), `kvTotalGB=${metrics.kvTotalGB}, expected ${expectedKvTotal}`);
  fail(!close(metrics.usedGB, metrics.weightsInstalledGB + metrics.kvTotalGB + metrics.overheadGB), "usedGB is not weights + KV + overhead");
  fail(!close(metrics.freeGB, metrics.capacityGB - metrics.usedGB), "freeGB is not capacity - used");
  fail(!close(metrics.usedFrac, metrics.usedGB / metrics.capacityGB), "usedFrac is inconsistent");
  fail(!close(metrics.kvPerUserGB, (metrics.kvPerTokenMB * metrics.context) / 1000), "kvPerUserGB is inconsistent with kvPerTokenMB");
  fail(!close(metrics.promptTokens, Math.max(64, context - outputTokens)), "promptTokens is inconsistent");
  fail(!close(metrics.batchPerGpu, metrics.mode === "replicas" ? metrics.batch / n : metrics.batch), "batchPerGpu is inconsistent");
  fail(!close(metrics.outputTps, (metrics.batch / metrics.tpotMs) * 1000), "outputTps is inconsistent");
  fail(!close(metrics.tokPerSecUser, 1000 / metrics.tpotMs), "tokPerSecUser is inconsistent");
  fail(!close(metrics.costPerHour, metrics.gpu.pricePerHour * n), "costPerHour is inconsistent");
  fail(metrics.fits !== expectedFits(metrics, config), "fits flag is inconsistent");

  if (metrics.fits && users <= MAX_USERS) {
    fail(metrics.maxUsers < users, `fits is true but maxUsers=${metrics.maxUsers} < users=${users}`);
  }

  return failures;
}

function checkAdvice(advice: TopologyAdvice, config: EngineConfig): string[] {
  const failures: string[] = [];
  const fail = (condition: boolean, message: string) => {
    if (condition) failures.push(message);
  };
  const integerBetween = (value: number, min: number, max: number, label: string) =>
    fail(!Number.isInteger(value) || value < min || value > max, `${label}=${value} is outside ${min}..${max}`);

  integerBetween(advice.minWeights, 1, 64, "minWeights");
  integerBetween(advice.minCapacity, 1, 64, "minCapacity");
  integerBetween(advice.minDecode, 0, 64, "minDecode");
  integerBetween(advice.minPrefill, 0, 64, "minPrefill");
  integerBetween(advice.recommended, 1, 64, "recommended");
  fail(!["single", "replicas", "tensor-parallel", "multi-node"].includes(advice.mode), `invalid advice.mode=${advice.mode}`);
  fail(advice.recommended === 1 && advice.mode !== "single", `recommended=1 but mode=${advice.mode}`);
  fail(!Number.isFinite(advice.efficiency) || advice.efficiency < 0 || advice.efficiency > 1 + EPSILON, `efficiency=${advice.efficiency}`);
  fail(!Number.isFinite(advice.ttftAt), `ttftAt=${advice.ttftAt}`);
  fail(!Number.isFinite(advice.tpotAt), `tpotAt=${advice.tpotAt}`);
  fail(!Number.isFinite(advice.maxUsersAt), `maxUsersAt=${advice.maxUsersAt}`);
  fail(!Array.isArray(advice.reasons), "reasons is not an array");
  fail(!Array.isArray(advice.warnings), "warnings is not an array");
  for (const [index, reason] of advice.reasons.entries()) {
    fail(typeof reason.label !== "string" || reason.label.length === 0, `reasons[${index}].label is empty`);
    fail(typeof reason.detail !== "string" || reason.detail.length === 0, `reasons[${index}].detail is empty`);
    fail(!Number.isFinite(reason.gpus), `reasons[${index}].gpus is not finite`);
  }
  for (const [index, warning] of advice.warnings.entries()) {
    fail(typeof warning !== "string" || warning.length === 0, `warnings[${index}] is empty`);
  }
  fail(advice.minWeights > 64, `minWeights=${advice.minWeights} exceeds 64`);
  fail(config.gpuCount < 1, "gpuCount below 1");
  return failures;
}

function summaryFor(metrics: Metrics, advice: TopologyAdvice | undefined): string {
  const fit = metrics.fits ? "fits" : "does not fit";
  const base = [
    metrics.mode,
    fit,
    `${metrics.maxUsers.toLocaleString()} slots`,
    `${metrics.usedGB.toFixed(1)}/${metrics.capacityGB.toFixed(1)} GB`,
    `TTFT ${metrics.ttftP95Ms.toFixed(0)} ms`,
    `TPOT ${metrics.tpotMs.toFixed(1)} ms`,
    `${metrics.outputTps.toFixed(0)} tok/s`,
  ];
  if (advice) base.push(`advice ${advice.recommended}× ${advice.mode}`);
  return base.join(" · ");
}

function addCoreMatrix() {
  for (const model of MODELS) {
    for (const quant of QUANTS) {
      for (const gpu of GPUS) {
        for (const gpuCount of [1, 2, 4, 8]) {
          const context = contextFor(8192, model.maxContext);
          const config = configWith({
            modelId: model.id,
            quantId: quant.id,
            kvQuantId: "kv8",
            gpuId: gpu.id,
            gpuCount,
            context,
            users: 150,
          });
          addCase(
            "Core matrix",
            `${model.name} · ${quant.label} weights · ${gpu.name} · ${gpuCount}× · ${contextLabel(context)} · 150 users · no cache`,
            config,
          );
        }
      }
    }
  }
}

function addScalingMatrix() {
  for (const model of MODELS) {
    for (const quantId of ["fp8", "int4"]) {
      const quant = QUANTS.find((item) => item.id === quantId);
      for (const gpuId of ["h100", "l40s", "rtx4090", "b200", "mi300x"]) {
        const gpu = GPUS.find((item) => item.id === gpuId);
        if (!quant || !gpu) continue;
        for (let gpuCount = 1; gpuCount <= 16; gpuCount += 1) {
          const context = contextFor(8192, model.maxContext);
          const config = configWith({
            modelId: model.id,
            quantId,
            kvQuantId: "kv8",
            gpuId,
            gpuCount,
            context,
            users: 150,
          });
          addCase(
            "GPU scaling",
            `${model.name} · ${quant.label} · ${gpu.name} · ${gpuCount}× · ${contextLabel(context)} · 150 users · no cache`,
            config,
          );
        }
      }
    }
  }
}

function addContextMatrix() {
  const contexts = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];
  for (const model of MODELS) {
    for (const gpuId of ["h100", "l40s", "a100", "mi300x"]) {
      const gpu = GPUS.find((item) => item.id === gpuId);
      if (!gpu) continue;
      for (const gpuCount of [1, 2, 4, 8]) {
        for (const rawContext of contexts) {
          const context = contextFor(rawContext, model.maxContext);
          const config = configWith({
            modelId: model.id,
            quantId: "fp8",
            kvQuantId: "kv8",
            gpuId,
            gpuCount,
            context,
            users: 50,
          });
          addCase(
            "Context sweep",
            `${model.name} · FP8 · ${gpu.name} · ${gpuCount}× · ${contextLabel(context)} · 50 users · no cache`,
            config,
          );
        }
      }
    }
  }
}

function addUserMatrix() {
  const userLoads = [1, 10, 50, 150, 500, 1500, 3000, 10000];
  for (const model of MODELS) {
    for (const gpuId of ["h100", "l40s", "b200", "mi300x"]) {
      const gpu = GPUS.find((item) => item.id === gpuId);
      if (!gpu) continue;
      for (const gpuCount of [1, 2, 4, 8]) {
        for (const users of userLoads) {
          const context = contextFor(8192, model.maxContext);
          const config = configWith({
            modelId: model.id,
            quantId: "fp8",
            kvQuantId: "kv8",
            gpuId,
            gpuCount,
            context,
            users,
          });
          addCase(
            "User load sweep",
            `${model.name} · FP8 · ${gpu.name} · ${gpuCount}× · ${contextLabel(context)} · ${users.toLocaleString()} users · no cache`,
            config,
          );
        }
      }
    }
  }
}

function addKvMatrix() {
  for (const model of MODELS) {
    for (const kvQuant of KV_QUANTS) {
      for (const gpuId of ["h100", "l40s", "a100", "mi300x"]) {
        const gpu = GPUS.find((item) => item.id === gpuId);
        if (!gpu) continue;
        for (const gpuCount of [1, 2, 4, 8]) {
          const context = contextFor(8192, model.maxContext);
          const config = configWith({
            modelId: model.id,
            quantId: "fp8",
            kvQuantId: kvQuant.id,
            gpuId,
            gpuCount,
            context,
            users: 150,
          });
          addCase(
            "KV precision",
            `${model.name} · FP8 weights · ${kvQuant.label} KV · ${gpu.name} · ${gpuCount}× · ${contextLabel(context)} · 150 users · no cache`,
            config,
          );
        }
      }
    }
  }
}

function addCacheMatrix() {
  for (const model of MODELS) {
    for (const gpuId of ["h100", "l40s", "b200"]) {
      const gpu = GPUS.find((item) => item.id === gpuId);
      if (!gpu) continue;
      for (const gpuCount of [1, 2, 4, 8]) {
        for (const cache of CACHE_PROFILES) {
          const context = contextFor(8192, model.maxContext);
          const config = configWith(
            {
              modelId: model.id,
              quantId: "fp8",
              kvQuantId: "kv8",
              gpuId,
              gpuCount,
              context,
              users: 150,
            },
            cache,
          );
          addCase(
            "Cache sweep",
            `${model.name} · FP8 · ${gpu.name} · ${gpuCount}× · ${contextLabel(context)} · 150 users · ${cache.name}`,
            config,
          );
        }
      }
    }
  }
}

function addTargetedCases() {
  const int4Base = configWith(
    { ...BASE_CONFIG, quantId: "int4" },
    {
      name: "default cache",
      exactHit: BASE_CONFIG.exactHit,
      semanticHit: BASE_CONFIG.semanticHit,
      prefixHit: BASE_CONFIG.prefixHit,
      prefixShare: BASE_CONFIG.prefixShare,
    },
  );
  addCase(
    "Regression / targeted",
    "70B INT4 on 4×H100 must stay tensor-parallel and fit 150 users",
    int4Base,
    {
      withAdvice: true,
      custom: (metrics, advice) => {
        const failures: string[] = [];
        if (metrics.mode !== "tensor-parallel") failures.push(`mode=${metrics.mode}, expected tensor-parallel`);
        if (!metrics.fits) failures.push("INT4 layout should fit the 4×H100 budget");
        if (metrics.usedGB > metrics.capacityGB + EPSILON) failures.push(`used ${metrics.usedGB.toFixed(1)} GB > capacity ${metrics.capacityGB.toFixed(1)} GB`);
        if (metrics.maxUsers < BASE_CONFIG.users) failures.push(`maxUsers=${metrics.maxUsers} < users=${BASE_CONFIG.users}`);
        if (advice && advice.mode !== "tensor-parallel") failures.push(`advice mode=${advice.mode}, expected tensor-parallel`);
        if (advice && advice.recommended > 4) failures.push(`advice recommends ${advice.recommended} GPUs, expected ≤4`);
        return failures;
      },
    },
  );

  addCase(
    "Regression / targeted",
    "INT4 is not slower or smaller than FP8 on the same 4×H100 70B layout",
    int4Base,
    {
      custom: () => {
        const failures: string[] = [];
        const fp8 = computeMetrics(BASE_CONFIG);
        const int4 = computeMetrics({ ...BASE_CONFIG, quantId: "int4" });
        if (int4.maxUsers <= fp8.maxUsers) failures.push(`INT4 slots ${int4.maxUsers} <= FP8 slots ${fp8.maxUsers}`);
        if (int4.tpotMs >= fp8.tpotMs) failures.push(`INT4 TPOT ${int4.tpotMs.toFixed(2)} ms >= FP8 TPOT ${fp8.tpotMs.toFixed(2)} ms`);
        if (int4.ttftP95Ms > fp8.ttftP95Ms + 1) failures.push(`INT4 TTFT ${int4.ttftP95Ms.toFixed(1)} ms > FP8 TTFT ${fp8.ttftP95Ms.toFixed(1)} ms`);
        if (int4.outputTps <= fp8.outputTps) failures.push(`INT4 throughput ${int4.outputTps.toFixed(0)} <= FP8 throughput ${fp8.outputTps.toFixed(0)}`);
        return failures;
      },
    },
  );

  const replicaFit = configWith({
    modelId: "llama8b",
    quantId: "bf16",
    kvQuantId: "kv8",
    gpuId: "l40s",
    gpuCount: 2,
    context: 4096,
    users: 80,
  });
  addCase(
    "Regression / targeted",
    "Small model with enough headroom uses replicas",
    replicaFit,
    {
      custom: (metrics) => {
        const failures: string[] = [];
        if (metrics.mode !== "replicas") failures.push(`mode=${metrics.mode}, expected replicas`);
        if (!close(metrics.weightsInstalledGB, metrics.weightsGB * 2)) failures.push("replica weight duplication is not counted");
        if (metrics.maxUsers < 80) failures.push(`replicas hold ${metrics.maxUsers} slots, expected ≥80`);
        return failures;
      },
    },
  );

  addCase(
    "Regression / targeted",
    "Replicas that cannot hold the load fall back to sharding",
    configWith({
      modelId: "llama8b",
      quantId: "bf16",
      kvQuantId: "kv8",
      gpuId: "l40s",
      gpuCount: 2,
      context: 4096,
      users: 300,
    }),
    {
      custom: (metrics) => {
        const failures: string[] = [];
        if (metrics.mode !== "tensor-parallel") failures.push(`mode=${metrics.mode}, expected tensor-parallel`);
        if (metrics.maxUsers < 200) failures.push(`sharded layout only holds ${metrics.maxUsers} slots`);
        return failures;
      },
    },
  );

  const qwen = configWith({
    modelId: "qwen3_8_27b",
    quantId: "fp8",
    kvQuantId: "kv8",
    gpuId: "l40s",
    gpuCount: 1,
    context: 8192,
    users: 40,
  });
  addCase(
    "Regression / targeted",
    "Qwen3.8 27B hybrid attention uses 16 KV layers",
    qwen,
    {
      custom: (metrics) => {
        const failures: string[] = [];
        if (metrics.model.maxContext !== 262144) failures.push(`maxContext=${metrics.model.maxContext}`);
        if (Math.abs(metrics.kvPerTokenMB - 0.032768) > 1e-6) failures.push(`kvPerTokenMB=${metrics.kvPerTokenMB}, expected 0.032768`);
        if (metrics.maxUsers < 30) failures.push(`maxUsers=${metrics.maxUsers}, expected >30`);
        return failures;
      },
    },
  );

  addCase(
    "Regression / targeted",
    "405B FP8 needs six H100s just for weights",
    configWith({
      modelId: "llama405b",
      quantId: "fp8",
      kvQuantId: "kv8",
      gpuId: "h100",
      gpuCount: 8,
      context: 8192,
      users: 10,
    }),
    {
      withAdvice: true,
      custom: (_metrics, advice) => {
        const failures: string[] = [];
        if (!advice) return ["advice missing"];
        if (advice.minWeights !== 6) failures.push(`minWeights=${advice.minWeights}, expected 6`);
        return failures;
      },
    },
  );

  addCase(
    "Regression / targeted",
    "70B BF16 needs two H100s for weights",
    configWith({
      modelId: "llama70b",
      quantId: "bf16",
      kvQuantId: "kv8",
      gpuId: "h100",
      gpuCount: 2,
      context: 8192,
      users: 10,
    }),
    {
      withAdvice: true,
      custom: (_metrics, advice) => {
        const failures: string[] = [];
        if (!advice) return ["advice missing"];
        if (advice.minWeights !== 2) failures.push(`minWeights=${advice.minWeights}, expected 2`);
        if (advice.mode !== "tensor-parallel") failures.push(`advice mode=${advice.mode}, expected tensor-parallel`);
        return failures;
      },
    },
  );

  addCase(
    "Regression / targeted",
    "One GPU stays in single mode",
    configWith({
      modelId: "llama70b",
      quantId: "fp8",
      kvQuantId: "kv8",
      gpuId: "h100",
      gpuCount: 1,
      context: 8192,
      users: 10,
    }),
    {
      custom: (metrics) => {
        const failures: string[] = [];
        if (metrics.mode !== "single") failures.push(`mode=${metrics.mode}, expected single`);
        if (metrics.maxUsers !== 0) failures.push(`maxUsers=${metrics.maxUsers}, expected 0 without KV room`);
        if (metrics.fits) failures.push("single 70B FP8 H100 should not fit at 8k context");
        return failures;
      },
    },
  );

  addCase(
    "Regression / targeted",
    "64-GPU ceiling is clamped, not negative",
    configWith({
      modelId: "llama70b",
      quantId: "fp8",
      kvQuantId: "kv8",
      gpuId: "h100",
      gpuCount: 64,
      context: 8192,
      users: 10000,
    }),
    {
      withAdvice: true,
      custom: (metrics, advice) => {
        const failures: string[] = [];
        if (metrics.n !== 64) failures.push(`n=${metrics.n}, expected 64`);
        if (metrics.maxUsers > MAX_USERS) failures.push(`maxUsers=${metrics.maxUsers} exceeds clamp`);
        if (advice && advice.recommended > 64) failures.push(`recommended=${advice.recommended} exceeds 64`);
        return failures;
      },
    },
  );

  addCase(
    "Regression / targeted",
    "Qwen3.8 27B native 262k context is preserved",
    configWith({
      modelId: "qwen3_8_27b",
      quantId: "fp8",
      kvQuantId: "kv8",
      gpuId: "h100",
      gpuCount: 8,
      context: 262144,
      users: 1,
    }),
    {
      custom: (metrics) => {
        const failures: string[] = [];
        if (metrics.context !== 262144) failures.push(`context=${metrics.context}, expected 262144`);
        return failures;
      },
    },
  );

  addCase(
    "Regression / targeted",
    "Context clamps up to 256 tokens",
    configWith({
      modelId: "llama70b",
      quantId: "fp8",
      kvQuantId: "kv8",
      gpuId: "h100",
      gpuCount: 4,
      context: 1,
      users: 1,
    }),
    {
      custom: (metrics) => (metrics.context === 256 ? [] : [`context=${metrics.context}, expected 256`]),
    },
  );

  addCase(
    "Regression / targeted",
    "Output tokens clamp below context",
    configWith({
      modelId: "llama8b",
      quantId: "fp8",
      kvQuantId: "kv8",
      gpuId: "l40s",
      gpuCount: 1,
      context: 1024,
      outputTokens: 5000,
      users: 1,
    }),
    {
      custom: (metrics) => {
        const failures: string[] = [];
        if (metrics.outputTokens !== 960) failures.push(`outputTokens=${metrics.outputTokens}, expected 960`);
        if (metrics.promptTokens !== 64) failures.push(`promptTokens=${metrics.promptTokens}, expected 64`);
        return failures;
      },
    },
  );

  addCase(
    "Regression / targeted",
    "Zero runtime overhead accounting",
    configWith({
      modelId: "llama8b",
      quantId: "fp8",
      kvQuantId: "kv8",
      gpuId: "l40s",
      gpuCount: 2,
      context: 4096,
      users: 80,
      overheadPerGpuGB: 0,
    }),
    {
      custom: (metrics) => (metrics.overheadGB === 0 ? [] : [`overheadGB=${metrics.overheadGB}, expected 0`]),
    },
  );

  addCase(
    "Regression / targeted",
    "High per-GPU overhead still returns finite metrics",
    configWith({
      modelId: "llama8b",
      quantId: "fp8",
      kvQuantId: "kv8",
      gpuId: "rtx3060",
      gpuCount: 2,
      context: 8192,
      users: 50,
      overheadPerGpuGB: 8,
    }),
  );

  addCase(
    "Regression / targeted",
    "Caching never increases GPU KV demand",
    configWith(
      {
        modelId: "llama70b",
        quantId: "fp8",
        kvQuantId: "kv8",
        gpuId: "h100",
        gpuCount: 4,
        context: 8192,
        users: 150,
      },
      {
        name: "high cache",
        exactHit: 0.5,
        semanticHit: 0.4,
        prefixHit: 0.9,
        prefixShare: 0.8,
      },
    ),
    {
      custom: (metrics) => {
        const failures: string[] = [];
        const raw = computeMetrics({
          ...BASE_CONFIG,
          modelId: "llama70b",
          quantId: "fp8",
          kvQuantId: "kv8",
          gpuId: "h100",
          gpuCount: 4,
          context: 8192,
          users: 150,
          exactHit: 0,
          semanticHit: 0,
          prefixHit: 0,
          prefixShare: 0,
        });
        if (metrics.maxUsers < raw.maxUsers) failures.push(`cached slots ${metrics.maxUsers} < raw slots ${raw.maxUsers}`);
        if (metrics.arrivalRate > raw.arrivalRate + EPSILON) failures.push(`cached arrivals ${metrics.arrivalRate} > raw arrivals ${raw.arrivalRate}`);
        return failures;
      },
    },
  );

  addCase(
    "Regression / targeted",
    "FP8 KV holds at least as much as BF16 KV",
    configWith({
      modelId: "llama8b",
      quantId: "fp8",
      kvQuantId: "kv8",
      gpuId: "l40s",
      gpuCount: 2,
      context: 8192,
      users: 100,
    }),
    {
      custom: (metrics) => {
        const failures: string[] = [];
        const bf16 = computeMetrics({
          ...BASE_CONFIG,
          modelId: "llama8b",
          quantId: "fp8",
          kvQuantId: "kv16",
          gpuId: "l40s",
          gpuCount: 2,
          context: 8192,
          users: 100,
          exactHit: 0,
          semanticHit: 0,
          prefixHit: 0,
          prefixShare: 0,
        });
        if (metrics.maxUsers < bf16.maxUsers) failures.push(`FP8 KV slots ${metrics.maxUsers} < BF16 KV slots ${bf16.maxUsers}`);
        return failures;
      },
    },
  );

  for (const preset of PRESETS) {
    addCase(
      "Presets",
      preset.name,
      { ...preset.config },
      { withAdvice: true },
    );
  }

  addCase(
    "Regression / targeted",
    "GPU catalogue enforces the 12 GB floor",
    configWith({
      modelId: "qwen3_8b",
      quantId: "fp8",
      kvQuantId: "kv8",
      gpuId: "rtx3060",
      gpuCount: 1,
      context: 4096,
      users: 1,
    }),
    {
      custom: () => {
        const failures: string[] = [];
        for (const gpu of GPUS) {
          if (gpu.vramGB < 12) failures.push(`${gpu.name} has ${gpu.vramGB} GB < 12 GB`);
        }
        return failures;
      },
    },
  );
}

function runCase(testCase: TestCase): TestResult {
  const started = Date.now();
  let metrics: Metrics | undefined;
  let advice: TopologyAdvice | undefined;
  const failures: string[] = [];

  try {
    metrics = computeMetrics(testCase.config);
    if (testCase.withAdvice) advice = adviseTopology(testCase.config);
  } catch (error) {
    failures.push(`threw: ${error instanceof Error ? error.message : String(error)}`);
  }

  if (metrics) {
    failures.push(...checkMetrics(metrics, testCase.config));
    if (advice) failures.push(...checkAdvice(advice, testCase.config));
    if (testCase.custom) {
      try {
        failures.push(...testCase.custom(metrics, advice));
      } catch (error) {
        failures.push(`custom check threw: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
  }

  return {
    id: testCase.id,
    suite: testCase.suite,
    name: testCase.name,
    status: failures.length === 0 ? "PASS" : "FAIL",
    summary: metrics ? summaryFor(metrics, advice) : "no metrics",
    failures,
    durationMs: Date.now() - started,
  };
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderSummary(results: TestResult[]) {
  const suites = new Map<string, { total: number; passed: number; failed: number }>();
  const modes = new Map<string, number>();
  let fits = 0;
  for (const result of results) {
    const entry = suites.get(result.suite) ?? { total: 0, passed: 0, failed: 0 };
    entry.total += 1;
    if (result.status === "PASS") entry.passed += 1;
    else entry.failed += 1;
    suites.set(result.suite, entry);
    const parts = result.summary.split(" · ");
    if (parts[1] === "fits") fits += 1;
    const mode = parts[0] ?? "unknown";
    modes.set(mode, (modes.get(mode) ?? 0) + 1);
  }
  return { suites, modes, fits };
}

function buildHtml(results: TestResult[], startedAt: string, durationMs: number): string {
  const total = results.length;
  const passed = results.filter((result) => result.status === "PASS").length;
  const failed = total - passed;
  const { suites, modes, fits } = renderSummary(results);
  const noFit = total - fits;
  const suiteRows = [...suites.entries()]
    .map(
      ([suite, stats]) => `<tr>
        <td>${escapeHtml(suite)}</td>
        <td>${stats.total}</td>
        <td class="pass">${stats.passed}</td>
        <td class="${stats.failed ? "fail" : ""}">${stats.failed}</td>
      </tr>`,
    )
    .join("");
  const modeRows = [...modes.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([mode, count]) => `<span class="pill">${escapeHtml(mode)}: ${count}</span>`)
    .join("");
  const rows = results
    .map((result) => {
      const failureHtml = result.failures.length
        ? `<ul class="failure-list">${result.failures.map((failure) => `<li>${escapeHtml(failure)}</li>`).join("")}</ul>`
        : "";
      return `<tr data-status="${result.status.toLowerCase()}" data-suite="${escapeHtml(result.suite)}">
        <td class="mono">${escapeHtml(result.id)}</td>
        <td>${escapeHtml(result.suite)}</td>
        <td class="case-name">${escapeHtml(result.name)}</td>
        <td><span class="status ${result.status.toLowerCase()}">${result.status}</span></td>
        <td class="mono summary">${escapeHtml(result.summary)}${failureHtml}</td>
        <td class="mono">${result.durationMs}</td>
      </tr>`;
    })
    .join("");

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Capacity Lab — inference matrix report</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #070b10;
      --panel: #0d141c;
      --panel-2: #111b25;
      --line: #263543;
      --text: #e7eef5;
      --muted: #93a4b4;
      --cyan: #6ee7f9;
      --lime: #b6f36b;
      --red: #ff7b72;
      --amber: #ffb347;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at 20% 0%, #123041 0, transparent 35rem), var(--bg);
      color: var(--text);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .wrap { max-width: 1680px; margin: 0 auto; padding: 28px; }
    h1 { margin: 0 0 6px; font-size: 28px; letter-spacing: -0.03em; }
    h2 { margin: 28px 0 10px; font-size: 15px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--cyan); }
    p { color: var(--muted); }
    .subtitle { margin: 0; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 22px 0; }
    .card { background: linear-gradient(180deg, var(--panel-2), var(--panel)); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
    .card .label { color: var(--muted); font-size: 11px; letter-spacing: 0.11em; text-transform: uppercase; }
    .card .value { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 26px; margin-top: 4px; }
    .card .value.pass { color: var(--lime); }
    .card .value.fail { color: var(--red); }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { color: var(--muted); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; background: var(--panel-2); position: sticky; top: 0; z-index: 1; }
    tr:hover td { background: #101c27; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    .case-name { min-width: 520px; }
    .summary { min-width: 480px; white-space: pre-wrap; }
    .status { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; }
    .status.pass { color: var(--lime); background: #1e3312; }
    .status.fail { color: var(--red); background: #3a1717; }
    .pass { color: var(--lime); }
    .fail { color: var(--red); }
    .failure-list { margin: 6px 0 0; padding-left: 18px; color: #ffb4ae; }
    .failure-list li { margin: 2px 0; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 16px 0; }
    .toolbar input, .toolbar select {
      background: var(--panel-2); color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 9px 11px; font: inherit;
    }
    .toolbar input { flex: 1 1 340px; }
    .pill { display: inline-block; margin: 0 6px 6px 0; border: 1px solid var(--line); border-radius: 999px; padding: 4px 9px; color: var(--muted); font-size: 12px; }
    .meta { color: var(--muted); font-size: 12px; }
    .note { border-left: 3px solid var(--cyan); padding-left: 12px; color: var(--muted); }
    .hidden { display: none; }
    .suite-table td:first-child { color: var(--muted); }
    .suite-table { max-width: 620px; }
    footer { margin-top: 26px; color: var(--muted); font-size: 12px; }
    @media (max-width: 900px) {
      .wrap { padding: 16px; }
      .case-name, .summary { min-width: 0; }
      th:nth-child(1), td:nth-child(1), th:nth-child(2), td:nth-child(2), th:nth-child(6), td:nth-child(6) { display: none; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Capacity Lab — inference matrix report</h1>
    <p class="subtitle">Every row is one deterministic case executed sequentially: models × weight precision × GPU × GPU count, plus context, user-load, KV-precision, cache, preset, and regression suites.</p>
    <p class="note">PASS means the case satisfied its invariants. A case can pass while reporting “does not fit” — that is a valid capacity outcome, not a test failure. Regression rows additionally assert exact expected modes and capacities.</p>
    <div class="cards">
      <div class="card"><div class="label">Cases</div><div class="value">${total.toLocaleString()}</div></div>
      <div class="card"><div class="label">Passed</div><div class="value pass">${passed.toLocaleString()}</div></div>
      <div class="card"><div class="label">Failed</div><div class="value ${failed ? "fail" : "pass"}">${failed.toLocaleString()}</div></div>
      <div class="card"><div class="label">Fits</div><div class="value">${fits.toLocaleString()}</div></div>
      <div class="card"><div class="label">Does not fit</div><div class="value">${noFit.toLocaleString()}</div></div>
      <div class="card"><div class="label">Runtime</div><div class="value">${(durationMs / 1000).toFixed(1)}s</div></div>
    </div>
    <p class="meta">Generated ${escapeHtml(startedAt)} · ${escapeHtml(String(total))} cases · ${escapeHtml(String(failed))} failures · report at ${escapeHtml(HTML_PATH)}</p>
    <div class="panel">
      <h2>Mode distribution</h2>
      <div>${modeRows}</div>
      <h2>Suites</h2>
      <table class="suite-table">
        <thead><tr><th>Suite</th><th>Total</th><th>Passed</th><th>Failed</th></tr></thead>
        <tbody>${suiteRows}</tbody>
      </table>
    </div>
    <h2>Cases</h2>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Search test case, model, GPU, suite..." />
      <select id="status">
        <option value="all">All statuses</option>
        <option value="fail">Failures only</option>
        <option value="pass">Passed only</option>
      </select>
      <select id="suite-filter"><option value="all">All suites</option>${[...suites.keys()]
        .map((suite) => `<option value="${escapeHtml(suite)}">${escapeHtml(suite)}</option>`)
        .join("")}</select>
      <span class="meta" id="visible-count">${total.toLocaleString()} visible</span>
    </div>
    <div class="panel" style="padding:0; overflow:auto; max-height: 70vh;">
      <table id="cases">
        <thead><tr><th>ID</th><th>Suite</th><th>Test case</th><th>Status</th><th>Result / failure</th><th>ms</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <footer>Generated by <span class="mono">scripts/verify-matrix.ts</span>. Full text log: <a href="inference-matrix.log">inference-matrix.log</a>.</footer>
  </div>
  <script>
    const search = document.getElementById('search');
    const status = document.getElementById('status');
    const suite = document.getElementById('suite-filter');
    const rows = [...document.querySelectorAll('#cases tbody tr')];
    const visible = document.getElementById('visible-count');
    function applyFilters() {
      const query = search.value.trim().toLowerCase();
      const wantedStatus = status.value;
      const wantedSuite = suite.value;
      let count = 0;
      for (const row of rows) {
        const matchesQuery = !query || row.innerText.toLowerCase().includes(query);
        const matchesStatus = wantedStatus === 'all' || row.dataset.status === wantedStatus;
        const matchesSuite = wantedSuite === 'all' || row.dataset.suite === wantedSuite;
        const show = matchesQuery && matchesStatus && matchesSuite;
        row.classList.toggle('hidden', !show);
        if (show) count += 1;
      }
      visible.textContent = count.toLocaleString() + ' visible';
    }
    search.addEventListener('input', applyFilters);
    status.addEventListener('change', applyFilters);
    suite.addEventListener('change', applyFilters);
    applyFilters();
  </script>
</body>
</html>`;
}

function main() {
  addCoreMatrix();
  addScalingMatrix();
  addContextMatrix();
  addUserMatrix();
  addKvMatrix();
  addCacheMatrix();
  addTargetedCases();

  const startedAt = new Date().toISOString();
  const start = performance.now();
  const results: TestResult[] = [];
  const logLines: string[] = [];

  console.log(`Running ${cases.length.toLocaleString()} inference cases sequentially...`);
  for (const [index, testCase] of cases.entries()) {
    const result = runCase(testCase);
    results.push(result);
    const line = `[${result.status}] ${result.id} | ${result.suite} | ${result.name} | ${result.summary}${
      result.failures.length ? ` | ${result.failures.join(" | ")}` : ""
    }`;
    logLines.push(line);
    if (process.env.MATRIX_VERBOSE === "1") {
      console.log(line);
    } else if (result.status === "FAIL") {
      console.log(`FAIL ${result.id} | ${result.name} | ${result.failures.join(" | ")}`);
    }
    if ((index + 1) % 500 === 0 || index + 1 === cases.length) {
      console.log(`  ${(index + 1).toLocaleString()}/${cases.length.toLocaleString()} cases complete`);
    }
  }

  const durationMs = performance.now() - start;
  const passed = results.filter((result) => result.status === "PASS").length;
  const failed = results.length - passed;
  const fits = results.filter((result) => result.summary.split(" · ")[1] === "fits").length;
  const noFit = results.length - fits;

  mkdirSync(REPORT_DIR, { recursive: true });
  writeFileSync(LOG_PATH, `${logLines.join("\n")}\n`, "utf8");
  writeFileSync(HTML_PATH, buildHtml(results, startedAt, durationMs), "utf8");

  console.log("");
  console.log(`Cases: ${results.length.toLocaleString()} | passed: ${passed.toLocaleString()} | failed: ${failed.toLocaleString()} | ${(durationMs / 1000).toFixed(1)}s`);
  console.log(`Fits: ${fits.toLocaleString()} | does not fit: ${noFit.toLocaleString()}`);
  console.log(`Log:  ${LOG_PATH}`);
  console.log(`HTML: ${HTML_PATH}`);

  if (failed > 0) {
    console.log("");
    console.log("First failures:");
    for (const result of results.filter((item) => item.status === "FAIL").slice(0, 20)) {
      console.log(`- ${result.id} ${result.name}`);
      for (const failure of result.failures) console.log(`  ${failure}`);
    }
    process.exitCode = 1;
  }
}

main();
