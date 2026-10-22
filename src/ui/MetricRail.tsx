import { fmtCompact, fmtGB, fmtMoney, fmtMs, fmtNum, fmtPct } from "../lib/format";
import type { Metrics, TopologyAdvice } from "../lib/formulas";
import { useStore } from "../state/store";
import { Info } from "./controls";

const BOTTLENECK_LABEL: Record<string, { text: string; tone: string }> = {
  vram: { text: "VRAM", tone: "red" },
  queue: { text: "Queue", tone: "red" },
  "kv-bandwidth": { text: "KV bandwidth", tone: "amber" },
  "weights-bandwidth": { text: "Weights BW", tone: "cyan" },
  "prefill-compute": { text: "Prefill FLOPs", tone: "amber" },
  balanced: { text: "Balanced", tone: "" },
};

export function MetricRail({ m, advice }: { m: Metrics; advice: TopologyAdvice }) {
  const showMath = useStore((s) => s.showMath);
  const config = useStore((s) => s.config);
  const setLens = useStore((s) => s.setLens);
  const setExplain = useStore((s) => s.setExplain);

  const scale = Math.max(m.capacityGB, m.usedGB, 1);
  const pct = (v: number) => `${Math.min(100, (v / scale) * 100)}%`;
  const overflow = Math.max(0, m.usedGB - m.capacityGB);
  const ttftTone =
    m.ttftP95Ms <= config.targetTtft ? "ok" : m.ttftP95Ms < 30000 ? "warn" : "bad";
  const bn = BOTTLENECK_LABEL[m.bottleneck] ?? BOTTLENECK_LABEL.balanced;
  const topoMode =
    advice.mode === "tensor-parallel"
      ? "tensor parallel"
      : advice.mode === "replicas"
        ? "replicas"
        : advice.mode === "multi-node"
          ? "multi-node TP/PP"
          : "single GPU";
  const kvActiveGB = (m.kvPerTokenMB / 1000) * m.context * m.batch;
  const weightsReadGB = m.mode === "tensor-parallel" ? m.weightsPerGpuGB : m.weightsGB;
  const readPerGpuGB = weightsReadGB + kvActiveGB / m.n;
  const targetGap =
    advice.maxUsersAt < config.users
      ? "capacity"
      : advice.minDecode === 0
        ? "TPOT"
        : advice.minPrefill === 0
          ? "TTFT"
          : "";

  return (
    <aside className="panel rail metrics" aria-label="Live readouts" data-tour="metrics">
      <div className="metric">
        <div className="metric-label">
          Concurrent capacity
          <Info topic="users" />
        </div>
        <div
          className="metric-value"
          style={{ color: m.maxUsers === 0 ? "#ff5f5f" : undefined }}
        >
          {m.maxUsers === 0 ? "0" : fmtNum(m.maxUsers)}
        </div>
        <div className="metric-sub">
          <span className={m.utilization > 0.9 ? "bad" : m.utilization > 0.75 ? "warn" : "ok"}>
            {fmtPct(m.utilization)} of KV slots
          </span>{" "}
          · {fmtNum(m.concurrency)} in flight
          <br />
          {fmtNum(config.users)} users · {fmtGB(m.kvPerUserGB)} at{" "}
          {fmtNum(m.context / 1024, 0)}k context
        </div>
      </div>

      <div className="metric">
        <div className="metric-label">
          Memory
          <Info topic="vram" />
        </div>
        <div className="metric-value sm">
          {fmtGB(m.usedGB)}
          <span className="unit">/ {fmtGB(m.capacityGB)}</span>
        </div>
        <div className="stackbar" aria-hidden>
          <span className="w" style={{ width: pct(m.weightsInstalledGB) }} />
          <span className="k" style={{ width: pct(m.kvTotalGB) }} />
          <span className="o" style={{ width: pct(m.overheadGB) }} />
          {overflow > 0 ? <span className="over" style={{ width: pct(overflow) }} /> : null}
          <span className="cap-mark" style={{ left: `${(m.capacityGB / scale) * 100}%` }} />
        </div>
        <div className="legend">
          <span>
            <i style={{ background: "#ffb347" }} /> weights {fmtGB(m.weightsInstalledGB)}
          </span>
          <span>
            <i style={{ background: "#4de1ff" }} /> KV {fmtGB(m.kvTotalGB)}
          </span>
          <span>
            <i style={{ background: "#3a4655" }} /> runtime {fmtGB(m.overheadGB)}
          </span>
        </div>
        {!m.fits ? (
          <div className="metric-sub" style={{ color: "#ff5f5f" }}>
            Over capacity by {fmtGB(m.usedGB - m.capacityGB)} — nothing is servable.
          </div>
        ) : null}
      </div>

      <div className="metric">
        <div className="metric-label">
          TTFT · p95
          <Info topic="ttft" />
        </div>
        <div className="metric-value">
          {fmtMs(m.ttftP95Ms)}
          <span className="unit">p95</span>
        </div>
        <div className="metric-sub">
          p50 {fmtMs(m.ttftP50Ms)} · target{" "}
          <b className={ttftTone}>{fmtMs(config.targetTtft)}</b>
          <br />
          queue {fmtMs(m.queueWaitP95Ms)} + prefill {fmtMs(m.prefillMs)}
        </div>
        <div className="metric-sub" style={{ marginTop: 6 }}>
          <span style={{ color: "#8496a9" }}>decode</span> {Math.round(m.tpotMs)} ms/token ·{" "}
          {m.tokPerSecUser.toFixed(1)} tok/s per user
        </div>
        {showMath ? (
          <div className="formula">TTFT = queue_wait + prefill</div>
        ) : null}
      </div>

      <div className="metric">
        <div className="metric-label">
          Throughput
          <Info topic="batching" />
        </div>
        <div className="metric-value">
          {fmtCompact(m.outputTps)}
          <span className="unit">tok/s</span>
        </div>
        <div className="metric-sub">
          batch of <b>{fmtNum(m.batch)}</b> · {fmtNum(m.arrivalRate, 1)} req/s ·{" "}
          {fmtNum(m.queueLength)} queued
          <br />
          {fmtGB(readPerGpuGB)} per GPU per decode step
        </div>
        <div className="metric-sub" style={{ marginTop: 6 }}>
          <span style={{ color: "#8496a9" }}>cost</span> {fmtMoney(m.costPerMillionOut)}/1M
          out · {fmtMoney(m.costPerHour)}/hr
          <Info topic="cost" />
        </div>
      </div>

      <div className="metric">
        <div className="metric-label">
          Constraint
          <Info topic="topology" />
        </div>
        <div style={{ marginTop: 4 }}>
          <span className={`tag ${bn.tone}`}>{bn.text}</span>
        </div>
        <div className="metric-sub">{m.bottleneckNote}</div>
      </div>

      <div className="metric">
        <div className="metric-label">
          Topology
          <Info topic={advice.mode === "replicas" ? "replicas" : "tp"} />
        </div>
        <div className="metric-value sm">
          {advice.recommended} × {m.gpu.short}
          <span className="unit">{topoMode}</span>
        </div>
        <div className="metric-sub">
          {targetGap ? (
            `${advice.recommended} GPUs is the best available; ${targetGap} still misses at 64.`
          ) : advice.recommended === m.n ? (
            "Current layout is the recommendation."
          ) : advice.recommended < m.n ? (
            `${advice.recommended} is enough; you have ${m.n}.`
          ) : (
            `You have ${m.n}. Move to ${advice.recommended} for this SLO.`
          )}
          <br />
          {advice.mode === "replicas"
            ? "Independent copies · no all-reduce"
            : `scaling efficiency ${fmtPct(advice.efficiency)}`}
        </div>
        <button
          className="ghost-btn"
          style={{ marginTop: 6, width: "100%", justifyContent: "space-between" }}
          onClick={() => {
            setLens("topology");
            setExplain(null);
          }}
        >
          Why these GPUs <span>→</span>
        </button>
      </div>
    </aside>
  );
}
