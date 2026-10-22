import { fmtGB, fmtMoney, fmtMs, fmtNum, fmtPct } from "../lib/format";
import type { EngineConfig, Metrics, TopologyAdvice } from "../lib/formulas";
import type { Lens } from "../state/store";

type Row = { label: string; value: string; dot?: string };

export function InsightCard({
  m,
  advice,
  config,
  lens,
}: {
  m: Metrics;
  advice: TopologyAdvice;
  config: EngineConfig;
  lens: Lens;
}) {
  if (lens === "overview") return null;

  let title = "";
  let rows: Row[] = [];
  let note = "";
  let warning = "";

  if (lens === "memory") {
    title = "Where the VRAM goes";
    rows = [
      { label: "Weights", value: fmtGB(m.weightsInstalledGB), dot: "#ffb347" },
      { label: "KV cache", value: fmtGB(m.kvTotalGB), dot: "#4de1ff" },
      { label: "Runtime", value: fmtGB(m.overheadGB), dot: "#3a4655" },
      { label: m.fits ? "Free" : "Over capacity", value: fmtGB(Math.abs(m.freeGB)), dot: m.fits ? "#b6f36b" : "#ff5f5f" },
    ];
    note = m.fits
      ? `Each GPU holds ${fmtGB(m.weightsPerGpuGB)} of weights and ${fmtGB(m.kvTotalGB / m.n)} of KV.`
      : `Sized for all ${fmtNum(config.users)} users: ${fmtGB(m.kvTotalGB)} of KV. Real servers admit what fits and queue the rest.`;
  } else if (lens === "latency") {
    title = "One request, end to end";
    rows = [
      { label: "Queue wait p95", value: fmtMs(m.queueWaitP95Ms), dot: "#ff5c8a" },
      { label: "Prefill", value: fmtMs(m.prefillMs), dot: "#ffb347" },
      { label: `Decode × ${m.outputTokens}`, value: fmtMs(m.outputTokens * m.tpotMs), dot: "#4de1ff" },
      { label: "Decode step", value: `${Math.round(m.tpotMs)} ms`, dot: "#4de1ff" },
      { label: "Batch", value: `${Math.round(m.batch)} seq`, dot: "#b6f36b" },
    ];
    note = `Service time ≈ ${fmtMs(m.prefillMs + m.outputTokens * m.tpotMs)} per request.`;
  } else if (lens === "caching") {
    title = "Cache tiers, in order";
    rows = [
      {
        label: "Exact response",
        value: config.exactHit > 0 ? `${fmtPct(config.exactHit)} hit` : "off",
        dot: config.exactHit > 0 ? "#4de1ff" : "#33404e",
      },
      {
        label: "Semantic",
        value: config.semanticHit > 0 ? `${fmtPct(config.semanticHit)} hit` : "off",
        dot: config.semanticHit > 0 ? "#a78bfa" : "#33404e",
      },
      {
        label: "Prefix reuse",
        value: config.prefixHit > 0 ? `${fmtPct(config.prefixHit)} hit` : "off",
        dot: config.prefixHit > 0 ? "#b6f36b" : "#33404e",
      },
      { label: "Prefill skipped", value: fmtPct(m.prefillSavedFrac), dot: "#ffb347" },
      { label: "KV deduped", value: fmtPct(m.sharedSaving), dot: "#4de1ff" },
      {
        label: "Provider input",
        value: `${fmtMoney(config.inputPricePerM * m.providerBlendedMult)}/M`,
        dot: "#ff5c8a",
      },
    ];
    note = `Response caches bypass the GPU for ${fmtPct(m.responseHitRate)} of requests; prefix reuse skips ${fmtPct(m.prefillSavedFrac)} of prefill.`;
  } else if (lens === "topology") {
    title = "Why this shape";
    if (advice.reasons.length === 0) {
      rows = [
        { label: "Weights fit", value: `${fmtGB(m.weightsGB)}`, dot: "#b6f36b" },
        { label: "KV fits", value: `${fmtGB(m.kvTotalGB)}`, dot: "#b6f36b" },
        { label: "Latency target", value: "met", dot: "#b6f36b" },
      ];
    } else {
      rows = advice.reasons.map((r) => ({
        label: r.label,
        value: `${r.gpus}${r.label.includes("unreachable") || r.label.includes("beyond") ? "+" : ""} GPU${
          r.gpus === 1 ? "" : "s"
        }`,
        dot: "#ffb347",
      }));
    }
    const gaps: string[] = [];
    if (advice.maxUsersAt < config.users) gaps.push("capacity");
    if (advice.minDecode === 0) gaps.push("TPOT");
    if (advice.minPrefill === 0) gaps.push("TTFT");
    note = gaps.length
      ? `${advice.recommended} GPUs is the best available layout; ${gaps.join(
          " and ",
        )} still ${gaps.length === 1 ? "misses" : "miss"} at 64.`
      : `${advice.recommended === 1 ? "One GPU" : `${advice.recommended} GPUs`} at ${fmtNum(
          advice.maxUsersAt,
        )} KV slots · p95 TTFT ${fmtMs(advice.ttftAt)}.`;
    warning = advice.warnings[0] ?? "";
  }

  return (
    <div className="scene-card">
      <div className="scene-card-title">{title}</div>
      {rows.map((r) => (
        <div className="scene-row" key={r.label}>
          <span>
            {r.dot ? <i style={{ background: r.dot }} /> : null}
            {r.label}
          </span>
          <b>{r.value}</b>
        </div>
      ))}
      {note ? <div className="scene-note">{note}</div> : null}
      {warning ? <div className="scene-note warn">{warning}</div> : null}
    </div>
  );
}
