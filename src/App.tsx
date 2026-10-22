import { useEffect, useMemo } from "react";
import {
  fmtCompact,
  fmtGB,
  fmtMs,
  fmtPct,
} from "./lib/format";
import { adviseTopology, computeMetrics, type Metrics, type TopologyAdvice } from "./lib/formulas";
import { useStore, type Lens } from "./state/store";
import { ChartStrip } from "./ui/ChartStrip";
import { ControlRail } from "./ui/ControlRail";
import { ExplainDrawer } from "./ui/ExplainDrawer";
import { Header } from "./ui/Header";
import { InsightCard } from "./ui/InsightCard";
import { MetricRail } from "./ui/MetricRail";
import { Onboarding } from "./ui/Onboarding";
import { Stage } from "./three/Stage";

const LENSES: { id: Lens; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "memory", label: "Memory" },
  { id: "latency", label: "Latency" },
  { id: "caching", label: "Caching" },
  { id: "topology", label: "Topology" },
];

function Foot({
  m,
  advice,
  lens,
}: {
  m: Metrics;
  advice: TopologyAdvice;
  lens: Lens;
}) {
  const sep = <span className="sep" />;
  const inputPrice = useStore((s) => s.config.inputPricePerM);
  const mode =
    m.mode === "tensor-parallel" ? "tensor parallel" : m.mode === "replicas" ? "replicas" : "single";

  if (lens === "memory") {
    return (
      <>
        <span className="lead">
          KV cache is <b>{fmtPct(m.kvTotalGB / Math.max(1, m.usedGB))}</b> of everything
          resident
        </span>
        {sep}
        <span>weights <b>{fmtGB(m.weightsInstalledGB)}</b></span>
        {sep}
        <span>KV <b>{fmtGB(m.kvTotalGB)}</b></span>
        {sep}
        <span><b>{fmtGB(m.kvPerUserGB)}</b> per user</span>
        {sep}
        <span>runtime <b>{fmtGB(m.overheadGB)}</b></span>
      </>
    );
  }

  if (lens === "latency") {
    return (
      <>
        <span className="lead">
          p95 TTFT <b>{fmtMs(m.ttftP95Ms)}</b>
        </span>
        {sep}
        <span>queue <b>{fmtMs(m.queueWaitP95Ms)}</b></span>
        {sep}
        <span>prefill <b>{fmtMs(m.prefillMs)}</b></span>
        {sep}
        <span>TPOT <b>{Math.round(m.tpotMs)} ms</b></span>
        {sep}
        <span>batch <b>{Math.round(m.batch)}</b></span>
      </>
    );
  }

  if (lens === "caching") {
    return (
      <>
        <span className="lead">
          <b>{fmtPct(m.responseHitRate)}</b> of responses never reach the GPU
        </span>
        {sep}
        <span>prefix saves <b>{fmtPct(m.prefillSavedFrac)}</b> of prefill</span>
        {sep}
        <span>KV dedupe <b>{fmtPct(m.sharedSaving)}</b></span>
        {sep}
        <span>provider cached input <b>${(inputPrice * m.providerBlendedMult).toFixed(2)}/M</b></span>
      </>
    );
  }

  if (lens === "topology") {
    return (
      <>
        <span className="lead">
          {advice.recommended} × {m.gpu.short} · {advice.mode}
        </span>
        {sep}
        {advice.mode === "replicas" ? (
          <span>independent copies · no all-reduce</span>
        ) : (
          <span>scaling <b>{fmtPct(advice.efficiency)}</b></span>
        )}
        {sep}
        <span>TTFT at size <b>{fmtMs(advice.ttftAt)}</b></span>
        {sep}
        <span>TPOT <b>{Math.round(advice.tpotAt)} ms</b></span>
        {sep}
        <span>link <b>{m.gpu.linkName}</b></span>
      </>
    );
  }

  return (
    <>
      <span className="lead">
        Batch <b>{Math.round(m.batch)}</b> · {m.n} × {m.gpu.short} · {mode}
      </span>
      {sep}
      <span><b>{fmtPct(m.utilization)}</b> slots used</span>
      {sep}
      <span><b>{fmtPct(m.responseHitRate)}</b> cached</span>
      {sep}
      <span><b>{fmtGB(m.usedGB)}</b> / {fmtGB(m.capacityGB)}</span>
      {sep}
      <span>GPU {fmtCompact(m.outputTps)} tok/s</span>
    </>
  );
}

export default function App() {
  const config = useStore((s) => s.config);
  const lens = useStore((s) => s.lens);
  const setLens = useStore((s) => s.setLens);

  const m = useMemo(() => computeMetrics(config), [config]);
  const advice = useMemo(() => adviseTopology(config), [config]);

  useEffect(() => {
    const params = new URLSearchParams();
    Object.entries(config).forEach(([k, v]) => params.set(k, String(v)));
    const id = window.setTimeout(() => {
      window.history.replaceState(null, "", `?${params.toString()}`);
    }, 180);
    return () => window.clearTimeout(id);
  }, [config]);

  return (
    <div className="app">
      <Header m={m} advice={advice} />

      <div className="workspace">
        <ControlRail m={m} />

        <section className="stage" aria-label="Visualisation" data-tour="stage">
          <Stage m={m} advice={advice} lens={lens} />

          <div className="stage-tabs" role="tablist" aria-label="View">
            {LENSES.map((l) => (
              <button
                key={l.id}
                role="tab"
                aria-selected={lens === l.id}
                className={lens === l.id ? "on" : ""}
                onClick={() => setLens(l.id)}
              >
                {l.label}
              </button>
            ))}
          </div>

          <div className="stage-chip">
            {m.model.name} · {m.quant.label} · <b>{m.n}× {m.gpu.short}</b>
          </div>

          <InsightCard m={m} advice={advice} config={config} lens={lens} />

          <div className="stage-foot">
            <Foot m={m} advice={advice} lens={lens} />
          </div>
        </section>

        <MetricRail m={m} advice={advice} />
      </div>

      <ChartStrip config={config} m={m} />
      <ExplainDrawer />
      <Onboarding />
    </div>
  );
}
