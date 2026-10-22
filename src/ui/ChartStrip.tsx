import { useMemo } from "react";
import { fmtCompact, fmtMs, fmtNum } from "../lib/format";
import { computeMetrics, type EngineConfig, type Metrics } from "../lib/formulas";
import { Chart } from "./Chart";

export function ChartStrip({ config, m }: { config: EngineConfig; m: Metrics }) {
  const contextData = useMemo(() => {
    const xs = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072].filter(
      (x) => x <= m.model.maxContext,
    );
    if (xs[xs.length - 1] !== m.model.maxContext) xs.push(m.model.maxContext);
    return {
      xs,
      values: xs.map((x) => computeMetrics({ ...config, context: x }).maxUsers),
    };
  }, [config, m.model.maxContext]);

  const userData = useMemo(() => {
    const xs = Array.from({ length: 26 }, (_, i) =>
      Math.round(8 * Math.pow(2500, i / 25)),
    );
    return {
      xs,
      values: xs.map((users) =>
        Math.min(60000, computeMetrics({ ...config, users }).ttftP95Ms),
      ),
    };
  }, [config]);

  const cacheData = useMemo(() => {
    const xs = Array.from({ length: 20 }, (_, i) => i * 0.05);
    return {
      xs,
      values: xs.map((hit) =>
        computeMetrics({ ...config, prefixHit: hit }).ttftP95Ms,
      ),
    };
  }, [config]);

  const gpuData = useMemo(() => {
    const xs = [1, 2, 3, 4, 5, 6, 7, 8];
    return {
      xs,
      values: xs.map((gpuCount) =>
        computeMetrics({ ...config, gpuCount }).outputTps,
      ),
    };
  }, [config]);

  return (
    <div className="charts" aria-label="Sensitivity charts" data-tour="charts">
      <div className="panel chart-card">
        <div className="chart-head">
          <span className="chart-title">Context → concurrency</span>
          <span className="chart-now">{fmtNum(m.maxUsers)} users</span>
        </div>
        <div className="chart-sub">bigger context, fewer resident users</div>
        <Chart
          xs={contextData.xs}
          series={[
            { label: "max users", color: "#4de1ff", values: contextData.values },
          ]}
          marker={m.context}
          logX
          formatX={(v) => `${Math.round(v / 1024)}k`}
          formatY={(v) => fmtCompact(v)}
        />
      </div>

      <div className="panel chart-card">
        <div className="chart-head">
          <span className="chart-title">Traffic → p95 TTFT</span>
          <span className="chart-now">{fmtMs(m.ttftP95Ms)}</span>
        </div>
        <div className="chart-sub">queueing is non-linear at the knee</div>
        <Chart
          xs={userData.xs}
          series={[
            { label: "p95 TTFT", color: "#ff5c8a", values: userData.values },
          ]}
          marker={config.users}
          logX
          logY
          formatX={(v) => fmtCompact(v)}
          formatY={fmtMs}
        />
      </div>

      <div className="panel chart-card">
        <div className="chart-head">
          <span className="chart-title">Prefix cache → TTFT</span>
          <span className="chart-now">{fmtMs(m.ttftP95Ms)}</span>
        </div>
        <div className="chart-sub">skipped prefill shows up immediately</div>
        <Chart
          xs={cacheData.xs}
          series={[
            { label: "p95 TTFT", color: "#b6f36b", values: cacheData.values },
          ]}
          marker={config.prefixHit}
          formatX={(v) => `${Math.round(v * 100)}%`}
          formatY={fmtMs}
        />
      </div>

      <div className="panel chart-card">
        <div className="chart-head">
          <span className="chart-title">GPUs → throughput</span>
          <span className="chart-now">{fmtCompact(m.outputTps)} tok/s</span>
        </div>
        <div className="chart-sub">sublinear: interconnect takes a cut</div>
        <Chart
          xs={gpuData.xs}
          series={[
            { label: "tok/s", color: "#ffb347", values: gpuData.values },
          ]}
          marker={config.gpuCount}
          formatX={(v) => `${v}`}
          formatY={fmtCompact}
        />
      </div>
    </div>
  );
}
