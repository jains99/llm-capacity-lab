import {
  fmtGB,
  fmtMs,
  fmtNum,
  fmtPct,
} from "../lib/format";
import type { Metrics } from "../lib/formulas";
import { GPUS, KV_QUANTS, MODELS, QUANTS, modelById } from "../lib/models";
import { useStore } from "../state/store";
import { Info, LogSlider, Section, Segmented, Slider, Toggle } from "./controls";

export function ControlRail({ m }: { m: Metrics }) {
  const config = useStore((s) => s.config);
  const set = useStore((s) => s.set);
  const model = m.model;
  const gpu = m.gpu;

  const modeLabel =
    m.mode === "tensor-parallel" ? "TP" : m.mode === "replicas" ? "replicas" : "single";

  return (
    <aside className="panel rail" aria-label="Configuration" data-tour="controls">
      <Section title="Model" note={`${m.weightsGB.toFixed(0)} GB weights`} info="quantization">
        <div className="field">
          <select
            className="select"
            value={config.modelId}
            onChange={(e) => {
              const next = modelById(e.target.value);
              set("modelId", next.id);
              if (config.context > next.maxContext) set("context", next.maxContext);
            }}
            aria-label="Model"
          >
            {MODELS.map((mo) => (
              <option key={mo.id} value={mo.id}>
                {mo.name} {mo.kind === "moe" ? "(MoE)" : ""}
              </option>
            ))}
          </select>
          <div className="field-hint">{model.note}</div>
        </div>

        <div className="field">
          <div className="field-top">
            <span className="field-label">
              Weight precision <Info topic="quantization" />
            </span>
            <span className="field-val">{m.quant.label}</span>
          </div>
          <Segmented
            options={QUANTS.map((q) => ({ id: q.id, label: q.label }))}
            value={config.quantId}
            onChange={(id) => set("quantId", id)}
          />
        </div>

        <div className="field">
          <div className="field-top">
            <span className="field-label">
              KV precision <Info topic="kv" />
            </span>
            <span className="field-val">{m.kvPerTokenMB.toFixed(3)} MB/token</span>
          </div>
          <Segmented
            options={KV_QUANTS.map((q) => ({ id: q.id, label: q.label }))}
            value={config.kvQuantId}
            onChange={(id) => set("kvQuantId", id)}
          />
        </div>
      </Section>

      <Section title="Hardware" note={`${m.gpu.vramGB} GB · ${m.gpu.bwGBs} GB/s`} info="vram">
        <div className="field">
          <select
            className="select"
            value={config.gpuId}
            onChange={(e) => set("gpuId", e.target.value)}
            aria-label="GPU"
          >
            {GPUS.map((g) => (
              <option key={g.id} value={g.id}>
                {g.name} — {g.vramGB} GB
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <div className="field-top">
            <span className="field-label">GPU count</span>
            <div className="stepper">
              <button
                onClick={() => set("gpuCount", Math.max(1, config.gpuCount - 1))}
                aria-label="Fewer GPUs"
              >
                −
              </button>
              <span className="count">{config.gpuCount}</span>
              <button
                onClick={() => set("gpuCount", Math.min(16, config.gpuCount + 1))}
                aria-label="More GPUs"
              >
                +
              </button>
              <span className={`tag ${m.mode === "replicas" ? "lime" : "cyan"}`}>
                {modeLabel}
              </span>
            </div>
          </div>
          <Slider
            label="Scale out"
            value={config.gpuCount}
            min={1}
            max={16}
            onChange={(v) => set("gpuCount", v)}
            format={(v) => `${v} × ${gpu.short}`}
          />
          {m.mode === "single" ? (
            <div className="field-hint">
              One GPU holds the weights and KV cache. No interconnect traffic.
            </div>
          ) : m.mode === "replicas" ? (
            <div className="field-hint">
              Each GPU holds a full copy of the weights and serves its own requests.
              No all-reduce, but the weights are duplicated.
            </div>
          ) : (
            <div className="field-hint">
              Weights and KV are split across GPUs. Every layer pays an all-reduce
              over {gpu.linkName}; sharding is used when replicas cannot hold the load.
            </div>
          )}
        </div>
      </Section>

      <Section title="Workload" note={`${Math.round(m.context / 1024)}k ctx`} info="users">
        <LogSlider
          label="Context per request"
          value={m.context}
          min={1024}
          max={model.maxContext}
          snap={512}
          onChange={(v) => set("context", v)}
          format={(v) => `${fmtNum(v)} tok`}
          hint={
            <>
              {fmtGB(m.kvPerUserGB)} KV per user · {fmtMs(m.prefillMs)} prefill ·{" "}
              {fmtPct(m.attentionShare)} attention
            </>
          }
        />
        <Slider
          label="Output tokens"
          value={config.outputTokens}
          min={16}
          max={2048}
          step={16}
          onChange={(v) => set("outputTokens", v)}
          format={(v) => `${v} tok`}
        />
        <LogSlider
          label="Active users"
          value={config.users}
          min={1}
          max={20000}
          onChange={(v) => set("users", v)}
          format={(v) => fmtNum(v)}
          accent="#ff5c8a"
          hint={
            <>
              {fmtNum(m.arrivalRate, 1)} req/s to GPU ·{" "}
              <b style={{ color: "#8496a9" }}>{fmtNum(m.concurrency)}</b> concurrent ·{" "}
              {fmtNum(m.maxUsers)} slots
            </>
          }
        />
        <Slider
          label="Think time"
          value={config.thinkTime}
          min={5}
          max={120}
          step={1}
          onChange={(v) => set("thinkTime", v)}
          format={(v) => `${v}s`}
          accent="#ff5c8a"
          hint="Seconds between a user's requests. Lower means more traffic."
        />
      </Section>

      <Section title="Caching" note={`${fmtPct(m.responseHitRate)} response hits`} info="prefix">
        <Toggle
          label="Exact response cache"
          checked={config.exactHit > 0}
          onChange={(v) => set("exactHit", v ? 0.12 : 0)}
        />
        {config.exactHit > 0 ? (
          <Slider
            label="Exact hit rate"
            value={config.exactHit}
            min={0.01}
            max={0.6}
            step={0.01}
            onChange={(v) => set("exactHit", v)}
            format={fmtPct}
            accent="#4de1ff"
          />
        ) : null}

        <Toggle
          label="Semantic cache"
          checked={config.semanticHit > 0}
          onChange={(v) => set("semanticHit", v ? 0.1 : 0)}
        />
        {config.semanticHit > 0 ? (
          <Slider
            label="Semantic hit rate"
            value={config.semanticHit}
            min={0.01}
            max={0.4}
            step={0.01}
            onChange={(v) => set("semanticHit", v)}
            format={fmtPct}
            accent="#a78bfa"
            hint="Higher hit rate means a looser similarity threshold and more wrong answers."
          />
        ) : null}

        <Toggle
          label="Prefix / KV reuse"
          checked={config.prefixHit > 0}
          onChange={(v) => set("prefixHit", v ? 0.75 : 0)}
        />
        {config.prefixHit > 0 ? (
          <>
            <Slider
              label="Prefix hit rate"
              value={config.prefixHit}
              min={0.05}
              max={1}
              step={0.01}
              onChange={(v) => set("prefixHit", v)}
              format={fmtPct}
              accent="#b6f36b"
            />
            <Slider
              label="Shared prefix"
              value={config.prefixShare}
              min={0}
              max={0.9}
              step={0.01}
              onChange={(v) => set("prefixShare", v)}
              format={fmtPct}
              accent="#b6f36b"
              hint={
                <>
                  {fmtPct(m.prefillSavedFrac)} of prefill skipped ·{" "}
                  {fmtGB(m.kvTotalGB)} KV total
                </>
              }
            />
          </>
        ) : null}
      </Section>

      <details className="adv">
        <summary>Advanced assumptions</summary>
        <div className="adv-body">
          <Slider
            label="Compute efficiency (MFU)"
            value={config.mfu}
            min={0.15}
            max={0.8}
            step={0.01}
            onChange={(v) => set("mfu", v)}
            format={fmtPct}
            accent="#ffb347"
            info="mfu"
          />
          <Slider
            label="Bandwidth efficiency (MBU)"
            value={config.mbu}
            min={0.2}
            max={0.9}
            step={0.01}
            onChange={(v) => set("mbu", v)}
            format={fmtPct}
            accent="#ffb347"
            info="mbu"
          />
          <Slider
            label="Runtime per GPU"
            value={config.overheadPerGpuGB}
            min={0.5}
            max={8}
            step={0.1}
            onChange={(v) => set("overheadPerGpuGB", v)}
            format={(v) => `${v.toFixed(1)} GB`}
            accent="#8496a9"
          />
          <Slider
            label="TTFT target"
            value={config.targetTtft}
            min={100}
            max={5000}
            step={50}
            onChange={(v) => set("targetTtft", v)}
            format={(v) => fmtMs(v)}
            accent="#ff5c8a"
          />
          <Slider
            label="TPOT target"
            value={config.targetTpot}
            min={10}
            max={200}
            step={1}
            onChange={(v) => set("targetTpot", v)}
            format={(v) => `${v} ms`}
            accent="#ff5c8a"
          />
        </div>
      </details>

      <details className="adv">
        <summary>Provider prompt cache</summary>
        <div className="adv-body">
          <div className="field-hint" style={{ marginTop: 0, marginBottom: 10 }}>
            Compare a hosted API's cached-input pricing with the self-hosted numbers on
            the right.
          </div>
          <Slider
            label="Input price"
            value={config.inputPricePerM}
            min={0.1}
            max={20}
            step={0.1}
            onChange={(v) => set("inputPricePerM", v)}
            format={(v) => `$${v.toFixed(2)}/M`}
            accent="#ffb347"
            info="provider"
          />
          <Slider
            label="Cached read multiplier"
            value={config.cacheReadMult}
            min={0}
            max={0.5}
            step={0.01}
            onChange={(v) => set("cacheReadMult", v)}
            format={(v) => `${v.toFixed(2)}×`}
            accent="#ffb347"
          />
          <Slider
            label="Cache write multiplier"
            value={config.cacheWriteMult}
            min={0}
            max={2}
            step={0.05}
            onChange={(v) => set("cacheWriteMult", v)}
            format={(v) => `${v.toFixed(2)}×`}
            accent="#ffb347"
          />
          <div className="field-hint">
            Blended input:{" "}
            <b style={{ color: "#e8eef6" }}>
              ${(config.inputPricePerM * m.providerBlendedMult).toFixed(2)}/M
            </b>{" "}
            at {fmtPct(config.prefixShare * config.prefixHit)} cached tokens.
          </div>
        </div>
      </details>
    </aside>
  );
}
