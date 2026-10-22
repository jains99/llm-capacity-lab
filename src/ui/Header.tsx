import { fmtMs, fmtNum } from "../lib/format";
import type { Metrics, TopologyAdvice } from "../lib/formulas";
import { PRESETS } from "../lib/presets";
import { useStore } from "../state/store";

function verdictFor(
  m: Metrics,
  advice: TopologyAdvice,
  targetTtft: number,
  targetTpot: number,
  users: number,
) {
  if (!m.fits || m.maxUsers === 0) {
    return {
      tone: "bad",
      title: "Does not fit",
      detail:
        advice.maxUsersAt < users
          ? `needs >${advice.recommended}× ${m.gpu.short}`
          : `needs ~${advice.recommended}× ${m.gpu.short}`,
    };
  }
  if (m.utilization >= 0.999 || m.ttftP95Ms >= 30000) {
    return {
      tone: "bad",
      title: "Overloaded",
      detail: `p95 TTFT ${fmtMs(m.ttftP95Ms)}`,
    };
  }
  if (m.ttftP95Ms > targetTtft) {
    return {
      tone: "warn",
      title: "Missing TTFT target",
      detail: `p95 ${fmtMs(m.ttftP95Ms)} vs ${fmtMs(targetTtft)}`,
    };
  }
  if (m.tpotMs > targetTpot) {
    return {
      tone: "warn",
      title: "Missing TPOT target",
      detail: `${Math.round(m.tpotMs)} ms/token vs ${targetTpot} ms`,
    };
  }
  if (m.utilization > 0.8) {
    return {
      tone: "warn",
      title: "Near capacity",
      detail: `${Math.round(m.utilization * 100)}% slots used`,
    };
  }
  return {
    tone: "good",
    title: `Fits · ${fmtNum(m.maxUsers)} slots`,
    detail: `p95 TTFT ${fmtMs(m.ttftP95Ms)}`,
  };
}

export function Header({ m, advice }: { m: Metrics; advice: TopologyAdvice }) {
  const presetId = useStore((s) => s.presetId);
  const applyPreset = useStore((s) => s.applyPreset);
  const showMath = useStore((s) => s.showMath);
  const setShowMath = useStore((s) => s.setShowMath);
  const openOnboarding = useStore((s) => s.openOnboarding);
  const targetTtft = useStore((s) => s.config.targetTtft);
  const targetTpot = useStore((s) => s.config.targetTpot);
  const users = useStore((s) => s.config.users);
  const v = verdictFor(m, advice, targetTtft, targetTpot, users);

  return (
    <header className="topbar" data-tour="header">
      <div className="brand">
        <div className="brand-mark" />
        <div>
          <h1>CAPACITY LAB</h1>
          <p>KV cache · batching · context · GPUs</p>
        </div>
      </div>

      <div className="scenario">
        <label htmlFor="scenario">Scenario</label>
        <select
          id="scenario"
          value={presetId}
          onChange={(e) => applyPreset(e.target.value)}
        >
          {presetId === "custom" ? <option value="custom">Custom config</option> : null}
          {PRESETS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name} — {p.tagline}
            </option>
          ))}
        </select>
        <span className="preset-tag">{presetId === "custom" ? "edited" : ""}</span>
      </div>

      <div className="spacer" />

      <button
        className="ghost-btn how-btn"
        onClick={openOnboarding}
        title="Replay the quick tour"
        aria-label="How it works — replay the onboarding tour"
      >
        <span className="help-glyph">?</span>
        <span className="how-label">How it works</span>
      </button>

      <button
        className={`ghost-btn ${showMath ? "on" : ""}`}
        onClick={() => setShowMath(!showMath)}
        title="Show the formulas behind each number"
      >
        <span className="dot" />
        Show the math
      </button>

      <div className={`verdict ${v.tone}`} role="status" aria-live="polite">
        <span className="vdot" />
        <b>{v.title}</b>
        <small>{v.detail}</small>
      </div>
    </header>
  );
}
