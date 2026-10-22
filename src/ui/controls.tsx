import type { CSSProperties, ReactNode } from "react";
import { useStore } from "../state/store";

export function Info({ topic }: { topic: string }) {
  const setExplain = useStore((s) => s.setExplain);
  return (
    <button
      className="info"
      onClick={() => setExplain(topic)}
      aria-label={`Explain ${topic}`}
      title="What is this?"
    >
      i
    </button>
  );
}

export function Section({
  title,
  note,
  info,
  children,
}: {
  title: string;
  note?: ReactNode;
  info?: string;
  children: ReactNode;
}) {
  return (
    <section className="sec">
      <div className="sec-head">
        <span className="sec-title">
          {title} {info ? <Info topic={info} /> : null}
        </span>
        {note ? <span className="sec-note">{note}</span> : null}
      </div>
      {children}
    </section>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format,
  accent = "#4de1ff",
  hint,
  info,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  accent?: string;
  hint?: ReactNode;
  info?: string;
}) {
  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
  return (
    <div className="field">
      <div className="field-top">
        <span className="field-label">
          {label} {info ? <Info topic={info} /> : null}
        </span>
        <span className="field-val">{format ? format(value) : value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ "--pct": `${pct}%`, "--accent": accent } as CSSProperties}
        aria-label={label}
      />
      {hint ? <div className="field-hint">{hint}</div> : null}
    </div>
  );
}

export function LogSlider({
  label,
  value,
  min,
  max,
  snap = 1,
  onChange,
  format,
  accent = "#4de1ff",
  hint,
  info,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  snap?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  accent?: string;
  hint?: ReactNode;
  info?: string;
}) {
  const t = Math.log(Math.max(value, min) / min) / Math.log(max / min);
  const pct = Math.max(0, Math.min(100, t * 100));
  const fromSlider = (p: number) => {
    const raw = min * Math.pow(max / min, p / 100);
    return Math.max(min, Math.round(raw / snap) * snap);
  };
  return (
    <div className="field">
      <div className="field-top">
        <span className="field-label">
          {label} {info ? <Info topic={info} /> : null}
        </span>
        <span className="field-val">{format ? format(value) : value}</span>
      </div>
      <input
        type="range"
        min={0}
        max={100}
        step={0.5}
        value={pct}
        onChange={(e) => onChange(fromSlider(Number(e.target.value)))}
        style={{ "--pct": `${pct}%`, "--accent": accent } as CSSProperties}
        aria-label={label}
      />
      {hint ? <div className="field-hint">{hint}</div> : null}
    </div>
  );
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { id: T; label: string }[];
  value: T;
  onChange: (id: T) => void;
}) {
  return (
    <div className="seg" role="group">
      {options.map((o) => (
        <button
          key={o.id}
          className={o.id === value ? "on" : ""}
          onClick={() => onChange(o.id)}
          type="button"
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="tgl">
      <span>{label}</span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="tgl-track" />
    </label>
  );
}
