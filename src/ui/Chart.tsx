import { useId, useState, type PointerEvent as ReactPointerEvent } from "react";

export type Series = {
  label: string;
  color: string;
  values: number[];
};

type Props = {
  xs: number[];
  series: Series[];
  logX?: boolean;
  logY?: boolean;
  marker?: number;
  formatX: (n: number) => string;
  formatY: (n: number) => string;
};

const W = 300;
const H = 104;
const PAD_L = 34;
const PAD_R = 7;
const PAD_T = 8;
const PAD_B = 16;
const PW = W - PAD_L - PAD_R;
const PH = H - PAD_T - PAD_B;

export function Chart({ xs, series, logX, logY, marker, formatX, formatY }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const gid = useId().replace(/:/g, "");

  const all = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v));
  let yMin = logY ? Math.min(...all.filter((v) => v > 0)) : Math.min(0, ...all);
  let yMax = Math.max(...all, yMin + 1e-6);
  if (!logY) yMax += (yMax - yMin) * 0.08;
  if (yMax <= yMin) yMax = yMin + 1;

  const xMin = xs[0];
  const xMax = xs[xs.length - 1];

  const xPos = (v: number) => {
    if (xMax === xMin) return PAD_L + PW / 2;
    const t = logX
      ? (Math.log(Math.max(v, 1e-6)) - Math.log(xMin)) /
        (Math.log(xMax) - Math.log(xMin))
      : (v - xMin) / (xMax - xMin);
    return PAD_L + Math.max(0, Math.min(1, t)) * PW;
  };

  const yPos = (v: number) => {
    const lo = logY ? Math.log10(Math.max(yMin, 1e-9)) : yMin;
    const hi = logY ? Math.log10(Math.max(yMax, yMin + 1e-9)) : yMax;
    const vv = logY ? Math.log10(Math.max(v, Math.max(yMin, 1e-9))) : v;
    const t = hi === lo ? 0.5 : (vv - lo) / (hi - lo);
    return PAD_T + (1 - Math.max(0, Math.min(1, t))) * PH;
  };

  const nearest = (v: number) => {
    let best = 0;
    let bd = Infinity;
    xs.forEach((x, i) => {
      const d = Math.abs(x - v);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    return best;
  };

  const pathFor = (values: number[]) =>
    values
      .map((v, i) => `${i === 0 ? "M" : "L"}${xPos(xs[i]).toFixed(1)},${yPos(v).toFixed(1)}`)
      .join(" ");

  const first = series[0];
  const areaPath = first
    ? `${pathFor(first.values)} L${xPos(xs[xs.length - 1]).toFixed(1)},${(
        PAD_T + PH
      ).toFixed(1)} L${xPos(xs[0]).toFixed(1)},${(PAD_T + PH).toFixed(1)} Z`
    : "";

  const markerIdx = marker !== undefined ? nearest(marker) : null;

  const onMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const rel = ((e.clientX - rect.left) / rect.width) * W;
    let best = 0;
    let bd = Infinity;
    xs.forEach((x, i) => {
      const d = Math.abs(xPos(x) - rel);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    setHover(best);
  };

  const hi = hover !== null ? hover : markerIdx;
  const tooltipLeft =
    hi !== null
      ? Math.max(12, Math.min(88, (xPos(xs[hi]) / W) * 100))
      : 50;

  return (
    <div style={{ position: "relative" }}>
      <svg
        className="chart-svg"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
        role="img"
        aria-label={series.map((s) => s.label).join(", ")}
      >
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={first?.color ?? "#4de1ff"} stopOpacity="0.08" />
            <stop offset="100%" stopColor={first?.color ?? "#4de1ff"} stopOpacity="0" />
          </linearGradient>
        </defs>

        {[0, 0.5, 1].map((t) => (
          <line
            key={t}
            x1={PAD_L}
            x2={W - PAD_R}
            y1={PAD_T + t * PH}
            y2={PAD_T + t * PH}
            stroke="#16202b"
            strokeWidth="0.6"
          />
        ))}

        <text x={2} y={PAD_T + 4}>
          {formatY(yMax)}
        </text>
        <text x={2} y={PAD_T + PH}>
          {formatY(yMin)}
        </text>
        <text x={PAD_L} y={H - 4}>
          {formatX(xs[0])}
        </text>
        <text x={W - PAD_R} y={H - 4} textAnchor="end">
          {formatX(xs[xs.length - 1])}
        </text>

        {first && !logY ? <path d={areaPath} fill={`url(#${gid})`} /> : null}

        {series.map((s) => (
          <path
            key={s.label}
            d={pathFor(s.values)}
            fill="none"
            stroke={s.color}
            strokeWidth="1.5"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}

        {hi !== null ? (
          <line
            x1={xPos(xs[hi])}
            x2={xPos(xs[hi])}
            y1={PAD_T}
            y2={PAD_T + PH}
            stroke="#5f7488"
            strokeWidth="0.7"
            strokeDasharray="2 3"
          />
        ) : null}

        {hi !== null
          ? series.map((s) => (
              <circle
                key={s.label}
                cx={xPos(xs[hi])}
                cy={yPos(s.values[hi])}
                r="2.4"
                fill={s.color}
                stroke="#05070a"
                strokeWidth="1"
              />
            ))
          : null}
      </svg>

      {hover !== null ? (
        <div className="chart-tip" style={{ left: `${tooltipLeft}%`, top: 30 }}>
          {formatX(xs[hover])}
          {series.map((s) => (
            <div key={s.label}>
              <em>{s.label}:</em> {formatY(s.values[hover])}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
