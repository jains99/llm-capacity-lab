export const clamp = (v: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, v));

export function fmtGB(gb: number): string {
  if (!isFinite(gb)) return "∞";
  if (gb >= 1000) return `${(gb / 1000).toFixed(2)} TB`;
  if (gb >= 100) return `${gb.toFixed(0)} GB`;
  if (gb >= 10) return `${gb.toFixed(1)} GB`;
  return `${gb.toFixed(2)} GB`;
}

export function fmtMB(mb: number): string {
  if (mb >= 1000) return `${(mb / 1000).toFixed(2)} GB`;
  if (mb >= 10) return `${mb.toFixed(0)} MB`;
  return `${mb.toFixed(2)} MB`;
}

export function fmtMs(ms: number): string {
  if (!isFinite(ms)) return "—";
  if (ms >= 10000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  if (ms >= 100) return `${ms.toFixed(0)} ms`;
  if (ms >= 10) return `${ms.toFixed(1)} ms`;
  return `${ms.toFixed(2)} ms`;
}

export function fmtNum(n: number, digits = 0): string {
  if (!isFinite(n)) return "∞";
  return n.toLocaleString("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

export function fmtCompact(n: number): string {
  if (!isFinite(n)) return "∞";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 10_000) return `${(n / 1000).toFixed(0)}k`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return `${Math.round(n)}`;
}

export function fmtMoney(n: number): string {
  if (!isFinite(n)) return "—";
  if (n >= 100) return `$${n.toFixed(0)}`;
  if (n >= 10) return `$${n.toFixed(1)}`;
  return `$${n.toFixed(2)}`;
}

export function fmtPct(v: number, digits = 0): string {
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtTokens(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k`;
  return `${Math.round(n)}`;
}
