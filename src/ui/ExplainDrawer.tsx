import { useEffect } from "react";
import { EXPLAINERS } from "../lib/explain";
import { useStore } from "../state/store";

export function ExplainDrawer() {
  const topic = useStore((s) => s.explain);
  const setExplain = useStore((s) => s.setExplain);

  useEffect(() => {
    if (!topic) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setExplain(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [topic, setExplain]);

  if (!topic) return null;
  const doc = EXPLAINERS[topic];
  if (!doc) return null;

  return (
    <>
      <div className="drawer-backdrop" onClick={() => setExplain(null)} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={doc.title}>
        <div className="drawer-kicker">
          <span>{doc.kicker}</span>
          <button
            className="drawer-close"
            onClick={() => setExplain(null)}
            aria-label="Close"
          >
            ×
          </button>
        </div>
        <h2>{doc.title}</h2>
        {doc.body.map((p, i) => (
          <p key={i}>{p}</p>
        ))}
        {doc.formula ? <div className="formula-block">{doc.formula}</div> : null}
        {doc.source ? (
          <a
            className="source"
            href={doc.source.href}
            target="_blank"
            rel="noreferrer"
          >
            {doc.source.label} ↗
          </a>
        ) : null}
      </aside>
    </>
  );
}
