import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { createPortal } from "react-dom";
import { useStore } from "../state/store";

type Placement = "below" | "right" | "left" | "above" | "inside";

type TourStep = {
  target: string;
  kicker: string;
  title: string;
  body: string;
  hint: string;
  placement: Placement;
};

type Rect = {
  top: number;
  left: number;
  width: number;
  height: number;
};

const TOUR: TourStep[] = [
  {
    target: '[data-tour="header"]',
    kicker: "1 · Pick a story",
    title: "Start with a scenario",
    body: "The presets are common serving shapes: startup chat, enterprise RAG, long documents, overloaded, frontier, and budget. The verdict in the top-right is the bottom line: fits, near capacity, missing an SLO, or does not fit.",
    hint: "Changing any control turns the scenario into Custom. You can always jump back with the dropdown.",
    placement: "below",
  },
  {
    target: '[data-tour="controls"]',
    kicker: "2 · Change the levers",
    title: "Model, hardware, workload — one at a time",
    body: "The left rail holds the four things that decide capacity: model precision, GPU memory and count, request shape and traffic, and cache hit rates. The small ⓘ buttons explain each idea.",
    hint: "Try raising context length or active users, then watch memory and TTFT respond.",
    placement: "right",
  },
  {
    target: '[data-tour="stage"]',
    kicker: "3 · Read the consequence",
    title: "Five lenses on the same system",
    body: "Overview shows the whole serving loop. Memory shows what occupies VRAM. Latency breaks down queue + prefill + decode. Caching shows work removed before the GPU — including hosted prompt-cache pricing. Topology explains when replicas beat tensor parallelism.",
    hint: "Cyan means cached, amber means compute, magenta means queueing, lime means fits, and red means overflow.",
    placement: "inside",
  },
  {
    target: '[data-tour="metrics"]',
    kicker: "4 · Check the constraint",
    title: "The right rail is the verdict, quantified",
    body: "Capacity is how many concurrent requests the KV cache can hold. Memory shows the hard VRAM wall. TTFT and throughput expose the latency/throughput trade-off. Constraint names the current bottleneck; Topology recommends a layout.",
    hint: "Turn on Show the math in the header to expose the formula behind every figure.",
    placement: "left",
  },
  {
    target: '[data-tour="charts"]',
    kicker: "5 · Find the knee",
    title: "Sensitivity beats a single number",
    body: "These charts sweep one variable while holding the rest fixed. The useful part is where the curve bends: where context kills concurrency, traffic makes queueing explode, or another GPU stops buying throughput.",
    hint: "The URL updates with your configuration, so you can share the exact setup you are looking at.",
    placement: "above",
  },
];

const CARD_WIDTH = 386;
const GAP = 14;

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function cardPosition(rect: Rect, placement: Placement): CSSProperties {
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  if (placement === "below") {
    const top = rect.top + rect.height + GAP;
    return {
      left: clamp(rect.left + rect.width / 2 - CARD_WIDTH / 2, GAP, vw - CARD_WIDTH - GAP),
      top,
      width: CARD_WIDTH,
      maxHeight: Math.max(220, vh - top - GAP),
    };
  }

  if (placement === "above") {
    const bottom = Math.max(GAP, vh - rect.top + GAP);
    return {
      left: clamp(rect.left + rect.width / 2 - CARD_WIDTH / 2, GAP, vw - CARD_WIDTH - GAP),
      bottom,
      width: CARD_WIDTH,
      maxHeight: Math.max(220, rect.top - GAP * 2),
    };
  }

  if (placement === "right") {
    const top = clamp(rect.top + 18, GAP, vh - 360);
    return {
      left: Math.min(rect.left + rect.width + GAP, vw - CARD_WIDTH - GAP),
      top,
      width: CARD_WIDTH,
      maxHeight: Math.max(220, vh - top - GAP),
    };
  }

  if (placement === "left") {
    const top = clamp(rect.top + 18, GAP, vh - 360);
    return {
      left: Math.max(GAP, rect.left - CARD_WIDTH - GAP),
      top,
      width: CARD_WIDTH,
      maxHeight: Math.max(220, vh - top - GAP),
    };
  }

  const top = Math.max(GAP, rect.top + 62);
  return {
    left: Math.max(GAP, rect.left + rect.width - CARD_WIDTH - 16),
    top,
    width: CARD_WIDTH,
    maxHeight: Math.max(220, rect.top + rect.height - top - 16),
  };
}

function useNarrowViewport() {
  const [narrow, setNarrow] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(max-width: 1080px)").matches,
  );

  useEffect(() => {
    const query = window.matchMedia("(max-width: 1080px)");
    const update = () => setNarrow(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  return narrow;
}

export function Onboarding() {
  const open = useStore((s) => s.onboardingOpen);
  const close = useStore((s) => s.closeOnboarding);
  const narrow = useNarrowViewport();
  const [step, setStep] = useState(0);
  const [targetRect, setTargetRect] = useState<Rect | null>(null);
  const cardRef = useRef<HTMLElement>(null);
  const current = step > 0 ? TOUR[step - 1] : null;
  const isWelcome = !current;
  const total = TOUR.length;

  const closeTour = useCallback(() => {
    setStep(0);
    setTargetRect(null);
    close();
  }, [close]);

  const goBack = useCallback(() => {
    setTargetRect(null);
    setStep((value) => Math.max(0, value - 1));
  }, []);

  const goNext = useCallback(() => {
    setTargetRect(null);
    setStep((value) => Math.min(TOUR.length, value + 1));
  }, []);

  const measure = useCallback(() => {
    if (!current) {
      setTargetRect(null);
      return;
    }
    const target = document.querySelector<HTMLElement>(current.target);
    if (!target) {
      setTargetRect(null);
      return;
    }
    const rect = target.getBoundingClientRect();
    setTargetRect({
      top: rect.top,
      left: rect.left,
      width: rect.width,
      height: rect.height,
    });
  }, [current]);

  useLayoutEffect(() => {
    if (!open || !current) return;

    const target = document.querySelector<HTMLElement>(current.target);
    if (!target) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    target.scrollIntoView({
      block: "center",
      inline: "nearest",
      behavior: reduceMotion ? "auto" : "smooth",
    });
    const frame = window.requestAnimationFrame(measure);
    const settle = window.setTimeout(measure, reduceMotion ? 0 : 380);
    const resizeObserver =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(measure);
    resizeObserver?.observe(target);
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);

    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(settle);
      resizeObserver?.disconnect();
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
    };
  }, [current, measure, open]);

  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    return () => {
      previous?.focus?.();
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const focus = window.setTimeout(() => cardRef.current?.focus(), 20);
    return () => window.clearTimeout(focus);
  }, [open, step]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeTour();
        return;
      }
      if (event.key !== "Tab") return;

      const card = cardRef.current;
      if (!card) return;
      const focusable = Array.from(
        card.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((element) => element.offsetParent !== null);

      if (focusable.length === 0) {
        event.preventDefault();
        card.focus();
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;

      if (!card.contains(active)) {
        event.preventDefault();
        first.focus();
      } else if (event.shiftKey && (active === first || active === card)) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (active === last || active === card)) {
        event.preventDefault();
        first.focus();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [closeTour, open]);

  if (!open) return null;

  const cardStyle =
    current && targetRect && !narrow
      ? cardPosition(targetRect, current.placement)
      : current && !narrow
        ? {
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
          }
        : undefined;
  const spotlightStyle = targetRect
    ? {
        top: targetRect.top,
        left: targetRect.left,
        width: targetRect.width,
        height: targetRect.height,
      }
    : undefined;

  return createPortal(
    <>
      <div className="onboarding-blocker" onClick={closeTour} aria-hidden="true" />
      {spotlightStyle ? (
        <div className="tour-spotlight" style={spotlightStyle} aria-hidden="true" />
      ) : (
        <div className="onboarding-scrim" aria-hidden="true" />
      )}

      <section
        ref={cardRef}
        className={`onboarding-card ${isWelcome ? "welcome" : ""}`}
        style={cardStyle}
        role="dialog"
        aria-modal="true"
        aria-labelledby="tour-title"
        tabIndex={-1}
      >
        <button
          className="tour-close"
          onClick={closeTour}
          aria-label="Close the tour"
          type="button"
        >
          ×
        </button>

        <div className="tour-kicker">
          <span>{isWelcome ? "Welcome to Capacity Lab" : current.kicker}</span>
          <span className="tour-count">
            {isWelcome ? "90 sec" : `${step} / ${total}`}
          </span>
        </div>

        {!isWelcome ? (
          <div className="tour-progress" aria-hidden="true">
            <span style={{ width: `${(step / total) * 100}%` }} />
          </div>
        ) : null}

        <h2 id="tour-title" className="tour-title">
          {isWelcome
            ? "See why an LLM deployment fits — or doesn’t."
            : current.title}
        </h2>

        {isWelcome ? (
          <>
            <p className="tour-body">
              This is one live serving configuration. Move the levers on the left; the stage
              and readouts show the consequences. The arithmetic is the same one a real
              inference server runs into.
            </p>

            <ul className="tour-points">
              <li>
                <i style={{ background: "#ffb347" }} />
                <div>
                  <b>VRAM is the hard limit.</b>
                  <span>
                    Weights, KV cache, and runtime overhead must all fit. If they don’t,
                    nothing is servable.
                  </span>
                </div>
              </li>
              <li>
                <i style={{ background: "#4de1ff" }} />
                <div>
                  <b>Context consumes capacity.</b>
                  <span>
                    Every active request holds its context in KV cache. Longer context means
                    fewer users per GPU.
                  </span>
                </div>
              </li>
              <li>
                <i style={{ background: "#b6f36b" }} />
                <div>
                  <b>Caching deletes work.</b>
                  <span>
                    Prefix hits skip prefill; response hits never reach the GPU. Both buy back
                    latency and slots.
                  </span>
                </div>
              </li>
            </ul>

            <div className="tour-legend" aria-label="Color language">
              <span><i style={{ background: "#4de1ff" }} />cached</span>
              <span><i style={{ background: "#ffb347" }} />compute</span>
              <span><i style={{ background: "#ff5c8a" }} />queue</span>
              <span><i style={{ background: "#b6f36b" }} />fits</span>
              <span><i style={{ background: "#ff5f5f" }} />overflow</span>
            </div>
          </>
        ) : (
          <>
            <p className="tour-body">{current.body}</p>
            <div className="tour-hint">{current.hint}</div>
          </>
        )}

        <div className="tour-actions">
          {isWelcome ? (
            <>
              <button className="tour-btn secondary" onClick={closeTour} type="button">
                Explore on my own
              </button>
              <button className="tour-btn primary" onClick={() => setStep(1)} type="button">
                Take the 90-second tour <span>→</span>
              </button>
            </>
          ) : (
            <>
              <button className="tour-skip" onClick={closeTour} type="button">
                Skip tour
              </button>
              <div className="tour-nav">
                <button
                  className="tour-btn secondary"
                  onClick={goBack}
                  type="button"
                >
                  Back
                </button>
                {step === total ? (
                  <button className="tour-btn primary" onClick={closeTour} type="button">
                    Start exploring <span>→</span>
                  </button>
                ) : (
                  <button
                    className="tour-btn primary"
                    onClick={goNext}
                    type="button"
                  >
                    Next <span>→</span>
                  </button>
                )}
              </div>
            </>
          )}
        </div>
      </section>
    </>,
    document.body,
  );
}
