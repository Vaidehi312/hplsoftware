// Small reusable pieces shared by the pipeline components — the JS
// equivalents of st.success/st.warning/st.error/st.info/st.expander/
// st.code/st.progress. Not one of the files the task enumerated, but
// splitting these out is what keeps every stage component from
// reimplementing its own box/collapsible.
import { useEffect, useRef, useState } from "react";

export function Alert({ type = "info", children }) {
  if (!children && children !== 0) return null;
  return <div className={`pipeline-alert pipeline-alert-${type}`}>{children}</div>;
}

export function Caption({ children, muted = true }) {
  if (children === null || children === undefined || children === "") return null;
  return <div className={`pipeline-caption${muted ? "" : " pipeline-caption-strong"}`}>{children}</div>;
}

export function CodeBlock({ children }) {
  return <pre className="pipeline-code">{children}</pre>;
}

// resetKey mirrors Streamlit's st.expander(expanded=...) behavior when the
// label itself carries the state (as _render_job_progress's step expanders
// do): without an explicit widget key, Streamlit treats a label change as a
// new widget and re-applies `expanded`, while an unchanged label preserves
// whatever the user toggled. Passing a value that changes when the step's
// state/summary changes reproduces that — the expander reopens/recloses
// with the state, but a manual toggle sticks until the state text changes.
export function Expander({ title, defaultOpen = false, children, resetKey }) {
  const [open, setOpen] = useState(defaultOpen);
  const prevResetKey = useRef(resetKey);
  useEffect(() => {
    if (resetKey !== undefined && resetKey !== prevResetKey.current) {
      setOpen(defaultOpen);
      prevResetKey.current = resetKey;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resetKey]);
  return (
    <details
      className="pipeline-expander"
      open={open}
      onToggle={(e) => setOpen(e.target.open)}
    >
      <summary className="pipeline-expander-summary">{title}</summary>
      <div className="pipeline-expander-body">{children}</div>
    </details>
  );
}

export function ProgressBar({ fraction, label }) {
  const pct = Math.max(0, Math.min(1, fraction || 0)) * 100;
  return (
    <div className="pipeline-progress">
      <div className="pipeline-progress-track">
        <div className="pipeline-progress-fill" style={{ width: `${pct}%` }} />
      </div>
      {label && <div className="pipeline-progress-label">{label}</div>}
    </div>
  );
}

export function Button({ children, onClick, disabled, kind = "default", ...rest }) {
  return (
    <button
      type="button"
      className={`pipeline-btn pipeline-btn-${kind}`}
      onClick={onClick}
      disabled={disabled}
      {...rest}
    >
      {children}
    </button>
  );
}

export function Field({ label, help, children }) {
  return (
    <label className="pipeline-field">
      <span className="pipeline-field-label">{label}</span>
      {children}
      {help && <span className="pipeline-field-help">{help}</span>}
    </label>
  );
}

export function RadioGroup({ label, options, value, onChange, help, name }) {
  return (
    <div className="pipeline-field">
      {label && <span className="pipeline-field-label">{label}</span>}
      <div className="pipeline-radio-group">
        {options.map((opt) => {
          const optValue = typeof opt === "string" ? opt : opt.value;
          const optLabel = typeof opt === "string" ? opt : opt.label;
          return (
            <label className="pipeline-radio-option" key={optValue}>
              <input
                type="radio"
                name={name}
                checked={value === optValue}
                onChange={() => onChange(optValue)}
              />
              {optLabel}
            </label>
          );
        })}
      </div>
      {help && <span className="pipeline-field-help">{help}</span>}
    </div>
  );
}

export function Metric({ label, value, caption }) {
  return (
    <div className="pipeline-metric">
      <div className="pipeline-metric-label">{label}</div>
      <div className="pipeline-metric-value">{value}</div>
      {caption && <div className="pipeline-metric-caption">{caption}</div>}
    </div>
  );
}

export function Spinner({ children }) {
  return (
    <div className="pipeline-spinner">
      <span className="pipeline-spinner-dot" />
      {children}
    </div>
  );
}
