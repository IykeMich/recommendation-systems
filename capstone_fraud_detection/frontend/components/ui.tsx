import type { Decision } from "../lib/api";

/** Coloured label for a decision; the CSS class (e.g. decision-review) picks the colour. */
export function DecisionBadge({ decision }: { decision: Decision }) {
  return <span className={`decision decision-${decision.toLowerCase()}`}>{decision}</span>;
}

/** Horizontal bar plus the numeric value for a 0–1 risk score. */
export function RiskBar({ riskScore }: { riskScore: number }) {
  // Colour bands use policy-v1's default thresholds (0.55 review, 0.85 block), not the live policy.
  const level = riskScore >= 0.85 ? "high" : riskScore >= 0.55 ? "medium" : "low";
  return (
    <span className="risk-bar" title={`Risk score ${riskScore.toFixed(3)}`}>
      <span className="risk-track" aria-hidden="true">
        {/* At least 2% wide so very low scores still show a sliver. */}
        <span className={`risk-fill risk-${level}`} style={{ width: `${Math.max(2, riskScore * 100)}%` }} />
      </span>
      <span className="risk-value">{riskScore.toFixed(3)}</span>
    </span>
  );
}

/** Shows a query/mutation error message (already made readable by requestJson), or nothing. */
export function PanelError({ error }: { error: Error | null }) {
  return error ? <p className="error-text small">{error.message}</p> : null;
}
