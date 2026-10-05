"use client";

import { useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type Outcome } from "../lib/api";
import { formatAmount, formatDateTime, formatPercent } from "../lib/format";
import { DecisionBadge, PanelError, RiskBar } from "./ui";

/**
 * Side drawer with everything about one decision: evidence-backed reasons, context facts, the feature
 * snapshot the model scored, the customer's reference profile, analyst outcome buttons and the raw
 * request. Closes via the Close button, a backdrop click or the Escape key.
 */
export function AlertDrawer({ transactionId, onClose }: { transactionId: string; onClose: () => void }) {
  const queryClient = useQueryClient();
  // ["decision", id]: full detail for this transaction, cached per ID (no polling).
  const detailQuery = useQuery({ queryKey: ["decision", transactionId], queryFn: () => api.decisionDetail(transactionId) });
  // Record the analyst's outcome. Afterwards, invalidate everything that shows it: this drawer's detail,
  // every filtered decisions list (prefix ["decisions"] matches all filters) and the KPI stats.
  const outcomeMutation = useMutation({
    mutationFn: (outcome: Outcome) => api.recordOutcome(transactionId, outcome),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["decision", transactionId] });
      queryClient.invalidateQueries({ queryKey: ["decisions"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  // Escape closes the drawer. The listener is removed on unmount (or when onClose changes) to avoid leaks.
  useEffect(() => {
    const closeOnEscape = (keyEvent: KeyboardEvent) => keyEvent.key === "Escape" && onClose();
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  const detail = detailQuery.data;
  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} aria-hidden="true" />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={`Transaction ${transactionId}`}>
        <div className="drawer-header">
          <div>
            <p className="muted small">Transaction</p>
            <h2><code>{transactionId}</code></h2>
          </div>
          <button type="button" className="button secondary" onClick={onClose}>Close</button>
        </div>
        <PanelError error={detailQuery.error} />
        {detail && (
          <div className="drawer-body">
            {/* Decision, score and the model/policy versions it was made under */}
            <div className="drawer-summary">
              <DecisionBadge decision={detail.decision} />
              <RiskBar riskScore={detail.risk_score} />
              <span className="muted small">{detail.model_version} · {detail.policy_version}</span>
            </div>

            {/* Reasons with training evidence; the fallback reason (feature null) has no evidence line. */}
            <section>
              <h3 className="section-title">Why</h3>
              <ul className="reasons">
                {detail.reasons.map((reason) => (
                  <li key={reason.text}>
                    <span>{reason.text}</span>
                    {reason.evidence.feature && reason.evidence.training_fraud_rate !== undefined && (
                      <span className="evidence">
                        training fraud rate {formatPercent(reason.evidence.training_fraud_rate)} vs{" "}
                        {formatPercent(reason.evidence.overall_fraud_rate ?? 0)} overall · {reason.evidence.lift}× ·{" "}
                        {reason.evidence.training_transactions} transactions
                      </span>
                    )}
                  </li>
                ))}
              </ul>
              {detail.context.length > 0 && (
                <>
                  <h3 className="section-title spaced">Context <span className="muted">(facts, not risk drivers in this data)</span></h3>
                  <ul className="context-list">
                    {detail.context.map((contextLine) => <li key={contextLine}>{contextLine}</li>)}
                  </ul>
                </>
              )}
            </section>

            {/* Feature snapshot: the server-computed inputs the model scored */}
            <section>
              <h3 className="section-title">Feature snapshot</h3>
              <dl className="snapshot">
                <dt>Amount</dt><dd>{formatAmount(detail.features.amount)}</dd>
                <dt>Hour</dt><dd>{detail.features.hour}:00</dd>
                <dt>Velocity (1h)</dt><dd>{detail.features.velocity_1h} <span className="muted small">({detail.features.velocity_source})</span></dd>
                <dt>Country · device</dt><dd>{detail.features.country} · {detail.features.device}</dd>
                <dt>Merchant category</dt><dd>{detail.features.merchant_category}</dd>
                <dt>Customer history</dt><dd>{detail.features.customer_txn_count} reference transactions</dd>
                <dt>Amount vs usual</dt><dd>{detail.features.amount_vs_customer_mean === null ? "—" : `${detail.features.amount_vs_customer_mean.toFixed(2)}×`}</dd>
                <dt>Device / country seen</dt><dd>{formatPercent(detail.features.customer_device_share, 0)} / {formatPercent(detail.features.customer_country_share, 0)}</dd>
              </dl>
            </section>

            {/* Customer profile from training history; hidden for customers with no reference data. */}
            {detail.customer_profile.known && (
              <section>
                <h3 className="section-title">Customer {detail.customer_id}</h3>
                <p className="small">
                  {detail.customer_profile.reference_transactions} reference transactions, average{" "}
                  {formatAmount(detail.customer_profile.mean_amount)}
                </p>
                <p className="muted small">
                  Devices: {Object.entries(detail.customer_profile.devices).map(([device, count]) => `${device} ${count}`).join(", ")} ·
                  Countries: {Object.entries(detail.customer_profile.countries).map(([country, count]) => `${country} ${count}`).join(", ")}
                </p>
              </section>
            )}

            {/* Analyst outcome. For replayed rows the synthetic label is shown as the known answer. */}
            <section>
              <h3 className="section-title">Outcome</h3>
              <p className="muted small">
                An alert is a signal, not confirmed fraud. Record what the review found.
                {detail.synthetic_label !== null && (
                  <> Synthetic label for this replayed transaction: <strong>{detail.synthetic_label ? "fraud" : "legitimate"}</strong>.</>
                )}
              </p>
              {/* aria-pressed marks whichever outcome is already recorded. */}
              <div className="outcome-actions">
                <button type="button" className="button danger" disabled={outcomeMutation.isPending}
                  aria-pressed={detail.outcome === "confirmed_fraud"} onClick={() => outcomeMutation.mutate("confirmed_fraud")}>
                  Confirm fraud
                </button>
                <button type="button" className="button secondary" disabled={outcomeMutation.isPending}
                  aria-pressed={detail.outcome === "legitimate"} onClick={() => outcomeMutation.mutate("legitimate")}>
                  Mark legitimate
                </button>
              </div>
              {detail.outcome && <p className="small">Recorded <strong>{detail.outcome.replace("_", " ")}</strong> at {formatDateTime(detail.outcome_at)}</p>}
              <PanelError error={outcomeMutation.error} />
            </section>

            <details>
              <summary className="small">Raw request payload</summary>
              <pre className="payload">{JSON.stringify(detail.payload, null, 2)}</pre>
            </details>
          </div>
        )}
      </aside>
    </>
  );
}
