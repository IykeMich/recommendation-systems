"use client";

import type { UseMutationResult, UseQueryResult } from "@tanstack/react-query";
import type { ModelInfo } from "../lib/api";
import { formatDateTime, formatPercent, STRATEGY_LABELS } from "../lib/format";

/**
 * Right-hand panel: active eligibility rules, temporal evaluation of the four strategies, the
 * ranker's top coefficients, and the feedback loop with a Retrain button.
 */
export function ModelPanel({
  modelInfoQuery,
  retrainMutation,
}: {
  modelInfoQuery: UseQueryResult<ModelInfo, Error>;
  retrainMutation: UseMutationResult<ModelInfo, Error, void>;
}) {
  const modelInfo = modelInfoQuery.data;
  const evaluation = modelInfo?.evaluation;
  const pendingCount = modelInfo?.pending_new_interactions ?? 0;
  // Metric keys include K (e.g. "recall@5"), so build the key from the evaluation's k.
  const recallKey = evaluation ? `recall@${evaluation.k}` : "";

  return (
    <aside className="panel model-panel">
      {modelInfoQuery.isError && <p className="error-text small">{modelInfoQuery.error.message}</p>}

      {modelInfo && (
        <>
          <section>
            <h2 className="section-title">Eligibility rules · v{modelInfo.eligibility_rules_version}</h2>
            <ol className="rules-list">
              {modelInfo.eligibility_rules.map((rule) => (
                <li key={rule.rule_id}>
                  <code>{rule.rule_id}</code>
                  <span>{rule.description}</span>
                </li>
              ))}
            </ol>
            <p className="muted tiny">Educational rules mirroring the synthetic data, not real bank policy.</p>
          </section>

          {/* Strategy comparison: the "NO eligibility layer" row shows the share of ineligible slots. */}
          {evaluation && (
            <section>
              <h2 className="section-title">Temporal evaluation</h2>
              <p className="muted tiny">
                {evaluation.test_requests} applications after {formatDateTime(evaluation.cutoff)}; about{" "}
                {evaluation.mean_eligible_products_per_request} eligible products each.
              </p>
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th scope="col">Strategy</th>
                    <th scope="col">Recall@{evaluation.k}</th>
                    <th scope="col">Ineligible shown</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(evaluation.strategies).map(([strategyName, metrics]) => (
                    <tr key={strategyName}>
                      <th scope="row">{STRATEGY_LABELS[strategyName] ?? strategyName}</th>
                      <td>{metrics[recallKey].toFixed(3)}</td>
                      <td className={metrics.eligibility_violation_rate > 0 ? "metric-bad" : "metric-good"}>
                        {formatPercent(metrics.eligibility_violation_rate, 1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted tiny">
                No strategy beats random on this data: behaviour barely depends on the profile. Without the rule
                layer, {formatPercent(evaluation.strategies.ranker_without_eligibility?.requests_with_violation ?? 0)} of
                requests would show an ineligible product.
              </p>

              <h3 className="section-title spaced">Eligibility rate by risk profile</h3>
              <ul className="rate-bars">
                {Object.entries(evaluation.eligibility_rate_by_risk_profile).map(([riskProfile, rate]) => (
                  <li key={riskProfile}>
                    <span>{riskProfile}</span>
                    <span className="rate-track"><span style={{ width: formatPercent(rate) }} /></span>
                    <span className="tiny">{formatPercent(rate)}</span>
                  </li>
                ))}
              </ul>
              <h3 className="section-title spaced">By income quartile</h3>
              <ul className="rate-bars">
                {Object.entries(evaluation.eligibility_rate_by_income_quartile).map(([quartile, rate]) => (
                  <li key={quartile}>
                    <span>{quartile}</span>
                    <span className="rate-track"><span style={{ width: formatPercent(rate) }} /></span>
                    <span className="tiny">{formatPercent(rate)}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          <section>
            <h2 className="section-title">Ranker</h2>
            <p className="small">
              Logistic regression · <code>{modelInfo.model_version}</code>
            </p>
            <p className="muted tiny">{modelInfo.label_rule}</p>
            <ul className="coefficients">
              {modelInfo.top_coefficients.slice(0, 6).map((coefficient) => (
                <li key={coefficient.feature} title={coefficient.feature}>
                  <span className={`coefficient-value ${coefficient.coefficient >= 0 ? "positive" : "negative"}`}>
                    {coefficient.coefficient >= 0 ? "+" : ""}
                    {coefficient.coefficient.toFixed(2)}
                  </span>
                  <span>{coefficient.label}</span>
                </li>
              ))}
            </ul>
          </section>

          {/* Retrain is only enabled when there is new feedback to learn from. */}
          <section className="retrain">
            <h2 className="section-title">Feedback loop</h2>
            <p className="small">
              {pendingCount === 0 ? (
                <span className="muted">No new feedback. “Learn more” and “Apply” are recorded as interactions.</span>
              ) : (
                <>
                  <strong>{pendingCount}</strong> new event{pendingCount === 1 ? "" : "s"}. History-based reasons already
                  use them; retrain to update the model’s weights.
                </>
              )}
            </p>
            <button
              type="button"
              className="button full-width"
              disabled={pendingCount === 0 || retrainMutation.isPending}
              onClick={() => retrainMutation.mutate()}
            >
              {retrainMutation.isPending ? "Retraining…" : "Retrain model"}
            </button>
            {retrainMutation.isError && <p className="error-text small">{retrainMutation.error.message}</p>}
          </section>
        </>
      )}
    </aside>
  );
}
