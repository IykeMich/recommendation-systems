"use client";

import type { UseMutationResult, UseQueryResult } from "@tanstack/react-query";
import type { ModelInfo } from "../lib/api";
import { formatDateTime, MODEL_LABELS } from "../lib/format";

/**
 * Side panel with model metadata, the offline evaluation table, top coefficients and the
 * feedback loop (pending events + Retrain button).
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
  const recallKey = evaluation ? `recall@${evaluation.k}` : "";
  // Highest recall across variants, used to highlight the winning row.
  const bestRecall = evaluation
    ? Math.max(...Object.values(evaluation.models).map((metrics) => metrics[recallKey]))
    : 0;

  return (
    <aside className="panel model-panel">
      <h2 className="section-title">Model</h2>

      {modelInfoQuery.isError && <p className="error-text small">{modelInfoQuery.error.message}</p>}

      {modelInfo && (
        <>
          <dl className="model-stats">
            <div>
              <dt>Ranker</dt>
              <dd>Logistic regression</dd>
            </div>
            <div>
              <dt>Trained</dt>
              <dd>{formatDateTime(modelInfo.trained_at)}</dd>
            </div>
            <div>
              <dt>Interactions</dt>
              <dd>{modelInfo.num_interactions.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Training rows</dt>
              <dd>{modelInfo.num_training_rows.toLocaleString()}</dd>
            </div>
          </dl>
          <p className="muted tiny">{modelInfo.label_rule}</p>

          {evaluation && (
            <section>
              <h3 className="section-title">Time-aware evaluation</h3>
              <p className="muted tiny">
                Trained before {formatDateTime(evaluation.cutoff)}, tested on {evaluation.test_requests} later
                clicks/redeems.
              </p>
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th scope="col">Model</th>
                    <th scope="col">Recall@{evaluation.k}</th>
                    <th scope="col">Diversity</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(evaluation.models).map(([modelName, metrics]) => (
                    <tr key={modelName}>
                      <th scope="row">{MODEL_LABELS[modelName] ?? modelName}</th>
                      <td className={metrics[recallKey] === bestRecall ? "metric-win" : ""}>
                        {metrics[recallKey].toFixed(3)}
                      </td>
                      <td>{metrics.diversity.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted tiny">
                On this dataset, context adds no measurable lift: preferred-category affinity carries the signal.
              </p>
            </section>
          )}

          <section>
            <h3 className="section-title">What the model learned</h3>
            <ul className="coefficients">
              {modelInfo.top_coefficients.slice(0, 8).map((coefficient) => (
                <li key={coefficient.feature} title={coefficient.feature}>
                  <span
                    className={`coefficient-value ${coefficient.coefficient >= 0 ? "positive" : "negative"}`}
                  >
                    {coefficient.coefficient >= 0 ? "+" : ""}
                    {coefficient.coefficient.toFixed(2)}
                  </span>
                  <span className="coefficient-label">{coefficient.label}</span>
                </li>
              ))}
            </ul>
          </section>

          {/* Retrain is only enabled when there is feedback the model hasn't been trained on. */}
          <section className="retrain">
            <h3 className="section-title">Feedback loop</h3>
            <p className="small">
              {pendingCount === 0 ? (
                <span className="muted">
                  No new feedback. Click or redeem an offer: history features update right away, and
                  the model’s weights update when you retrain.
                </span>
              ) : (
                <>
                  <strong>{pendingCount}</strong> new event{pendingCount === 1 ? "" : "s"} since training.
                  They already count in history features. Retrain to update the model’s weights.
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
