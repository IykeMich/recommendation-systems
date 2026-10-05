"use client";

import type { UseMutationResult, UseQueryResult } from "@tanstack/react-query";
import type { ModelInfo } from "../lib/api";
import { formatDateTime, formatPercent } from "../lib/format";

// Display names for metric keys from the API ("recall@10" -> "Recall@10").
const METRIC_LABELS: Record<string, string> = {
  recall: "Recall",
  precision: "Precision",
  ndcg: "NDCG",
  coverage: "Coverage",
};

/** Turn an API metric key like "ndcg@10" into a label like "NDCG@10". */
function metricLabel(metricKey: string): string {
  const [metricName, cutoff] = metricKey.split("@");
  return `${METRIC_LABELS[metricName] ?? metricName}${cutoff ? `@${cutoff}` : ""}`;
}

/**
 * Right-hand panel: model metadata, offline metrics vs the popularity baseline, and the retrain
 * button. The query and mutation live in page.tsx so other components share their state.
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

  return (
    <aside className="panel model-panel">
      <h2 className="section-title">Model</h2>

      {modelInfoQuery.isError && (
        <p className="error-text small">{modelInfoQuery.error.message}</p>
      )}

      {modelInfo && (
        <>
          <dl className="model-stats">
            <div>
              <dt>Algorithm</dt>
              <dd>Adaptive hybrid (content + item-item)</dd>
            </div>
            <div>
              <dt>Trained</dt>
              <dd>{formatDateTime(modelInfo.trained_at)}</dd>
            </div>
            <div>
              <dt>Matrix</dt>
              <dd>
                {modelInfo.num_users.toLocaleString()} × {modelInfo.num_items.toLocaleString()}
              </dd>
            </div>
            <div>
              <dt>Sparsity</dt>
              <dd>{formatPercent(modelInfo.sparsity)}</dd>
            </div>
          </dl>

          <p className="muted tiny">
            Event weights:{" "}
            {Object.entries(modelInfo.event_weights)
              .map(([eventType, weight]) => `${eventType}=${weight}`)
              .join(", ")}
          </p>

          {evaluation && (
            <section>
              <h3 className="section-title">Offline evaluation</h3>
              <p className="muted tiny">
                {evaluation.protocol}, {evaluation.evaluated_users} shoppers
              </p>
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th scope="col">Metric</th>
                    {evaluation.hybrid && <th scope="col">Hybrid</th>}
                    <th scope="col">Item-item</th>
                    <th scope="col">Popular</th>
                  </tr>
                </thead>
                <tbody>
                  {/* One row per metric; the best value in each row is highlighted. */}
                  {Object.keys(evaluation.item_item).map((metricKey) => {
                    const hybridValue = evaluation.hybrid?.[metricKey];
                    const modelValue = evaluation.item_item[metricKey];
                    const baselineValue = evaluation.popularity_baseline[metricKey];
                    const bestValue = Math.max(hybridValue ?? -Infinity, modelValue, baselineValue);
                    return (
                      <tr key={metricKey}>
                        <th scope="row">{metricLabel(metricKey)}</th>
                        {hybridValue !== undefined && (
                          <td className={hybridValue === bestValue ? "metric-win" : ""}>
                            {hybridValue.toFixed(3)}
                          </td>
                        )}
                        <td className={modelValue === bestValue ? "metric-win" : ""}>
                          {modelValue.toFixed(3)}
                        </td>
                        <td className={baselineValue === bestValue ? "metric-win" : ""}>
                          {baselineValue.toFixed(3)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </section>
          )}

          <section className="retrain">
            <h3 className="section-title">Feedback loop</h3>
            <p className="small">
              {pendingCount === 0 ? (
                <span className="muted">
                  No new interactions. Use View / Add to cart / Buy on a product to create
                  some.
                </span>
              ) : (
                <>
                  <strong>{pendingCount}</strong> new interaction
                  {pendingCount === 1 ? "" : "s"} recorded since training. Recommendations
                  won’t change until you retrain.
                </>
              )}
            </p>
            {/* Disabled when there is nothing new to learn from or a retrain is already running. */}
            <button
              type="button"
              className="button full-width"
              disabled={pendingCount === 0 || retrainMutation.isPending}
              onClick={() => retrainMutation.mutate()}
            >
              {retrainMutation.isPending ? "Retraining…" : "Retrain model"}
            </button>
            {retrainMutation.isError && (
              <p className="error-text small">{retrainMutation.error.message}</p>
            )}
          </section>
        </>
      )}
    </aside>
  );
}
