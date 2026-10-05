"use client";

import type { UseQueryResult } from "@tanstack/react-query";
import type { ModelInfo } from "../lib/api";
import { formatDateTime, formatPercent, MODEL_LABELS } from "../lib/format";
import { PanelError } from "./ui";

/**
 * Model panel: what is being served (from the training manifest) and a test-set comparison of all
 * candidate models against the synthetic-rule reference ceiling. Also shows how calibration changes
 * the test decision mix (calibrated scores send suspicious transactions to REVIEW instead of BLOCK).
 */
export function ModelPanel({ modelInfoQuery }: { modelInfoQuery: UseQueryResult<ModelInfo, Error> }) {
  const modelInfo = modelInfoQuery.data;
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Model</h2>
        {modelInfo && <code className="version">{modelInfo.manifest.model_version}</code>}
      </div>
      <PanelError error={modelInfoQuery.error} />
      {modelInfo && (
        <>
          {/* Serving summary */}
          <dl className="facts">
            <dt>Serving</dt><dd>{MODEL_LABELS[modelInfo.metrics.served_model] ?? modelInfo.metrics.served_model}</dd>
            <dt>Trained</dt><dd>{formatDateTime(modelInfo.manifest.trained_at)}</dd>
            <dt>Features</dt><dd>{modelInfo.manifest.feature_schema_version} · {modelInfo.manifest.features.numeric.length + modelInfo.manifest.features.categorical.length} inputs</dd>
            <dt>Split</dt><dd>{modelInfo.metrics.split.method}</dd>
          </dl>
          {/* Model comparison on the test set; the served model and the rule reference are highlighted. */}
          <div className="table-scroll">
            <table className="metrics-table">
              <thead>
                <tr><th scope="col">Model (test set)</th><th scope="col">PR-AUC</th><th scope="col">ROC-AUC</th><th scope="col">Brier</th><th scope="col">Recall</th><th scope="col">Precision</th></tr>
              </thead>
              <tbody>
                {Object.entries(modelInfo.metrics.models).map(([modelName, splits]) => (
                  <tr key={modelName} className={modelName === modelInfo.metrics.served_model ? "served" : modelName === "synthetic_rule_reference" ? "reference" : ""}>
                    <th scope="row">{MODEL_LABELS[modelName] ?? modelName}</th>
                    <td>{splits.test.pr_auc.toFixed(3)}</td>
                    <td>{splits.test.roc_auc.toFixed(3)}</td>
                    <td>{splits.test.brier_score.toFixed(3)}</td>
                    <td>{formatPercent(splits.test.at_review_threshold.recall, 0)}</td>
                    <td>{formatPercent(splits.test.at_review_threshold.precision, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted tiny">
            Recall/precision at the review threshold. The label is a synthetic rule (hours 1–4 with velocity ≥ 6), so strong models sit at the
            rule’s ceiling. Selected by {modelInfo.metrics.selection_rule}.
          </p>
          <p className="small">
            Test decision mix: calibrated {formatPercent(modelInfo.metrics.test_decision_mix.REVIEW)} review /{" "}
            {formatPercent(modelInfo.metrics.test_decision_mix.BLOCK)} block · uncalibrated{" "}
            {formatPercent(modelInfo.metrics.test_decision_mix_uncalibrated.REVIEW)} /{" "}
            {formatPercent(modelInfo.metrics.test_decision_mix_uncalibrated.BLOCK)}
          </p>
        </>
      )}
    </section>
  );
}
