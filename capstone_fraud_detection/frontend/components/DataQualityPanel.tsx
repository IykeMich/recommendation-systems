"use client";

import type { UseQueryResult } from "@tanstack/react-query";
import type { ModelInfo, Stats } from "../lib/api";
import { formatDateTime } from "../lib/format";

// Training data-quality checks to display (keys of data_quality.json) and their labels, in display order.
const CHECK_LABELS: Record<string, string> = {
  rows: "Rows",
  duplicate_transaction_ids: "Duplicate IDs",
  missing_amount: "Missing amount",
  non_positive_amount: "Non-positive amount",
  hour_out_of_range: "Hour out of range",
  invalid_label: "Invalid label",
  rows_rejected: "Rows rejected",
  fraud_rate: "Fraud rate",
};

/**
 * Data-quality panel with two views: the checks run on the training data (from /v1/model, fixed
 * until retraining) and live ingestion health (from the polled /v1/stats): latest ingestion
 * time and requests rejected at the API boundary with 422.
 */
export function DataQualityPanel({
  modelInfoQuery,
  statsQuery,
}: {
  modelInfoQuery: UseQueryResult<ModelInfo, Error>;
  statsQuery: UseQueryResult<Stats, Error>;
}) {
  const quality = modelInfoQuery.data?.data_quality;
  const stats = statsQuery.data;
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Data quality</h2>
      </div>
      {/* Training data checks */}
      {quality && (
        <>
          <h3 className="section-title">Training data</h3>
          <dl className="facts">
            {/* Non-zero is a problem for every check except the counts "rows" and "fraud_rate". */}
            {Object.entries(CHECK_LABELS).map(([check, label]) => (
              <div key={check} className={Number(quality[check]) > 0 && !["rows", "fraud_rate"].includes(check) ? "fact-bad" : ""}>
                <dt>{label}</dt>
                <dd>{String(quality[check])}</dd>
              </div>
            ))}
          </dl>
          <p className="muted tiny">
            Missing from the guide’s contract: {(quality.unavailable_contract_fields as string[]).join(", ")}, so there is no temporal split and no per-device/per-merchant novelty.
          </p>
        </>
      )}
      {/* Live ingestion and rejected requests */}
      {stats && (
        <>
          <h3 className="section-title spaced">Live</h3>
          <dl className="facts">
            <div><dt>Latest ingestion</dt><dd>{formatDateTime(stats.latest_ingestion)}</dd></div>
            <div className={stats.rejected_requests > 0 ? "fact-bad" : ""}><dt>Rejected requests</dt><dd>{stats.rejected_requests}</dd></div>
          </dl>
          {/* The server returns the 5 most recent rejections with their field-level errors. */}
          {stats.latest_rejections.length > 0 && (
            <ul className="rejections">
              {stats.latest_rejections.map((rejection) => (
                <li key={rejection.received_at + JSON.stringify(rejection.errors)}>
                  <span className="muted tiny">{formatDateTime(rejection.received_at)}</span>
                  {rejection.errors.map((error) => (
                    <span key={error.field + error.message} className="small"><code>{error.field || "request"}</code> {error.message}</span>
                  ))}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
}
