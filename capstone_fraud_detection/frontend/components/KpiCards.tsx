"use client";

import type { UseQueryResult } from "@tanstack/react-query";
import type { Stats } from "../lib/api";
import { formatPercent } from "../lib/format";

/** Row of headline metrics built from the polled /v1/stats query. Shows "—" until data arrives. */
export function KpiCards({ statsQuery }: { statsQuery: UseQueryResult<Stats, Error> }) {
  const stats = statsQuery.data;
  // Labelled precision: of the flagged (REVIEW/BLOCK) replayed transactions that have a synthetic label,
  // the share that really were fraud. null (shown as "—") until at least one such row exists,
  // which avoids dividing by zero.
  const labelledPrecision = stats && stats.flagged_with_label > 0 ? stats.flagged_and_labelled_fraud / stats.flagged_with_label : null;
  // `tone` adds a colour class; `hint` becomes the card's hover tooltip.
  const cards = [
    { label: "Transactions scored", value: stats?.scored.toLocaleString() ?? "—" },
    { label: "Sent to review", value: stats?.review.toLocaleString() ?? "—", tone: "review" },
    { label: "Blocked", value: stats?.block.toLocaleString() ?? "—", tone: "block" },
    { label: "Average risk score", value: stats ? stats.average_risk.toFixed(3) : "—" },
    {
      label: "Flagged that were fraud",
      value: labelledPrecision === null ? "—" : formatPercent(labelledPrecision),
      hint: "Replayed test transactions only: their synthetic labels stand in for outcomes that arrive later.",
    },
    // Highlighted as a warning only when at least one request has been rejected.
    { label: "Rejected requests", value: stats?.rejected_requests.toLocaleString() ?? "—", tone: stats?.rejected_requests ? "warn" : undefined },
  ];
  return (
    <section className="kpis" aria-label="Key metrics">
      {cards.map((card) => (
        <div key={card.label} className={`kpi ${card.tone ? `kpi-${card.tone}` : ""}`} title={card.hint}>
          <span className="kpi-label">{card.label}</span>
          <strong className="kpi-value">{card.value}</strong>
        </div>
      ))}
    </section>
  );
}
