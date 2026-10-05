"use client";

import type { UseQueryResult } from "@tanstack/react-query";
import type { Decision, DecisionRow } from "../lib/api";
import { formatAmount, formatTime } from "../lib/format";
import { DecisionBadge, PanelError, RiskBar } from "./ui";

// Filter buttons in display order; null means "All". REVIEW comes first because those are the alerts.
const FILTERS: (Decision | null)[] = [null, "REVIEW", "BLOCK", "ALLOW"];

/**
 * "Recent decisions" table with decision filter buttons. The query and filter state live in the
 * page, so this component only renders them. Clicking a row (or pressing Enter on it)
 * selects it, which opens the alert drawer.
 */
export function AlertsTable({
  decisionsQuery,
  decisionFilter,
  onDecisionFilterChange,
  selectedTransactionId,
  onSelectTransaction,
}: {
  decisionsQuery: UseQueryResult<{ decisions: DecisionRow[] }, Error>;
  decisionFilter: Decision | null;
  onDecisionFilterChange: (nextFilter: Decision | null) => void;
  selectedTransactionId: string | null;
  onSelectTransaction: (transactionId: string) => void;
}) {
  const decisions = decisionsQuery.data?.decisions ?? [];
  return (
    <section className="panel alerts">
      <div className="panel-header">
        <h2>Recent decisions</h2>
        <div className="segmented" role="tablist" aria-label="Filter by decision">
          {FILTERS.map((filter) => (
            <button key={filter ?? "all"} type="button" role="tab" aria-selected={decisionFilter === filter}
              onClick={() => onDecisionFilterChange(filter)}>
              {filter ?? "All"}
            </button>
          ))}
        </div>
      </div>
      <PanelError error={decisionsQuery.error} />
      {/* Empty state only after a successful load, so it doesn't flash while loading. */}
      {decisionsQuery.isSuccess && decisions.length === 0 && (
        <p className="empty muted">No decisions yet. Score a transaction or stream the test set.</p>
      )}
      {decisions.length > 0 && (
        <div className="table-scroll">
          <table className="alerts-table">
            <thead>
              <tr>
                <th scope="col">Transaction</th>
                <th scope="col">Customer</th>
                <th scope="col" className="numeric">Amount</th>
                <th scope="col">Risk</th>
                <th scope="col">Decision</th>
                <th scope="col">Top reason</th>
                <th scope="col">Outcome</th>
                <th scope="col">Scored</th>
              </tr>
            </thead>
            <tbody>
              {/* Rows are focusable (tabIndex) so keyboard users can open an alert with Enter. */}
              {decisions.map((row) => (
                <tr key={row.transaction_id} aria-selected={row.transaction_id === selectedTransactionId}
                  onClick={() => onSelectTransaction(row.transaction_id)} tabIndex={0}
                  onKeyDown={(keyEvent) => keyEvent.key === "Enter" && onSelectTransaction(row.transaction_id)}>
                  <td><code>{row.transaction_id}</code>{row.source === "replay" && <span className="source-tag">replay</span>}</td>
                  <td>{row.customer_id}</td>
                  <td className="numeric">{formatAmount(row.amount)}</td>
                  <td><RiskBar riskScore={row.risk_score} /></td>
                  <td><DecisionBadge decision={row.decision} /></td>
                  <td className="reason-cell">{row.reasons[0]}</td>
                  <td>{row.outcome ? <span className={`outcome outcome-${row.outcome}`}>{row.outcome.replace("_", " ")}</span> : <span className="muted">—</span>}</td>
                  <td className="muted">{formatTime(row.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
