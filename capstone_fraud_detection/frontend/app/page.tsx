"use client";

import { useState } from "react";
import Link from "next/link";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api, API_BASE_URL, type Decision } from "../lib/api";
import { AlertDrawer } from "../components/AlertDrawer";
import { AlertsTable } from "../components/AlertsTable";
import { DataQualityPanel } from "../components/DataQualityPanel";
import { KpiCards } from "../components/KpiCards";
import { ModelPanel } from "../components/ModelPanel";
import { PolicyPanel } from "../components/PolicyPanel";
import { ReplayPanel } from "../components/ReplayPanel";
import { ScorePanel } from "../components/ScorePanel";

// How often live data (KPIs and the decisions table) is polled.
const REFRESH_INTERVAL_MS = 10_000;

/**
 * The single dashboard page. It owns the shared state (decision filter, selected transaction)
 * and the queries used by more than one panel, then lays out the panels and the alert drawer.
 */
export default function FraudOpsDashboard() {
  // null filter = show all decisions; null selection = alert drawer closed.
  const [decisionFilter, setDecisionFilter] = useState<Decision | null>(null);
  const [selectedTransactionId, setSelectedTransactionId] = useState<string | null>(null);

  // ["health"]: polled every 15 s to drive the online/offline badge; no retries, so "offline" shows quickly.
  const healthQuery = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15_000, retry: false });
  // ["stats"]: KPI and live data-quality numbers, polled every 10 s.
  const statsQuery = useQuery({ queryKey: ["stats"], queryFn: api.stats, refetchInterval: REFRESH_INTERVAL_MS });
  // ["decisions", filter]: one cache entry per filter. keepPreviousData keeps the old rows on screen
  // while a newly chosen filter loads, so the table doesn't flash empty.
  const decisionsQuery = useQuery({
    queryKey: ["decisions", decisionFilter],
    queryFn: () => api.decisions(decisionFilter),
    refetchInterval: REFRESH_INTERVAL_MS,
    placeholderData: keepPreviousData,
  });
  // ["model"]: training artifacts only change on retraining, so fetch once and never treat as stale.
  const modelInfoQuery = useQuery({ queryKey: ["model"], queryFn: api.modelInfo, staleTime: Infinity });

  const connectionState = healthQuery.isPending ? "checking" : healthQuery.isSuccess ? "online" : "offline";

  return (
    <div className="page">
      {/* Header: active model/policy versions and API connection status */}
      <header className="site-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">!</span>
          <div>
            <h1>Fraud Ops</h1>
            <p className="muted small">Risk scoring → policy → decision</p>
          </div>
        </div>
        <div className="header-meta">
          <Link href="/about" className="header-link">
            <span className="header-link-icon" aria-hidden="true">i</span>
            Project details
          </Link>
          {healthQuery.data && (
            <span className="muted small"><code>{healthQuery.data.model_version}</code> · <code>{healthQuery.data.policy_version}</code></span>
          )}
          <span className={`api-status api-${connectionState}`} title={API_BASE_URL}>
            <span className="status-dot" aria-hidden="true" />
            {connectionState === "online" ? "API online" : connectionState === "offline" ? "API offline" : "Checking API…"}
          </span>
        </div>
      </header>

      <p className="disclaimer" role="note">
        Synthetic transactions with a learning-only fraud label. Not a live financial control system.
      </p>

      <KpiCards statsQuery={statsQuery} />

      <div className="layout">
        {/* Main column: decisions table, then model and data-quality panels */}
        <div className="main-column">
          <AlertsTable
            decisionsQuery={decisionsQuery}
            decisionFilter={decisionFilter}
            onDecisionFilterChange={setDecisionFilter}
            selectedTransactionId={selectedTransactionId}
            onSelectTransaction={setSelectedTransactionId}
          />
          <div className="two-up">
            <ModelPanel modelInfoQuery={modelInfoQuery} />
            <DataQualityPanel modelInfoQuery={modelInfoQuery} statsQuery={statsQuery} />
          </div>
        </div>
        {/* Side column: actions (score one, stream the test set, change the policy) */}
        <div className="side-column">
          {/* Scoring a transaction opens its detail in the drawer straight away. */}
          <ScorePanel onScored={setSelectedTransactionId} />
          <ReplayPanel />
          <PolicyPanel />
        </div>
      </div>

      {/* The drawer is mounted only while a transaction is selected. */}
      {selectedTransactionId && (
        <AlertDrawer transactionId={selectedTransactionId} onClose={() => setSelectedTransactionId(null)} />
      )}
    </div>
  );
}
