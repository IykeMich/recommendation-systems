"use client";

import { useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api, API_BASE_URL } from "./lib/api";
import { MovieExplorer } from "./components/MovieExplorer";
import { RetailExplorer } from "./components/RetailExplorer";

/** The two playground tabs; `as const` lets TypeScript derive the literal tab ID union below. */
const PROJECT_TABS = [
  { tabId: "movies", label: "Movies", subtitle: "Content-based" },
  { tabId: "retail", label: "Retail", subtitle: "Personalized" },
] as const;

/** "movies" | "retail", derived from PROJECT_TABS so the two never drift apart. */
type ProjectTabId = (typeof PROJECT_TABS)[number]["tabId"];

/** Header badge showing whether the FastAPI backend is reachable (hover shows the API URL). */
function ApiStatusIndicator() {
  // Poll GET /health every 15s; no retries so an outage shows as "offline" straight away.
  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15_000,
    retry: false,
  });

  // Collapse the query state into one of three display states.
  const connectionState = healthQuery.isPending
    ? "checking"
    : healthQuery.isSuccess
      ? "online"
      : "offline";

  const statusLabels = {
    checking: "Checking API…",
    online: "API online",
    offline: "API offline",
  };

  return (
    <span className={`api-status api-${connectionState}`} title={API_BASE_URL}>
      <span className="status-dot" aria-hidden="true" />
      {statusLabels[connectionState]}
    </span>
  );
}

/** Home page: header, tab bar and the active project's explorer. */
export default function Home() {
  const [activeTabId, setActiveTabId] = useState<ProjectTabId>("movies");

  return (
    <div className="page">
      <header className="site-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            R
          </span>
          <div>
            <h1>RecLab</h1>
            <p className="muted small">Recommendation playground</p>
          </div>
        </div>
        <div className="header-actions">
          <Link href="/about" className="header-link">
            Project details
          </Link>
          <ApiStatusIndicator />
        </div>
      </header>

      {/* Tab bar: ARIA tab pattern linking each tab to the panel below */}
      <nav className="tabs" role="tablist" aria-label="Recommendation projects">
        {PROJECT_TABS.map((projectTab) => (
          <button
            key={projectTab.tabId}
            type="button"
            role="tab"
            id={`tab-${projectTab.tabId}`}
            aria-selected={activeTabId === projectTab.tabId}
            aria-controls={`panel-${projectTab.tabId}`}
            className="tab"
            onClick={() => setActiveTabId(projectTab.tabId)}
          >
            {projectTab.label}
            <span className="tab-subtitle">{projectTab.subtitle}</span>
          </button>
        ))}
      </nav>

      <main
        role="tabpanel"
        id={`panel-${activeTabId}`}
        aria-labelledby={`tab-${activeTabId}`}
      >
        {/* Only the active explorer is mounted; its query results stay in the shared cache */}
        {activeTabId === "movies" ? <MovieExplorer /> : <RetailExplorer />}
      </main>
    </div>
  );
}
