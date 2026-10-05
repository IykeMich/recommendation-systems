"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, ApiError, type ProductRecommendation, type RetailEvent } from "../lib/api";
import { useDebouncedValue } from "../lib/useDebouncedValue";
import { LimitSelect, ScoreBar, SkeletonCards, StatusMessage } from "./ui";

/** Quick-pick IDs; NEW_VISITOR isn't in the dataset, so it demonstrates the cold-start fallback. */
const SAMPLE_SHOPPERS = [
  { userId: "R00001", label: "R00001" },
  { userId: "R00042", label: "R00042" },
  { userId: "R00500", label: "R00500" },
  { userId: "NEW_VISITOR", label: "New visitor" },
];

/** Human-readable label + explanation for each `strategy` value the backend can return. */
const STRATEGY_DESCRIPTIONS = {
  collaborative_item_item_baseline: {
    name: "Collaborative filtering",
    explanation: "Items that shoppers with overlapping history also interacted with.",
  },
  popular_cold_start: {
    name: "Popular (cold start)",
    explanation: "No history for this shopper yet, so we fall back to the most popular items.",
  },
};

/** A sent event plus UI-only fields (product name, timestamp) for the "Events sent" list. */
type LoggedEvent = RetailEvent & { productName: string; loggedAt: Date };

/** Retail tab: pick a shopper, see their profile and personalised products, and send events. */
export function RetailExplorer() {
  const [userIdInput, setUserIdInput] = useState("R00001");
  const [resultLimit, setResultLimit] = useState(10);
  const [loggedEvents, setLoggedEvents] = useState<LoggedEvent[]>([]);
  // Normalise (trim + uppercase, as dataset IDs are like "R00001") and debounce 400ms while typing.
  const userId = useDebouncedValue(userIdInput.trim().toUpperCase(), 400);

  // Shopper profile. Skipped when the ID is empty; a 404 (unknown shopper) is final, so don't retry it.
  const profileQuery = useQuery({
    queryKey: ["retail-user", userId],
    queryFn: () => api.retailUserProfile(userId),
    enabled: userId.length > 0,
    retry: (failureCount, error) =>
      !(error instanceof ApiError && error.status === 404) && failureCount < 2,
  });

  // Recommendations for the shopper; unknown IDs still succeed (the backend returns cold-start items).
  const recommendationsQuery = useQuery({
    queryKey: ["user-recommendations", userId, resultLimit],
    queryFn: () => api.userRecommendations(userId, resultLimit),
    enabled: userId.length > 0,
  });

  // POST an interaction event. UI-only fields are stripped before sending; on success the event is
  // prepended to the local activity list (capped at 8 entries).
  const recordEventMutation = useMutation({
    mutationFn: (loggedEvent: LoggedEvent) => {
      const { productName, loggedAt, ...retailEvent } = loggedEvent;
      return api.recordEvent(retailEvent);
    },
    onSuccess: (_response, loggedEvent) =>
      setLoggedEvents((previousEvents) => [loggedEvent, ...previousEvents].slice(0, 8)),
  });

  // Button handler: "cart" is weighted 3x a "view", a stronger signal of interest.
  function logProductEvent(product: ProductRecommendation, eventType: "view" | "cart") {
    recordEventMutation.mutate({
      user_id: userId,
      item_id: product.item_id,
      event_type: eventType,
      event_value: eventType === "cart" ? 3 : 1,
      project: "retail",
      productName: product.name,
      loggedAt: new Date(),
    });
  }

  // Derived display values: strategy label, max score for the bars, and the unknown-shopper flag.
  const recommendationData = recommendationsQuery.data;
  const strategyDescription = recommendationData
    ? STRATEGY_DESCRIPTIONS[recommendationData.strategy]
    : undefined;
  const topScore = recommendationData?.recommendations[0]?.score ?? 0;
  const isUnknownShopper =
    profileQuery.error instanceof ApiError && profileQuery.error.status === 404;

  return (
    <div className="explorer">
      {/* Sidebar: shopper picker, profile, and log of events sent */}
      <aside className="panel sidebar">
        <label className="field-label" htmlFor="shopper-id">
          Shopper ID
        </label>
        <input
          id="shopper-id"
          className="text-input"
          placeholder="e.g. R00001"
          value={userIdInput}
          onChange={(changeEvent) => setUserIdInput(changeEvent.target.value)}
          autoComplete="off"
          spellCheck={false}
        />
        <div className="quick-picks">
          {SAMPLE_SHOPPERS.map((sampleShopper) => (
            <button
              key={sampleShopper.userId}
              type="button"
              className="pill"
              aria-pressed={userId === sampleShopper.userId}
              onClick={() => setUserIdInput(sampleShopper.userId)}
            >
              {sampleShopper.label}
            </button>
          ))}
        </div>

        <div className="profile">
          {profileQuery.isPending && userId && <div className="skeleton-line wide" />}
          {isUnknownShopper && (
            <p className="muted small">
              <strong>{userId}</strong> isn’t in the dataset. They’ll get cold-start
              recommendations.
            </p>
          )}
          {profileQuery.data && (
            <dl className="profile-grid">
              <dt>Segment</dt>
              <dd>{profileQuery.data.segment}</dd>
              <dt>Age</dt>
              <dd>{profileQuery.data.age_band}</dd>
              <dt>Prefers</dt>
              <dd>{profileQuery.data.preferred_category}</dd>
              <dt>Device</dt>
              <dd>{profileQuery.data.device_type}</dd>
              <dt>Region</dt>
              <dd>{profileQuery.data.region}</dd>
              <dt>History</dt>
              <dd>{profileQuery.data.event_count} events</dd>
            </dl>
          )}
        </div>

        <div className="activity">
          <h4>Events sent</h4>
          {recordEventMutation.isError && (
            <p className="error-text small">{recordEventMutation.error.message}</p>
          )}
          {loggedEvents.length === 0 ? (
            <p className="muted small">
              Use “View” or “Add to cart” on a product to send an event to
              <code> POST /v1/events</code>.
            </p>
          ) : (
            <ul className="activity-list">
              {loggedEvents.map((loggedEvent) => (
                <li key={`${loggedEvent.item_id}-${loggedEvent.loggedAt.getTime()}`}>
                  <span className={`event-tag event-${loggedEvent.event_type}`}>
                    {loggedEvent.event_type === "cart" ? "add to cart" : loggedEvent.event_type}
                  </span>
                  <span className="activity-name">{loggedEvent.productName}</span>
                  <time className="muted small">
                    {loggedEvent.loggedAt.toLocaleTimeString()}
                  </time>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      {/* Recommendation results with View / Add to cart actions */}
      <section className="results">
        <div className="results-header">
          <div>
            <h3>Recommended for {userId || "…"}</h3>
            {strategyDescription && (
              <p className="muted small">
                <span className="badge">{strategyDescription.name}</span>{" "}
                {strategyDescription.explanation}
              </p>
            )}
          </div>
          <LimitSelect resultLimit={resultLimit} onResultLimitChange={setResultLimit} />
        </div>

        {!userId && (
          <StatusMessage tone="empty" title="Enter a shopper ID">
            Or pick one of the sample shoppers.
          </StatusMessage>
        )}

        {userId && recommendationsQuery.isPending && <SkeletonCards cardCount={6} />}

        {recommendationsQuery.isError && (
          <StatusMessage tone="error" title="Couldn't load recommendations">
            {recommendationsQuery.error.message}
          </StatusMessage>
        )}

        {recommendationData && (
          <ol className="card-grid">
            {recommendationData.recommendations.map((product, rankIndex) => (
              <li key={product.item_id} className="card">
                <span className="rank">#{rankIndex + 1}</span>
                <span className="card-title">{product.name}</span>
                <div className="chips">
                  <span className="chip">{product.category}</span>
                  <code className="small muted">{product.item_id}</code>
                </div>
                {/* Cold-start items have no score, so no bar */}
                {product.score !== undefined && (
                  <ScoreBar
                    score={product.score}
                    maxScore={topScore}
                    label={`score ${product.score.toLocaleString()}`}
                  />
                )}
                <div className="card-actions">
                  <button
                    type="button"
                    className="button secondary"
                    onClick={() => logProductEvent(product, "view")}
                  >
                    View
                  </button>
                  <button
                    type="button"
                    className="button"
                    onClick={() => logProductEvent(product, "cart")}
                  >
                    Add to cart
                  </button>
                </div>
              </li>
            ))}
          </ol>
        )}
      </section>
    </div>
  );
}
