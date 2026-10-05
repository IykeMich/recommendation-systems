"use client";

import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import { EVENT_LABELS, formatDateTime } from "../lib/format";

/** Left column: the selected shopper's profile, model status, event counts and recent activity. */
export function ShopperPanel({ userId }: { userId: string }) {
  // Key ["shopper", userId] is invalidated in page.tsx after feedback events and retraining.
  // A 404 means "new shopper", not a failure, so it isn't retried; other errors retry up to twice.
  const shopperQuery = useQuery({
    queryKey: ["shopper", userId],
    queryFn: () => api.shopperDetail(userId),
    retry: (failureCount, error) =>
      !(error instanceof ApiError && error.status === 404) && failureCount < 2,
  });

  const isNewShopper =
    shopperQuery.error instanceof ApiError && shopperQuery.error.status === 404;
  const shopper = shopperQuery.data;

  return (
    <aside className="panel shopper-panel">
      <div className="shopper-heading">
        <span className="avatar" aria-hidden="true">
          {userId.slice(0, 1)}
        </span>
        <div>
          <h2>{userId}</h2>
          {shopper && (
            <span className={`model-tag ${shopper.known_to_model ? "known" : "unknown"}`}>
              {shopper.known_to_model ? "Known to model" : "Not in model yet"}
            </span>
          )}
          {isNewShopper && <span className="model-tag unknown">New shopper</span>}
        </div>
      </div>

      {shopperQuery.isPending && (
        <div className="stack">
          <div className="skeleton-line wide" />
          <div className="skeleton-line" />
          <div className="skeleton-line" />
        </div>
      )}

      {isNewShopper && (
        <p className="muted small">
          This shopper has no history, so the model can’t personalize anything yet and
          the API falls back to popular products. Interact with a few products, then
          retrain the model to see them turn into a known shopper.
        </p>
      )}

      {shopperQuery.isError && !isNewShopper && (
        <p className="error-text small">{shopperQuery.error.message}</p>
      )}

      {shopper?.profile && (
        <dl className="profile-grid">
          <dt>Segment</dt>
          <dd>{shopper.profile.segment}</dd>
          <dt>Age</dt>
          <dd>{shopper.profile.age_band}</dd>
          <dt>Prefers</dt>
          <dd>{shopper.profile.preferred_category}</dd>
          <dt>Device</dt>
          <dd>{shopper.profile.device_type}</dd>
          <dt>Region</dt>
          <dd>{shopper.profile.region}</dd>
        </dl>
      )}

      {shopper && (
        <>
          <div className="event-counts">
            {(["view", "cart", "purchase"] as const).map((eventType) => (
              <div key={eventType} className={`event-count event-${eventType}`}>
                <strong>{shopper.event_type_counts[eventType] ?? 0}</strong>
                <span>{EVENT_LABELS[eventType].toLowerCase()}</span>
              </div>
            ))}
          </div>

          <section>
            <h3 className="section-title">Recent activity</h3>
            <ol className="history-list">
              {shopper.history.map((historyEvent) => (
                <li
                  key={`${historyEvent.item_id}-${historyEvent.event_type}-${historyEvent.timestamp}`}
                >
                  <span className={`event-tag event-${historyEvent.event_type}`}>
                    {EVENT_LABELS[historyEvent.event_type]}
                  </span>
                  <span className="history-name" title={historyEvent.item_id}>
                    {historyEvent.name ?? historyEvent.item_id}
                  </span>
                  {/* Pending events aren't in the model yet, so show a "new" tag instead of a time. */}
                  {historyEvent.pending ? (
                    <span
                      className="pending-tag"
                      title="Recorded after the last training run. Retrain to use it."
                    >
                      new
                    </span>
                  ) : (
                    <time className="muted tiny">
                      {formatDateTime(historyEvent.timestamp)}
                    </time>
                  )}
                </li>
              ))}
            </ol>
          </section>
        </>
      )}
    </aside>
  );
}
