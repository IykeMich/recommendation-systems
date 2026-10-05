"use client";

import type { UseQueryResult } from "@tanstack/react-query";
import { ApiError, type UserDetail } from "../lib/api";
import { EVENT_LABELS, formatDateTime } from "../lib/format";

/**
 * Side panel for the selected shopper: profile, event counts and recent offer history.
 * A 404 means an unknown visitor, which is explained rather than shown as an error.
 */
export function UserPanel({
  userId,
  userDetailQuery,
}: {
  userId: string;
  userDetailQuery: UseQueryResult<UserDetail, Error>;
}) {
  const isUnknownUser =
    userDetailQuery.error instanceof ApiError && userDetailQuery.error.status === 404;
  const userDetail = userDetailQuery.data;

  return (
    <aside className="panel user-panel">
      <div className="user-heading">
        <span className="avatar" aria-hidden="true">
          {userId.slice(0, 1)}
        </span>
        <div>
          <h2>{userId}</h2>
          {userDetail?.profile && (
            <span className="muted small">
              {userDetail.profile.segment} · {userDetail.profile.age_band}
            </span>
          )}
          {isUnknownUser && <span className="tag tag-warning">Unknown visitor</span>}
        </div>
      </div>

      {userDetailQuery.isPending && (
        <div className="stack">
          <div className="skeleton-line wide" />
          <div className="skeleton-line" />
        </div>
      )}

      {isUnknownUser && (
        <p className="muted small">
          No profile or offer history, so the ranker has nothing personal to go on. The API
          falls back to the most popular offers that pass the rules for this context.
        </p>
      )}

      {userDetailQuery.isError && !isUnknownUser && (
        <p className="error-text small">{userDetailQuery.error.message}</p>
      )}

      {userDetail?.profile && (
        <dl className="profile-grid">
          <dt>Prefers</dt>
          <dd>
            <strong>{userDetail.profile.preferred_category}</strong>
          </dd>
          <dt>Usual device</dt>
          <dd>{userDetail.profile.device_type}</dd>
          <dt>Home region</dt>
          <dd>{userDetail.profile.region}</dd>
        </dl>
      )}

      {userDetail && (
        <>
          <div className="event-counts">
            {(["impression", "click", "redeem"] as const).map((eventType) => (
              <div key={eventType} className={`event-count event-${eventType}`}>
                <strong>{userDetail.event_type_counts[eventType] ?? 0}</strong>
                <span>{EVENT_LABELS[eventType].toLowerCase()}</span>
              </div>
            ))}
          </div>

          <section>
            <h3 className="section-title">Offer history</h3>
            <ol className="history-list">
              {userDetail.history.map((historyEvent) => (
                <li key={`${historyEvent.offer_id}-${historyEvent.event_type}-${historyEvent.timestamp}`}>
                  <span className={`event-tag event-${historyEvent.event_type}`}>
                    {EVENT_LABELS[historyEvent.event_type]}
                  </span>
                  <span className="history-name" title={`${historyEvent.offer_id} · ${historyEvent.category}`}>
                    {historyEvent.title ?? historyEvent.offer_id}
                  </span>
                  {/* Events added since training are flagged "new" instead of showing a time. */}
                  {historyEvent.after_training ? (
                    <span className="tag tag-warning" title="Recorded after the model was trained">
                      new
                    </span>
                  ) : (
                    <time className="muted tiny">{formatDateTime(historyEvent.timestamp)}</time>
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
