"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  API_BASE_URL,
  ApiError,
  type ModelInfo,
  type RecommendationContext,
  type RecommendationRules,
  type RecommendedOffer,
} from "./lib/api";
import { DAY_NAMES, formatHour } from "./lib/format";
import { useDebouncedValue } from "./lib/useDebouncedValue";
import { ContextControls } from "./components/ContextControls";
import { ModelPanel } from "./components/ModelPanel";
import { OfferGrid, type RankMovement } from "./components/OfferGrid";
import { UserPanel } from "./components/UserPanel";
import { UserPicker } from "./components/UserPicker";

// How many offers to request and show.
const RESULT_LIMIT = 10;

/** Snapshot taken by "Pin this context": whose list it was, its context, and each offer's rank. */
type PinnedContext = {
  userId: string;
  context: RecommendationContext;
  rankByOfferId: Map<string, number>;
};

/** One-line summary of a context, e.g. "Tue 09:00 · tablet · Abuja". */
function describeContext(context: RecommendationContext): string {
  return `${DAY_NAMES[context.day_of_week]} ${formatHour(context.hour)} · ${context.device_type} · ${context.region}`;
}

/** Header badge that polls /health every 15 s and shows checking / online / offline. */
function ApiStatusIndicator() {
  const healthQuery = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15_000, retry: false });
  const connectionState = healthQuery.isPending ? "checking" : healthQuery.isSuccess ? "online" : "offline";
  const statusLabels = { checking: "Checking API…", online: "API online", offline: "API offline" };
  return (
    <span className={`api-status api-${connectionState}`} title={API_BASE_URL}>
      <span className="status-dot" aria-hidden="true" />
      {statusLabels[connectionState]}
    </span>
  );
}

/**
 * The single page: pick a shopper, set the context and rules, and see ranked offers. Holds the
 * shared state (user, context, rules, pinned comparison) and the queries and mutations.
 */
export default function Home() {
  const queryClient = useQueryClient();
  const [selectedUserId, setSelectedUserId] = useState("R00001");
  const [context, setContext] = useState<RecommendationContext>({
    device_type: "tablet",
    region: "Abuja",
    hour: 9,
    day_of_week: 1,
  });
  const [rules, setRules] = useState<RecommendationRules>({
    enforce_region: true,
    enforce_channel: true,
    max_per_category: null,
  });
  const [pinnedContext, setPinnedContext] = useState<PinnedContext | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  // Dragging the hour slider changes context rapidly; only fetch once it settles for 150 ms.
  const debouncedContext = useDebouncedValue(context, 150);

  // Profile + history for the selected shopper. A 404 (unknown visitor) is expected, so no retry.
  const userDetailQuery = useQuery({
    queryKey: ["user", selectedUserId],
    queryFn: () => api.userDetail(selectedUserId),
    retry: (failureCount, error) => !(error instanceof ApiError && error.status === 404) && failureCount < 2,
  });

  // When a different shopper is picked, start from their usual device and region.
  const contextSyncedForUserId = useRef<string | null>(null);
  useEffect(() => {
    const profile = userDetailQuery.data?.profile;
    if (!profile || profile.user_id !== selectedUserId) return;
    if (contextSyncedForUserId.current === selectedUserId) return;
    contextSyncedForUserId.current = selectedUserId;
    setContext((previousContext) => ({ ...previousContext, device_type: profile.device_type, region: profile.region }));
  }, [userDetailQuery.data, selectedUserId]);

  // Every value that changes the request is part of the query key (guide §34).
  const offersQueryKey = ["offers", selectedUserId, debouncedContext, rules, RESULT_LIMIT];
  const offersQuery = useQuery({
    queryKey: offersQueryKey,
    queryFn: () => api.offerRecommendations(selectedUserId, debouncedContext, rules, RESULT_LIMIT),
    placeholderData: keepPreviousData,
  });

  // Model metadata, evaluation and pending-feedback count for the Model panel.
  const modelInfoQuery = useQuery({ queryKey: ["model"], queryFn: api.modelInfo });

  // Click/redeem feedback. On success, refetch this user's history and offers (prefix keys match
  // every context) and patch the cached pending count instead of refetching /v1/model.
  const recordEventMutation = useMutation({
    mutationFn: api.recordEvent,
    onSuccess: (eventResponse, feedbackEvent) => {
      // History features change immediately, so fresh recommendations differ right away.
      queryClient.invalidateQueries({ queryKey: ["user", feedbackEvent.user_id] });
      queryClient.invalidateQueries({ queryKey: ["offers", feedbackEvent.user_id] });
      queryClient.setQueryData<ModelInfo>(["model"], (previousModelInfo) =>
        previousModelInfo && {
          ...previousModelInfo,
          pending_new_interactions: eventResponse.pending_new_interactions,
        },
      );
    },
    onError: (requestError) => setToastMessage(requestError.message),
  });

  // Retrain: store the returned model info directly, then refetch all offers and user details,
  // since new weights can change every list.
  const retrainMutation = useMutation({
    mutationFn: api.retrainModel,
    onSuccess: (retrainedModelInfo) => {
      queryClient.setQueryData(["model"], retrainedModelInfo);
      queryClient.invalidateQueries({ queryKey: ["offers"] });
      queryClient.invalidateQueries({ queryKey: ["user"] });
      setToastMessage("Model retrained with the new feedback.");
    },
  });

  useEffect(() => {
    // Auto-hide the toast after 3.5 s; a newer message restarts the timer.
    if (!toastMessage) return;
    const timeoutId = setTimeout(() => setToastMessage(null), 3500);
    return () => clearTimeout(timeoutId);
  }, [toastMessage]);

  /** Send feedback for an offer in the current (debounced) context, tied to the list's request_id. */
  function recordOfferEvent(offer: RecommendedOffer, eventType: "click" | "redeem") {
    recordEventMutation.mutate({
      user_id: selectedUserId,
      offer_id: offer.offer_id,
      event_type: eventType,
      request_id: offersQuery.data?.request_id,
      context: debouncedContext,
    });
    setToastMessage(
      eventType === "redeem"
        ? `Redeemed “${offer.title}”. It won’t be recommended again.`
        : `Clicked “${offer.title}”. Their category affinity just changed.`,
    );
  }

  /** Remember the current list's ranks so later lists can show what moved. */
  function pinCurrentContext() {
    const currentOffers = offersQuery.data?.recommendations ?? [];
    setPinnedContext({
      userId: selectedUserId,
      context: debouncedContext,
      rankByOfferId: new Map(currentOffers.map((offer, rankIndex) => [offer.offer_id, rankIndex + 1])),
    });
  }

  // Compare only while the same shopper is selected; a pin for another user is ignored.
  const isComparing = pinnedContext !== null && pinnedContext.userId === selectedUserId;
  const currentOfferIds = offersQuery.data?.recommendations.map((offer) => offer.offer_id) ?? [];
  const sharedOfferCount = isComparing
    ? currentOfferIds.filter((offerId) => pinnedContext.rankByOfferId.has(offerId)).length
    : 0;

  /** Badge data for an offer: "new" if absent from the pinned list, else places moved (+ = up). */
  function rankMovementFor(offerId: string, currentRank: number): RankMovement | undefined {
    if (!isComparing) return undefined;
    const pinnedRank = pinnedContext.rankByOfferId.get(offerId);
    return pinnedRank === undefined ? { kind: "new" } : { kind: "moved", places: pinnedRank - currentRank };
  }

  return (
    <div className="page">
      <header className="site-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">%</span>
          <div>
            <h1>ShopSmart Offers</h1>
            <p className="muted small">Context-aware offer recommendations</p>
          </div>
        </div>
        <div className="header-actions">
          <Link href="/about" className="details-link">
            Project details
          </Link>
          <ApiStatusIndicator />
        </div>
      </header>

      <UserPicker selectedUserId={selectedUserId} onSelectUser={setSelectedUserId} />
      <ContextControls context={context} onContextChange={setContext} rules={rules} onRulesChange={setRules} />

      {/* Pin/compare bar: summary of overlap while comparing, otherwise the pin button. */}
      <div className="experiment-bar">
        {isComparing ? (
          <>
            <span>
              Pinned <strong>{describeContext(pinnedContext.context)}</strong>. Now{" "}
              <strong>{describeContext(debouncedContext)}</strong>:{" "}
              <strong>
                {sharedOfferCount}/{currentOfferIds.length}
              </strong>{" "}
              offers shared
            </span>
            <button type="button" className="link-button" onClick={() => setPinnedContext(null)}>
              Unpin
            </button>
          </>
        ) : (
          <>
            <span className="muted">
              Context experiment: pin this list, then change the hour, day, device or region to see what moves.
            </span>
            <button type="button" className="button secondary" onClick={pinCurrentContext} disabled={!offersQuery.data}>
              Pin this context
            </button>
          </>
        )}
      </div>

      <div className="layout">
        <UserPanel userId={selectedUserId} userDetailQuery={userDetailQuery} />
        <OfferGrid offersQuery={offersQuery} rankMovementFor={rankMovementFor} onOfferEvent={recordOfferEvent} />
        <ModelPanel modelInfoQuery={modelInfoQuery} retrainMutation={retrainMutation} />
      </div>

      {toastMessage && (
        <div className="toast" role="status">
          {toastMessage}
        </div>
      )}
    </div>
  );
}
