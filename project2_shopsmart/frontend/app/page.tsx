"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  API_BASE_URL,
  type InteractionEventType,
  type Product,
  type RecommendationResponse,
} from "./lib/api";
import { EVENT_LABELS } from "./lib/format";
import { ModelPanel } from "./components/ModelPanel";
import { ProductSearch } from "./components/ProductSearch";
import { RecommendationGrid, type RankMovement } from "./components/RecommendationGrid";
import { ShopperPanel } from "./components/ShopperPanel";
import { ShopperPicker } from "./components/ShopperPicker";

/** Each item's rank just before a retrain, for one user, so cards can show ▲/▼/new afterwards. */
type RankingSnapshot = { userId: string; rankByItemId: Map<string, number> };

/** Header badge that polls GET /health every 15s and shows checking / online / offline. */
function ApiStatusIndicator() {
  // retry: false so an outage shows as "offline" straight away instead of after several retries.
  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15_000,
    retry: false,
  });
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

/**
 * The storefront page. Owns the selected shopper, result limit, rank snapshot and toast, and wires
 * the queries and mutations that the panels display.
 */
export default function Home() {
  const queryClient = useQueryClient();
  const [selectedUserId, setSelectedUserId] = useState("R00001");
  const [resultLimit, setResultLimit] = useState(12);
  const [rankingSnapshot, setRankingSnapshot] = useState<RankingSnapshot | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // The key includes user and limit, so changing either fetches (and caches) a separate list.
  const recommendationsQueryKey = ["recommendations", selectedUserId, resultLimit];
  const recommendationsQuery = useQuery({
    queryKey: recommendationsQueryKey,
    queryFn: () => api.recommendations(selectedUserId, resultLimit),
  });

  const modelInfoQuery = useQuery({ queryKey: ["model"], queryFn: api.modelInfo });

  // Record a view/cart/purchase. Recommendations are deliberately NOT refetched: the model only
  // changes on retrain. The shopper's history is refetched (to show the pending event), and the
  // cached model info gets the new pending count without another request.
  const recordEventMutation = useMutation({
    mutationFn: api.recordEvent,
    onSuccess: (feedbackResponse, feedbackEvent) => {
      queryClient.invalidateQueries({ queryKey: ["shopper", feedbackEvent.user_id] });
      queryClient.setQueryData(["model"], (previousModelInfo: typeof modelInfoQuery.data) =>
        previousModelInfo && {
          ...previousModelInfo,
          pending_new_interactions: feedbackResponse.pending_new_interactions,
        },
      );
    },
    onError: (requestError) => setToastMessage(requestError.message),
  });

  const retrainMutation = useMutation({
    mutationFn: api.retrainModel,
    onMutate: () => {
      // Remember the current ranking so cards can show how they moved.
      const currentRecommendations =
        queryClient.getQueryData<RecommendationResponse>(recommendationsQueryKey);
      if (currentRecommendations) {
        setRankingSnapshot({
          userId: selectedUserId,
          rankByItemId: new Map(
            currentRecommendations.recommendations.map((recommendation, rankIndex) => [
              recommendation.item_id,
              rankIndex + 1,
            ]),
          ),
        });
      }
    },
    // Use the returned model info directly, then mark every user's recommendations and shopper
    // details stale so they refetch from the new model.
    onSuccess: (retrainedModelInfo) => {
      queryClient.setQueryData(["model"], retrainedModelInfo);
      queryClient.invalidateQueries({ queryKey: ["recommendations"] });
      queryClient.invalidateQueries({ queryKey: ["shopper"] });
      setToastMessage("Model retrained with the new interactions.");
    },
  });

  // Auto-hide the toast after 3.5s; a new message restarts the timer.
  useEffect(() => {
    if (!toastMessage) return;
    const timeoutId = setTimeout(() => setToastMessage(null), 3500);
    return () => clearTimeout(timeoutId);
  }, [toastMessage]);

  /**
   * Send a product action. Actions on recommended cards carry the request id of the list they
   * came from; actions from product search don't, so the feedback log keeps the two apart.
   */
  function recordProductEvent(
    product: Product,
    eventType: InteractionEventType,
    recommendationRequestId?: string,
  ) {
    recordEventMutation.mutate({
      user_id: selectedUserId,
      item_id: product.item_id,
      event_type: eventType,
      recommendation_request_id: recommendationRequestId,
    });
    setToastMessage(
      `${EVENT_LABELS[eventType]} “${product.name ?? product.item_id}”. It will count after the next retrain.`,
    );
  }

  /**
   * Compare an item's current rank with the pre-retrain snapshot. Only applies to the user the
   * snapshot was taken for; items absent from the snapshot are "new". Positive places = moved up.
   */
  function rankMovementFor(itemId: string, currentRank: number): RankMovement | undefined {
    if (!rankingSnapshot || rankingSnapshot.userId !== selectedUserId) return undefined;
    const previousRank = rankingSnapshot.rankByItemId.get(itemId);
    if (previousRank === undefined) return { kind: "new" };
    return { kind: "moved", places: previousRank - currentRank };
  }

  return (
    <div className="page">
      <header className="site-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            S
          </span>
          <div>
            <h1>ShopSmart</h1>
            <p className="muted small">Adaptive hybrid product recommendations</p>
          </div>
        </div>
        <div className="header-actions">
          <Link href="/about" className="details-link">
            Project details →
          </Link>
          <ApiStatusIndicator />
        </div>
      </header>

      <ShopperPicker selectedUserId={selectedUserId} onSelectUser={setSelectedUserId} />
      <ProductSearch selectedUserId={selectedUserId} onProductEvent={recordProductEvent} />

      {/* Three columns: shopper profile/history, recommendations, model info + retrain. */}
      <div className="layout">
        <ShopperPanel userId={selectedUserId} />
        <RecommendationGrid
          recommendationsQuery={recommendationsQuery}
          rankMovementFor={rankMovementFor}
          onProductEvent={(recommendation, eventType) =>
            recordProductEvent(
              recommendation,
              eventType,
              recommendationsQuery.data?.recommendation_request_id,
            )
          }
          resultLimit={resultLimit}
          onResultLimitChange={setResultLimit}
        />
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
