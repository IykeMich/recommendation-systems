"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, API_BASE_URL, ApiError, type ModelInfo, type ProfileOverrides, type RecommendedProduct } from "./lib/api";
import { useDebouncedValue } from "./lib/useDebouncedValue";
import { ModelPanel } from "./components/ModelPanel";
import { ProfilePanel } from "./components/ProfilePanel";
import { RecommendationSection } from "./components/RecommendationSection";
import { UserPicker } from "./components/UserPicker";

/** Top-K requested from the API (matches the backend's TOP_K). */
const RESULT_LIMIT = 5;

/** Header badge that polls /health every 15s and shows checking / online / offline. */
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
 * Main page: picks a user, holds the what-if overrides, and wires the queries and mutations
 * (recommendations, user detail, model info, feedback, retrain) to the three panels.
 */
export default function Home() {
  const queryClient = useQueryClient();
  const [selectedUserId, setSelectedUserId] = useState("F00278");
  const [profileOverrides, setProfileOverrides] = useState<ProfileOverrides>({});
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  // Debounce overrides so typing an income doesn't fire one request per keystroke.
  const debouncedOverrides = useDebouncedValue(profileOverrides, 300);

  /** Switch user and clear any what-if overrides from the previous user. */
  function selectUser(userId: string) {
    setSelectedUserId(userId);
    setProfileOverrides({});
  }

  const userDetailQuery = useQuery({
    queryKey: ["user", selectedUserId],
    queryFn: () => api.userDetail(selectedUserId),
    // A 404 means an unknown visitor (expected), so don't retry it; retry other errors twice.
    retry: (failureCount, error) => !(error instanceof ApiError && error.status === 404) && failureCount < 2,
  });

  // Every override is part of the query key, so each what-if profile is its own cached request.
  const recommendationsQuery = useQuery({
    queryKey: ["recommendations", selectedUserId, debouncedOverrides, RESULT_LIMIT],
    queryFn: () => api.recommendations(selectedUserId, debouncedOverrides, RESULT_LIMIT),
    // Keep showing the previous result while a new what-if request loads (no flash to skeleton).
    placeholderData: keepPreviousData,
  });

  const modelInfoQuery = useQuery({ queryKey: ["model"], queryFn: api.modelInfo });

  // Feedback: refetch this user's history and recommendations, and update the pending count in
  // the cached model info directly instead of refetching /v1/model.
  const recordEventMutation = useMutation({
    mutationFn: api.recordEvent,
    onSuccess: (eventResponse, feedbackEvent) => {
      queryClient.invalidateQueries({ queryKey: ["user", feedbackEvent.user_id] });
      queryClient.invalidateQueries({ queryKey: ["recommendations", feedbackEvent.user_id] });
      queryClient.setQueryData<ModelInfo>(["model"], (previousModelInfo) =>
        previousModelInfo && { ...previousModelInfo, pending_new_interactions: eventResponse.pending_new_interactions },
      );
    },
    onError: (requestError) => setToastMessage(requestError.message),
  });

  // Retrain: store the returned model info and invalidate every user's cached data.
  const retrainMutation = useMutation({
    mutationFn: api.retrainModel,
    onSuccess: (retrainedModelInfo) => {
      queryClient.setQueryData(["model"], retrainedModelInfo);
      queryClient.invalidateQueries({ queryKey: ["recommendations"] });
      queryClient.invalidateQueries({ queryKey: ["user"] });
      setToastMessage("Model retrained with the new feedback.");
    },
  });

  // Auto-hide the toast after 3.5s; the cleanup cancels the timer if the message changes.
  useEffect(() => {
    if (!toastMessage) return;
    const timeoutId = setTimeout(() => setToastMessage(null), 3500);
    return () => clearTimeout(timeoutId);
  }, [toastMessage]);

  /** Send a learn/apply event tagged with the request ID and strategy that produced the card. */
  function recordProductAction(product: RecommendedProduct, action: "learn" | "apply") {
    recordEventMutation.mutate({
      user_id: selectedUserId,
      product_id: product.product_id,
      event_type: action,
      request_id: recommendationsQuery.data?.request_id,
      recommendation_strategy: recommendationsQuery.data?.strategy,
    });
    setToastMessage(
      action === "apply"
        ? `Application for “${product.name}” recorded as feedback. This demo doesn’t approve anything.`
        : `Recorded that ${selectedUserId} read about “${product.name}”.`,
    );
  }

  return (
    <div className="page">
      <header className="site-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">F</span>
          <div>
            <h1>Financial Product Recommender</h1>
            <p className="muted small">Eligibility first, ranking second, every reason traceable</p>
          </div>
        </div>
        <div className="header-actions">
          <Link href="/about" className="header-link">
            Project details
          </Link>
          <ApiStatusIndicator />
        </div>
      </header>

      <p className="disclaimer" role="note">
        Learning project on synthetic data. Not financial advice, a credit decision or real bank eligibility policy.
      </p>

      <UserPicker selectedUserId={selectedUserId} onSelectUser={selectUser} />

      <div className="layout">
        <ProfilePanel
          userId={selectedUserId}
          userDetailQuery={userDetailQuery}
          // Only pass the server's merged profile if it belongs to the selected user (not stale data).
          effectiveProfile={recommendationsQuery.data?.user_id === selectedUserId ? recommendationsQuery.data.profile : null}
          profileOverrides={profileOverrides}
          onProfileOverridesChange={setProfileOverrides}
        />
        <RecommendationSection recommendationsQuery={recommendationsQuery} onProductAction={recordProductAction} />
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
