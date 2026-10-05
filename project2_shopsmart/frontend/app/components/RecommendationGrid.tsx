"use client";

import { useId } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import type {
  InteractionEventType,
  Recommendation,
  RecommendationReason,
  RecommendationResponse,
} from "../lib/api";
import { formatPrice } from "../lib/format";

// Heading and explanation shown for each strategy the API can return.
const STRATEGY_DESCRIPTIONS = {
  adaptive_hybrid: {
    title: "Picked for this shopper",
    explanation:
      "A blend of products similar to what this shopper engaged with and products picked by shoppers with overlapping histories. A short history leans on similar products; the mix shifts towards “shoppers like you” as the history grows. Products this shopper already interacted with are filtered out.",
  },
  popular_fallback: {
    title: "Popular right now",
    explanation:
      "The model has no history for this shopper (cold start), so it falls back to the most popular products overall.",
  },
};

// Buttons on each card (and in product search); each sends the matching feedback event type.
export const PRODUCT_ACTIONS: { eventType: InteractionEventType; label: string }[] = [
  { eventType: "view", label: "View" },
  { eventType: "cart", label: "Add to cart" },
  { eventType: "purchase", label: "Buy" },
];

// Tooltip heading for the hybrid part a card mainly came from.
const SOURCE_LABELS = {
  similar_products: "Similar product",
  shoppers_like_you: "Shoppers like you",
} as const;

// How the shopper's own action on a product reads in a sentence ("you bought").
const YOUR_EVENT_VERBS = { view: "viewed", cart: "carted", purchase: "bought" } as const;

/** One "Why this?" line, e.g. "37 shoppers who engaged with Doritos (you bought it) also engaged with this". */
function reasonText(reason: RecommendationReason): string {
  const productName = reason.name ?? reason.item_id;
  const yourAction = reason.your_event ? `you ${YOUR_EVENT_VERBS[reason.your_event]} it` : "in your history";
  if (reason.type === "co_interaction") {
    const shoppers = `${reason.shopper_count} shopper${reason.shopper_count === 1 ? "" : "s"}`;
    return `${shoppers} who engaged with ${productName} (${yourAction}) also engaged with this`;
  }
  const sharedDetail =
    reason.shared === "description" ? "a similar description" : `the same ${reason.shared}: ${reason.value}`;
  return `Similar to ${productName} (${yourAction}), with ${sharedDetail}`;
}

/**
 * Small "Why?" button that reveals the card's explanation in a tooltip on hover or focus
 * (tapping focuses it on touch screens), keeping the card itself uncluttered.
 */
function WhyTooltip({ recommendation }: { recommendation: Recommendation }) {
  const tooltipId = useId();
  if (recommendation.source === "popular" || recommendation.reasons.length === 0) return null;
  return (
    <span className="why-tooltip">
      <button type="button" className="why-trigger" aria-describedby={tooltipId}>
        Why?
      </button>
      <span role="tooltip" id={tooltipId} className="why-popup small">
        <span className={`why-source why-source-${recommendation.source}`}>
          {SOURCE_LABELS[recommendation.source]}
        </span>
        {/* Strongest two reasons; the first matches the source. */}
        <ul>
          {recommendation.reasons.slice(0, 2).map((reason) => (
            <li key={`${reason.type}-${reason.item_id}`}>{reasonText(reason)}</li>
          ))}
        </ul>
      </span>
    </span>
  );
}

/** How a card's rank changed since the last retrain: newly appeared, or moved by `places` (+ = up). */
export type RankMovement = { kind: "new" } | { kind: "moved"; places: number };

/**
 * Middle column: strategy explanation, result-limit picker, and the ranked product cards
 * (with loading skeletons and an error state).
 */
export function RecommendationGrid({
  recommendationsQuery,
  rankMovementFor,
  onProductEvent,
  resultLimit,
  onResultLimitChange,
}: {
  recommendationsQuery: UseQueryResult<RecommendationResponse, Error>;
  rankMovementFor: (itemId: string, currentRank: number) => RankMovement | undefined;
  onProductEvent: (recommendation: Recommendation, eventType: InteractionEventType) => void;
  resultLimit: number;
  onResultLimitChange: (nextLimit: number) => void;
}) {
  const recommendationData = recommendationsQuery.data;
  const strategyDescription = recommendationData
    ? STRATEGY_DESCRIPTIONS[recommendationData.strategy]
    : undefined;
  // Results are sorted, so the first score is the maximum; bars are drawn relative to it.
  const topScore = recommendationData?.recommendations[0]?.score ?? 0;

  return (
    <section className="recommendations" aria-live="polite">
      <div className="recommendations-header">
        <div>
          <h2>{strategyDescription?.title ?? "Recommendations"}</h2>
          {strategyDescription && (
            <p className="muted small">
              <span className={`strategy-badge strategy-${recommendationData?.strategy}`}>
                {recommendationData?.strategy}
              </span>{" "}
              {strategyDescription.explanation}
            </p>
          )}
        </div>
        <label className="limit-select">
          Show
          <select
            value={resultLimit}
            onChange={(changeEvent) => onResultLimitChange(Number(changeEvent.target.value))}
          >
            {[6, 12, 24].map((limitOption) => (
              <option key={limitOption} value={limitOption}>
                {limitOption}
              </option>
            ))}
          </select>
        </label>
      </div>

      {recommendationsQuery.isPending && (
        <div className="product-grid" aria-busy="true">
          {Array.from({ length: 6 }, (_, placeholderIndex) => (
            <div key={placeholderIndex} className="product-card skeleton">
              <div className="skeleton-line short" />
              <div className="skeleton-line wide" />
              <div className="skeleton-line" />
            </div>
          ))}
        </div>
      )}

      {recommendationsQuery.isError && (
        <div className="status status-error" role="alert">
          <strong>Couldn’t load recommendations</strong>
          <p>{recommendationsQuery.error.message}</p>
        </div>
      )}

      {/* Ranked list; dimmed (is-refreshing) while a background refetch is in flight. */}
      {recommendationData && (
        <ol
          className={`product-grid ${recommendationsQuery.isFetching ? "is-refreshing" : ""}`}
        >
          {recommendationData.recommendations.map((recommendation, rankIndex) => (
            <ProductCard
              key={recommendation.item_id}
              recommendation={recommendation}
              rank={rankIndex + 1}
              topScore={topScore}
              isFallback={recommendationData.strategy === "popular_fallback"}
              rankMovement={rankMovementFor(recommendation.item_id, rankIndex + 1)}
              onProductEvent={onProductEvent}
            />
          ))}
        </ol>
      )}
    </section>
  );
}

/** One recommended product: rank, movement badge, "Why?" tooltip, details, score bar, actions. */
function ProductCard({
  recommendation,
  rank,
  topScore,
  isFallback,
  rankMovement,
  onProductEvent,
}: {
  recommendation: Recommendation;
  rank: number;
  topScore: number;
  isFallback: boolean;
  rankMovement?: RankMovement;
  onProductEvent: (recommendation: Recommendation, eventType: InteractionEventType) => void;
}) {
  // Bar width relative to the top score, with a 4% minimum so low scores stay visible.
  const scorePercent = topScore > 0 ? Math.max(4, (recommendation.score / topScore) * 100) : 0;

  return (
    <li className="product-card">
      <div className="product-top">
        <span className="rank">#{rank}</span>
        {rankMovement && <RankMovementBadge rankMovement={rankMovement} />}
        <span className="product-category">{recommendation.category}</span>
        <WhyTooltip recommendation={recommendation} />
      </div>

      <div>
        <p className="product-brand">
          {recommendation.brand} · {recommendation.subcategory}
        </p>
        <h3 className="product-name">{recommendation.name ?? recommendation.item_id}</h3>
        <p className="product-price">{formatPrice(recommendation.price)}</p>
      </div>

      <div className="score">
        <div
          className="score-track"
          role="meter"
          aria-label={isFallback ? "Popularity" : "Recommendation score"}
          aria-valuemin={0}
          aria-valuemax={topScore}
          aria-valuenow={recommendation.score}
        >
          <div className="score-fill" style={{ width: `${scorePercent}%` }} />
        </div>
        <span className="score-label">
          {isFallback ? "popularity" : "score"} {recommendation.score.toFixed(2)}
        </span>
      </div>

      <div className="product-actions">
        {PRODUCT_ACTIONS.map((productAction) => (
          <button
            key={productAction.eventType}
            type="button"
            className={`button ${productAction.eventType === "purchase" ? "" : "secondary"}`}
            onClick={() => onProductEvent(recommendation, productAction.eventType)}
          >
            {productAction.label}
          </button>
        ))}
      </div>
      <code className="item-id tiny muted">{recommendation.item_id}</code>
    </li>
  );
}

/** Small "new" / ▲n / ▼n badge; renders nothing when the rank didn't change. */
function RankMovementBadge({ rankMovement }: { rankMovement: RankMovement }) {
  if (rankMovement.kind === "new") {
    return <span className="movement movement-new">new</span>;
  }
  if (rankMovement.places === 0) return null;
  const movedUp = rankMovement.places > 0;
  return (
    <span
      className={`movement ${movedUp ? "movement-up" : "movement-down"}`}
      title="Change since the last retrain"
    >
      {movedUp ? "▲" : "▼"} {Math.abs(rankMovement.places)}
    </span>
  );
}
