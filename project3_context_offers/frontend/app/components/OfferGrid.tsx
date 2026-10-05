"use client";

import type { UseQueryResult } from "@tanstack/react-query";
import type { OfferRecommendationResponse, RecommendedOffer } from "../lib/api";
import { CHANNEL_LABELS, formatDate, formatPercent } from "../lib/format";

/** Heading and explanation shown for each strategy the API can return. */
const STRATEGY_DESCRIPTIONS = {
  contextual_ranker: {
    title: "Offers for this shopper, right now",
    explanation: "Eligible offers ranked by a logistic-regression model using the shopper’s history, the offer and the current context.",
  },
  cold_start_popular: {
    title: "Popular offers",
    explanation: "Unknown shopper: eligible offers ranked by how many shoppers engaged with them.",
  },
  no_eligible_offers: {
    title: "No eligible offers",
    explanation: "Every offer was removed by the rules for this context. Try relaxing a rule.",
  },
};

/** How an offer's rank compares with the pinned list: absent there ("new") or moved N places. */
export type RankMovement = { kind: "new" } | { kind: "moved"; places: number };

/**
 * Main results area: strategy heading, candidate funnel, loading/error states and offer cards.
 * Keeps the previous list visible (dimmed via is-refreshing) while a new context loads.
 */
export function OfferGrid({
  offersQuery,
  rankMovementFor,
  onOfferEvent,
}: {
  offersQuery: UseQueryResult<OfferRecommendationResponse, Error>;
  rankMovementFor: (offerId: string, currentRank: number) => RankMovement | undefined;
  onOfferEvent: (offer: RecommendedOffer, eventType: "click" | "redeem") => void;
}) {
  const offerData = offersQuery.data;
  const strategyDescription = offerData ? STRATEGY_DESCRIPTIONS[offerData.strategy] : undefined;

  return (
    <section className="offers" aria-live="polite">
      <div className="offers-header">
        <h2>{strategyDescription?.title ?? "Offers"}</h2>
        {strategyDescription && offerData && (
          <p className="muted small">
            <span className={`strategy-badge strategy-${offerData.strategy}`}>{offerData.strategy}</span>{" "}
            {strategyDescription.explanation}
          </p>
        )}
      </div>

      {offerData && <CandidateFunnel offerData={offerData} />}

      {offersQuery.isPending && (
        <div className="offer-grid" aria-busy="true">
          {Array.from({ length: 6 }, (_, placeholderIndex) => (
            <div key={placeholderIndex} className="offer-card skeleton">
              <div className="skeleton-line short" />
              <div className="skeleton-line wide" />
              <div className="skeleton-line" />
            </div>
          ))}
        </div>
      )}

      {offersQuery.isError && (
        <div className="status status-error" role="alert">
          <strong>Couldn’t load offers</strong>
          <p>{offersQuery.error.message}</p>
        </div>
      )}

      {offerData && (
        <ol className={`offer-grid ${offersQuery.isFetching ? "is-refreshing" : ""}`}>
          {offerData.recommendations.map((offer, rankIndex) => (
            <OfferCard
              key={offer.offer_id}
              offer={offer}
              rank={rankIndex + 1}
              isPopularityScore={offerData.strategy === "cold_start_popular"}
              rankMovement={rankMovementFor(offer.offer_id, rankIndex + 1)}
              onOfferEvent={onOfferEvent}
            />
          ))}
        </ol>
      )}
    </section>
  );
}

/**
 * Bars showing how many offers survived each candidate rule, plus the final Top-K count.
 * Widths are relative to the first step, with an 8% minimum so small counts stay visible.
 */
function CandidateFunnel({ offerData }: { offerData: OfferRecommendationResponse }) {
  const funnelSteps = [
    ...offerData.candidate_funnel,
    { step: "shown (top-K)", remaining: offerData.recommendations.length },
  ];
  const totalOffers = funnelSteps[0]?.remaining || 1;
  return (
    <ol className="funnel" aria-label="Candidate generation">
      {funnelSteps.map((funnelStep, stepIndex) => (
        <li key={funnelStep.step} className={stepIndex === funnelSteps.length - 1 ? "funnel-final" : ""}>
          <span className="funnel-bar" style={{ width: `${Math.max(8, (funnelStep.remaining / totalOffers) * 100)}%` }} />
          <span className="funnel-count">{funnelStep.remaining}</span>
          <span className="funnel-step">{funnelStep.step}</span>
        </li>
      ))}
    </ol>
  );
}

/** One offer: rank, movement badge, terms, score bar, reasons and Click/Redeem buttons. */
function OfferCard({
  offer,
  rank,
  isPopularityScore,
  rankMovement,
  onOfferEvent,
}: {
  offer: RecommendedOffer;
  rank: number;
  isPopularityScore: boolean;
  rankMovement?: RankMovement;
  onOfferEvent: (offer: RecommendedOffer, eventType: "click" | "redeem") => void;
}) {
  return (
    <li className="offer-card">
      <div className="offer-top">
        <span className="rank">#{rank}</span>
        {rankMovement && <RankMovementBadge rankMovement={rankMovement} />}
        <span className="offer-category">{offer.category}</span>
      </div>

      <div className="offer-body">
        <p className="offer-discount">
          {offer.discount_pct}% <span>off</span>
        </p>
        <h3 className="offer-title">{offer.title}</h3>
        <p className="offer-terms muted small">
          {offer.min_spend > 0 ? `Min. spend ${offer.min_spend}` : "No minimum spend"} ·{" "}
          {CHANNEL_LABELS[offer.channel]} · {offer.region === "all" ? "All regions" : offer.region}
        </p>
        <p className="muted tiny">Expires {formatDate(offer.expires_at)}</p>
      </div>

      <div className="score">
        <div
          className="score-track"
          role="meter"
          aria-label={isPopularityScore ? "Relative popularity" : "Predicted engagement"}
          aria-valuemin={0}
          aria-valuemax={1}
          aria-valuenow={offer.score}
        >
          <div className="score-fill" style={{ width: `${Math.max(3, offer.score * 100)}%` }} />
        </div>
        <span className="score-label">
          {isPopularityScore ? "popularity" : "engage"} {formatPercent(offer.score)}
        </span>
      </div>

      {offer.reasons.length > 0 && (
        <ul className="reasons" aria-label="Why this offer">
          {offer.reasons.map((reason) => (
            <li key={reason.label}>{reason.label}</li>
          ))}
        </ul>
      )}

      <div className="offer-actions">
        <button type="button" className="button secondary" onClick={() => onOfferEvent(offer, "click")}>
          Click
        </button>
        <button type="button" className="button" onClick={() => onOfferEvent(offer, "redeem")}>
          Redeem
        </button>
      </div>
    </li>
  );
}

/** "new", or an up/down arrow with places moved; nothing if the rank is unchanged. */
function RankMovementBadge({ rankMovement }: { rankMovement: RankMovement }) {
  if (rankMovement.kind === "new") {
    return <span className="movement movement-new" title="Not in the pinned context’s list">new</span>;
  }
  if (rankMovement.places === 0) return null;
  const movedUp = rankMovement.places > 0;
  return (
    <span className={`movement ${movedUp ? "movement-up" : "movement-down"}`} title="Change vs the pinned context">
      {movedUp ? "▲" : "▼"} {Math.abs(rankMovement.places)}
    </span>
  );
}
