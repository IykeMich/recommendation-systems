"use client";

import { useState } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import type { Evidence, Reason, RecommendationResponse, RecommendedProduct } from "../lib/api";
import { formatMoney, RULE_LABELS, titleCase } from "../lib/format";

/** Heading and one-line explanation for each serving strategy the API can return. */
const STRATEGY_DESCRIPTIONS = {
  eligibility_then_ranking: {
    title: "Recommended products",
    explanation: "Rules removed ineligible products first; the model then ranked the rest.",
  },
  cold_start_popular: {
    title: "Popular products for a new visitor",
    explanation: "No profile: only products that pass the rules with missing data, ranked by popularity.",
  },
  no_eligible_products: {
    title: "No eligible products",
    explanation: "The rules removed every product. Nothing is shown rather than showing something ineligible.",
  },
};

type ProductAction = "learn" | "apply";

/** Evidence object as "field: value" lines, shown as a tooltip on each reason. */
function describeEvidence(evidence: Evidence): string {
  return Object.entries(evidence)
    .map(([field, value]) => `${field}: ${Array.isArray(value) ? value.join(", ") : value}`)
    .join("\n");
}

/**
 * Centre column: strategy summary, eligibility meter, ranked product cards, the products the
 * rules excluded, and a toggleable "no eligibility layer" comparison.
 */
export function RecommendationSection({
  recommendationsQuery,
  onProductAction,
}: {
  recommendationsQuery: UseQueryResult<RecommendationResponse, Error>;
  onProductAction: (product: RecommendedProduct, action: ProductAction) => void;
}) {
  const [showComparison, setShowComparison] = useState(false);
  const result = recommendationsQuery.data;
  const strategyDescription = result ? STRATEGY_DESCRIPTIONS[result.strategy] : undefined;
  // Score bars are scaled relative to the best shown score, so the top card fills the bar.
  const topScore = Math.max(...(result?.recommendations.map((product) => product.score) ?? [0]), 0);

  return (
    <section className="recommendations" aria-live="polite">
      <div className="recommendations-header">
        <h2>{strategyDescription?.title ?? "Recommendations"}</h2>
        {strategyDescription && result && (
          <p className="muted small">
            <span className={`strategy-badge strategy-${result.strategy}`}>{result.strategy}</span>{" "}
            {strategyDescription.explanation}
          </p>
        )}
      </div>

      {/* Eligibility meter: share of the catalog that passed the rules. */}
      {result && (
        <div className="eligibility-bar" role="status">
          <div className="eligibility-meter" aria-hidden="true">
            <span style={{ width: `${(result.eligibility.eligible_products / result.eligibility.total_products) * 100}%` }} />
          </div>
          <span>
            <strong>{result.eligibility.eligible_products}</strong> of {result.eligibility.total_products} products eligible
          </span>
          <span className="muted tiny">rules {result.eligibility.rules_version}</span>
        </div>
      )}

      {/* Skeleton cards only on the very first load (later loads keep previous data). */}
      {recommendationsQuery.isPending && (
        <div className="product-list" aria-busy="true">
          {Array.from({ length: 4 }, (_, placeholderIndex) => (
            <div key={placeholderIndex} className="product-card skeleton">
              <div className="skeleton-line short" />
              <div className="skeleton-line wide" />
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

      {result && (
        <ol className={`product-list ${recommendationsQuery.isFetching ? "is-refreshing" : ""}`}>
          {result.recommendations.map((product, rankIndex) => (
            <ProductCard
              key={product.product_id}
              product={product}
              rank={rankIndex + 1}
              topScore={topScore}
              isPopularityScore={result.strategy === "cold_start_popular"}
              onProductAction={onProductAction}
            />
          ))}
        </ol>
      )}

      {/* Excluded products with the failed rules, which never reached the ranker. */}
      {result && result.eligibility.ineligible.length > 0 && (
        <details className="panel ineligible">
          <summary>
            <strong>{result.eligibility.ineligible.length} products excluded by the rules</strong>
            <span className="muted small"> · never passed to the ranker</span>
          </summary>
          <ul className="ineligible-list">
            {result.eligibility.ineligible.map((product) => (
              <li key={product.product_id}>
                <div>
                  <strong>{product.name}</strong>{" "}
                  <span className="muted small">
                    {product.product_id} · {product.category} · {product.risk_level} risk
                  </span>
                </div>
                {product.failed_rules.map((failedRule) => (
                  <p key={failedRule.rule_id} className="failed-rule small">
                    <span className="rule-tag">{RULE_LABELS[failedRule.rule_id] ?? failedRule.rule_id}</span> {failedRule.detail}
                  </p>
                ))}
              </li>
            ))}
          </ul>
        </details>
      )}

      {/* Teaching comparison: the same model ranking ALL products; ineligible rows are highlighted. */}
      {result?.unfiltered_ranking && (
        <section className="panel comparison">
          <div className="section-row">
            <div>
              <h3>What if there were no eligibility layer?</h3>
              <p className="muted small">The same model ranking every product, including ones the rules exclude.</p>
            </div>
            <button type="button" className="button secondary" onClick={() => setShowComparison((isShown) => !isShown)}>
              {showComparison ? "Hide" : "Show"}
            </button>
          </div>
          {showComparison && (
            <ol className="comparison-list">
              {result.unfiltered_ranking.map((row, rankIndex) => (
                <li key={row.product_id} className={row.eligible ? "" : "violation"}>
                  <span className="rank">#{rankIndex + 1}</span>
                  <span className="comparison-name">{row.name}</span>
                  <span className="muted tiny">{row.score.toFixed(3)}</span>
                  {row.eligible ? (
                    <span className="tag tag-ok">eligible</span>
                  ) : (
                    <span className="tag tag-danger" title={row.failed_rules.map((failedRule) => failedRule.detail).join("\n")}>
                      ineligible · {row.failed_rules.map((failedRule) => RULE_LABELS[failedRule.rule_id]).join(", ")}
                    </span>
                  )}
                </li>
              ))}
            </ol>
          )}
        </section>
      )}
    </section>
  );
}

/** One recommendation: product info, score bar, reasons/cautions and Learn more / Apply actions. */
function ProductCard({
  product,
  rank,
  topScore,
  isPopularityScore,
  onProductAction,
}: {
  product: RecommendedProduct;
  rank: number;
  topScore: number;
  isPopularityScore: boolean;
  onProductAction: (product: RecommendedProduct, action: ProductAction) => void;
}) {
  return (
    <li className="product-card">
      <div className="product-main">
        <span className="rank">#{rank}</span>
        <div className="product-heading">
          <h3>{product.name}</h3>
          <p className="muted small">
            {titleCase(product.category)} · {product.product_id} ·{" "}
            {product.min_income > 0 ? `min. income ${formatMoney(product.min_income)}` : "no minimum income"}
          </p>
        </div>
        <span className={`risk-badge risk-${product.risk_level}`}>{product.risk_level} risk</span>
      </div>

      <div className="score">
        <div className="score-track" role="meter" aria-label={isPopularityScore ? "Relative popularity" : "Model score"}
          aria-valuemin={0} aria-valuemax={topScore} aria-valuenow={product.score}>
          {/* Width relative to topScore, with a 3% minimum so tiny scores stay visible. */}
          <div className="score-fill" style={{ width: `${topScore > 0 ? Math.max(3, (product.score / topScore) * 100) : 0}%` }} />
        </div>
        <span className="score-label">{isPopularityScore ? "popularity" : "relative score"} {product.score.toFixed(3)}</span>
      </div>

      <div className="why">
        <h4>Why this was shown</h4>
        <ul className="reason-list">
          {product.reasons.map((reason) => (
            <ReasonItem key={reason.text} reason={reason} />
          ))}
        </ul>
        {product.cautions.length > 0 && (
          <ul className="reason-list cautions">
            {product.cautions.map((caution) => (
              <ReasonItem key={caution.text} reason={caution} isCaution />
            ))}
          </ul>
        )}
      </div>

      <div className="product-actions">
        <button type="button" className="button secondary" onClick={() => onProductAction(product, "learn")}>
          Learn more
        </button>
        <button type="button" className="button" onClick={() => onProductAction(product, "apply")}>
          Apply
        </button>
      </div>
    </li>
  );
}

/** A reason (or caution) labelled with its evidence source; hover shows the full evidence. */
function ReasonItem({ reason, isCaution = false }: { reason: Reason; isCaution?: boolean }) {
  return (
    <li className={isCaution ? "caution" : ""} title={`Evidence\n${describeEvidence(reason.evidence)}`}>
      <span className={`evidence-source source-${reason.evidence.source}`}>{reason.evidence.source}</span>
      {reason.text}
    </li>
  );
}
