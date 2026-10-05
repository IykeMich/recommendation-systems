"use client";

import { useState, type FormEvent } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api, type InteractionEventType, type Product } from "../lib/api";
import { formatPrice } from "../lib/format";
import { PRODUCT_ACTIONS } from "./RecommendationGrid";

/**
 * "Find a product" panel: search the whole catalog (optionally within one category) and
 * View / Add to cart / Buy any product as the selected shopper. Useful for testing how the model
 * reacts when a shopper interacts across different categories, not just with what it recommended.
 * The search runs when the Search button (or Enter) is pressed; an empty search lists all products.
 */
export function ProductSearch({
  selectedUserId,
  onProductEvent,
}: {
  selectedUserId: string;
  onProductEvent: (product: Product, eventType: InteractionEventType) => void;
}) {
  const [searchInput, setSearchInput] = useState("");
  const [category, setCategory] = useState("");
  // The search that was last submitted; null until the first Search, so nothing loads up front.
  const [submittedSearch, setSubmittedSearch] = useState<{ term: string; category: string } | null>(
    null,
  );

  /** Search button / Enter: run the search with the current text and category. */
  function submitSearch(submitEvent: FormEvent) {
    submitEvent.preventDefault();
    setSubmittedSearch({ term: searchInput.trim(), category });
  }

  // Categories come with every response, so one cheap request fills the filter before any search.
  const categoriesQuery = useQuery({
    queryKey: ["product-categories"],
    queryFn: () => api.searchProducts("", "", 1),
    staleTime: Infinity,
  });
  const productSearchQuery = useQuery({
    queryKey: ["product-search", submittedSearch?.term, submittedSearch?.category],
    queryFn: () => api.searchProducts(submittedSearch!.term, submittedSearch!.category),
    enabled: submittedSearch !== null,
    placeholderData: keepPreviousData,
  });

  const foundProducts = productSearchQuery.data?.products ?? [];

  return (
    <section className="panel product-search" aria-labelledby="product-search-title">
      <div className="product-search-header">
        <h2 id="product-search-title" className="section-title">
          Find a product
        </h2>
        <span className="muted small">
          Acting as <strong>{selectedUserId}</strong>. Actions count after the next retrain.
        </span>
      </div>

      <form className="product-search-controls" role="search" onSubmit={submitSearch}>
        <label htmlFor="product-search" className="visually-hidden">
          Search products
        </label>
        <input
          id="product-search"
          className="search-input"
          placeholder="Search all products by name, brand, category or subcategory… e.g. Nike, Milo, Snacks"
          value={searchInput}
          onChange={(changeEvent) => setSearchInput(changeEvent.target.value)}
          autoComplete="off"
          spellCheck={false}
        />
        <label className="limit-select">
          <span className="visually-hidden">Category</span>
          <select value={category} onChange={(changeEvent) => setCategory(changeEvent.target.value)}>
            <option value="">All categories</option>
            {categoriesQuery.data?.categories.map((categoryName) => (
              <option key={categoryName} value={categoryName}>
                {categoryName}
              </option>
            ))}
          </select>
        </label>
        <button type="submit" className="button">
          Search
        </button>
      </form>

      {productSearchQuery.isError && (
        <p className="error-text small">{productSearchQuery.error.message}</p>
      )}

      {submittedSearch !== null && productSearchQuery.isSuccess && (
        <>
          <p className="muted tiny">
            {productSearchQuery.data.total === 0
              ? "No matching products."
              : `Showing ${foundProducts.length} of ${productSearchQuery.data.total} products`}
          </p>
          <ul className="product-search-results">
            {foundProducts.map((product) => (
              <li key={product.item_id} className="product-search-row">
                <div className="product-search-details">
                  <span className="product-search-name">{product.name ?? product.item_id}</span>
                  <span className="muted tiny">
                    {product.brand} · {product.category} › {product.subcategory} ·{" "}
                    {formatPrice(product.price)}
                  </span>
                </div>
                <div className="product-search-actions">
                  {PRODUCT_ACTIONS.map((productAction) => (
                    <button
                      key={productAction.eventType}
                      type="button"
                      className={`button ${productAction.eventType === "purchase" ? "" : "secondary"}`}
                      onClick={() => onProductEvent(product, productAction.eventType)}
                    >
                      {productAction.label}
                    </button>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
