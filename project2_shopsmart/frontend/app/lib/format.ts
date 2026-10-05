/** Display helpers shared by the components. */

// Created once and reused: building Intl formatters is relatively expensive.
const priceFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
});

/** Format a price as USD, or an em dash when the product is unknown. */
export function formatPrice(price: number | null): string {
  return price === null ? "—" : priceFormatter.format(price);
}

/** Format a date that is either an ISO string (model trained_at) or Unix seconds (event timestamps). */
export function formatDateTime(isoOrUnixSeconds: string | number): string {
  const date =
    typeof isoOrUnixSeconds === "number"
      ? new Date(isoOrUnixSeconds * 1000)
      : new Date(isoOrUnixSeconds);
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

/** 0.9876 -> "98.8%". */
export function formatPercent(fraction: number): string {
  return `${(fraction * 100).toFixed(1)}%`;
}

/** Past-tense labels for each event type, used in history tags and toasts. */
export const EVENT_LABELS = {
  view: "Viewed",
  cart: "Carted",
  purchase: "Bought",
  recommendation_click: "Clicked",
} as const;
