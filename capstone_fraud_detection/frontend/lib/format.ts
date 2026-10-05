// Shared formatter: always two decimal places, with thousands separators.
const amountFormatter = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

/** Formats a money amount as e.g. "NGN 1,500.00". The dataset has one currency, so NGN is the default. */
export function formatAmount(amount: number, currency = "NGN"): string {
  return `${currency} ${amountFormatter.format(amount)}`;
}

/** Turns a 0–1 fraction (rate, precision, recall) into a percentage string, e.g. 0.078 → "7.8%". */
export function formatPercent(fraction: number, digits = 1): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** Formats an ISO timestamp as a local date and time; null (e.g. nothing ingested yet) shows "—". */
export function formatDateTime(isoDate: string | null): string {
  if (!isoDate) return "—";
  return new Date(isoDate).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "medium" });
}

/** Formats an ISO timestamp as a local time only (used in the compact decisions table). */
export function formatTime(isoDate: string): string {
  return new Date(isoDate).toLocaleTimeString(undefined, { timeStyle: "medium" });
}

/** Human-readable names for the model keys used in metrics.json; unknown keys fall back to the raw key. */
export const MODEL_LABELS: Record<string, string> = {
  logistic_regression: "Logistic regression (guide baseline)",
  logistic_regression_hour_categorical: "Logistic regression, hour as category",
  random_forest: "Random forest",
  hist_gradient_boosting: "Gradient boosting",
  hist_gradient_boosting_calibrated: "Gradient boosting, calibrated (served)",
  random_forest_calibrated: "Random forest, calibrated (served)",
  synthetic_rule_reference: "Synthetic rule (reference ceiling)",
};

// Category values seen in the training data, used for the score form's dropdowns.
// The API also accepts unseen values; the model handles them without crashing.
export const DEVICES = ["android", "ios", "web"];
export const COUNTRIES = ["NG", "GH", "GB", "US"];
export const MERCHANT_CATEGORIES = ["digital", "electronics", "food", "fuel", "retail", "travel"];

/**
 * Generates a fresh, practically unique transaction ID (time in base 36 plus 4 random chars).
 * The transaction ID is the API's idempotency key, so a new one means a new decision.
 */
export function randomTransactionId(): string {
  return `TXN-${Date.now().toString(36).toUpperCase()}-${Math.random().toString(36).slice(2, 6).toUpperCase()}`;
}
