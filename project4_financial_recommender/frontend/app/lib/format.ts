const moneyFormatter = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

/** The dataset has no currency, so amounts are shown as plain numbers. */
export function formatMoney(amount: number | null | undefined): string {
  return amount === null || amount === undefined ? "unknown" : moneyFormatter.format(amount);
}

/** Localised medium date + short time for an ISO timestamp. */
export function formatDateTime(isoDate: string): string {
  return new Date(isoDate).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** 0.123 -> "12%" (or "12.3%" with digits = 1). */
export function formatPercent(fraction: number, digits = 0): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** "self_employed" -> "Self Employed"; null/undefined -> "Unknown". */
export function titleCase(text: string | null | undefined): string {
  return (text ?? "unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

/** Display labels for interaction types. */
export const EVENT_LABELS = { view: "Viewed", learn: "Read about", apply: "Applied" } as const;

/** Display labels for eligibility rule IDs. */
export const RULE_LABELS: Record<string, string> = {
  min_income: "Minimum income",
  risk_suitability: "Risk suitability",
};

/** Display labels for the four evaluated strategies (keys match metrics.json). */
export const STRATEGY_LABELS: Record<string, string> = {
  random_eligible: "Random (eligible only)",
  popularity_eligible: "Popularity (eligible only)",
  ranker_without_eligibility: "Ranker, NO eligibility layer",
  eligibility_then_ranking: "Eligibility → ranker",
};

/** Quick-pick monthly incomes for the what-if editor. */
export const INCOME_PRESETS = [50_000, 100_000, 300_000, 600_000, 1_200_000, 2_500_000];
