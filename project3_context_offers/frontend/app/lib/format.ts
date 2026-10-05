/** Monday = 0, matching pandas' dayofweek used by the model. */
export const DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/** 9 -> "09:00". */
export function formatHour(hour: number): string {
  return `${String(hour).padStart(2, "0")}:00`;
}

/** Same buckets as the backend's daypart_for_hour, so the UI label matches the model's feature. */
export function daypartForHour(hour: number): string {
  if (hour < 6) return "night";
  if (hour < 12) return "morning";
  if (hour < 18) return "afternoon";
  return "evening";
}

/** Locale-formatted date, e.g. "15 Oct 2026". */
export function formatDate(isoDate: string): string {
  return new Date(isoDate).toLocaleDateString(undefined, { dateStyle: "medium" });
}

/** Locale-formatted date and time. */
export function formatDateTime(isoDate: string): string {
  return new Date(isoDate).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** 0.42 -> "42%". */
export function formatPercent(fraction: number, digits = 0): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** The browser's current hour and weekday; JS getDay() has Sunday = 0, so shift to Monday = 0. */
export function currentLocalContextTime(): { hour: number; day_of_week: number } {
  const now = new Date();
  return { hour: now.getHours(), day_of_week: (now.getDay() + 6) % 7 };
}

/** Display names for event types and offer channels. */
export const EVENT_LABELS = {
  impression: "Seen",
  click: "Clicked",
  redeem: "Redeemed",
} as const;

export const CHANNEL_LABELS = {
  both: "Web & app",
  web: "Web only",
  mobile: "App only",
} as const;

/** Display names for the evaluation variants in metrics.json. */
export const MODEL_LABELS: Record<string, string> = {
  contextual: "Contextual ranker",
  no_context: "Without context",
  exposure_labels: "Exposure labels",
  popularity: "Popularity",
  contextual_max_3_per_category: "Contextual, max 3/category",
};
