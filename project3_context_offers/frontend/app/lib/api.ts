/** FastAPI base URL; set NEXT_PUBLIC_API_URL in .env.local to point elsewhere. */
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8003";

export type DeviceType = "mobile" | "tablet" | "web";

/** The "right now" context sent with each request. day_of_week: Monday = 0 (pandas convention). */
export type RecommendationContext = {
  device_type: DeviceType;
  region: string;
  hour: number;
  day_of_week: number;
};

/** Business-rule toggles; max_per_category null means no per-category cap. */
export type RecommendationRules = {
  enforce_region: boolean;
  enforce_channel: boolean;
  max_per_category: number | null;
};

/** A reason label and its contribution to the score's logit (0 for cold-start reasons). */
export type OfferReason = { label: string; weight: number };

/** One ranked offer: score is a predicted probability, or relative popularity for cold start. */
export type RecommendedOffer = {
  offer_id: string;
  title: string;
  category: string;
  discount_pct: number;
  min_spend: number;
  channel: "web" | "mobile" | "both";
  region: string;
  expires_at: string;
  score: number;
  reasons: OfferReason[];
};

export type RecommendationStrategy =
  | "contextual_ranker"
  | "cold_start_popular"
  | "no_eligible_offers";

/** One candidate-generation rule and how many offers remained after it. */
export type FunnelStep = { step: string; remaining: number };

/** Response of GET /v1/offers/recommendations/{user_id}. */
export type OfferRecommendationResponse = {
  strategy: RecommendationStrategy;
  user_id: string;
  request_id: string;
  context: RecommendationContext & { daypart: string };
  candidate_funnel: FunnelStep[];
  model_trained_at: string;
  recommendations: RecommendedOffer[];
};

export type UserProfile = {
  user_id: string;
  segment: string;
  age_band: string;
  preferred_category: string;
  device_type: DeviceType;
  region: string;
};

export type OfferEventType = "impression" | "click" | "redeem";

/** A past event; after_training is true if the current model was trained without it. */
export type OfferHistoryEvent = {
  offer_id: string;
  title: string | null;
  category: string | null;
  event_type: OfferEventType;
  timestamp: string;
  device_type: DeviceType;
  region: string;
  after_training: boolean;
};

export type UserDetail = {
  user_id: string;
  profile: UserProfile | null;
  event_type_counts: Partial<Record<OfferEventType, number>>;
  history: OfferHistoryEvent[];
};

export type UserSearchResponse = { total: number; users: UserProfile[] };

export type RankingMetrics = Record<string, number>;

/** Response of GET /v1/model (and POST /v1/model/retrain). */
export type ModelInfo = {
  model_type: string;
  trained_at: string;
  label_rule: string;
  num_interactions: number;
  num_training_rows: number;
  num_offers: number;
  num_users: number;
  pending_new_interactions: number;
  top_coefficients: { feature: string; label: string; coefficient: number }[];
  evaluation?: {
    protocol: string;
    k: number;
    cutoff: string;
    test_requests: number;
    models: Record<string, RankingMetrics>;
    by_segment: Record<string, Record<string, Record<string, number>>>;
  };
};

export type ContextOptions = { device_types: DeviceType[]; regions: string[] };

/** Body of POST /v1/offers/events. */
export type FeedbackEvent = {
  user_id: string;
  offer_id: string;
  event_type: "click" | "redeem";
  request_id?: string;
  context: RecommendationContext;
};

/** Error carrying the HTTP status and FastAPI's `detail` message. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/**
 * fetch wrapper: prefixes the base URL, sends JSON, and turns network failures and non-2xx
 * responses into ApiError (using FastAPI's `detail` string when there is one).
 */
async function requestJson<ResponseBody>(
  path: string,
  requestOptions?: RequestInit,
): Promise<ResponseBody> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...requestOptions,
      headers: { "Content-Type": "application/json", ...requestOptions?.headers },
    });
  } catch {
    throw new ApiError(`Can't reach the API at ${API_BASE_URL}. Is uvicorn running?`, 0);
  }

  if (!response.ok) {
    const errorBody = await response.json().catch(() => null);
    const errorMessage =
      typeof errorBody?.detail === "string"
        ? errorBody.detail
        : `Request failed (${response.status})`;
    throw new ApiError(errorMessage, response.status);
  }
  return response.json();
}

/** Typed client for every backend endpoint the UI uses. */
export const api = {
  health: () => requestJson<{ status: string }>("/health"),

  contextOptions: () => requestJson<ContextOptions>("/v1/context/options"),

  modelInfo: () => requestJson<ModelInfo>("/v1/model"),

  retrainModel: () => requestJson<ModelInfo>("/v1/model/retrain", { method: "POST" }),

  searchUsers: (searchTerm: string, resultLimit = 8) =>
    requestJson<UserSearchResponse>(
      `/v1/users?search=${encodeURIComponent(searchTerm)}&limit=${resultLimit}`,
    ),

  userDetail: (userId: string) =>
    requestJson<UserDetail>(`/v1/users/${encodeURIComponent(userId)}`),

  offerRecommendations: (
    userId: string,
    context: RecommendationContext,
    rules: RecommendationRules,
    resultLimit: number,
  ) => {
    const queryParameters = new URLSearchParams({
      device_type: context.device_type,
      region: context.region,
      hour: String(context.hour),
      day_of_week: String(context.day_of_week),
      limit: String(resultLimit),
      enforce_region: String(rules.enforce_region),
      enforce_channel: String(rules.enforce_channel),
    });
    // Omit the param entirely when there is no cap (the API treats a missing value as no cap).
    if (rules.max_per_category) {
      queryParameters.set("max_per_category", String(rules.max_per_category));
    }
    return requestJson<OfferRecommendationResponse>(
      `/v1/offers/recommendations/${encodeURIComponent(userId)}?${queryParameters}`,
    );
  },

  recordEvent: (feedbackEvent: FeedbackEvent) =>
    requestJson<{ accepted: boolean; pending_new_interactions: number }>("/v1/offers/events", {
      method: "POST",
      body: JSON.stringify(feedbackEvent),
    }),
};
