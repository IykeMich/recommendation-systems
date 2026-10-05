/**
 * Typed client for the FastAPI backend. Types mirror the JSON the API returns.
 * The base URL comes from NEXT_PUBLIC_API_URL (set in .env.local), defaulting to local uvicorn.
 */
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8004";

export type RiskLevel = "low" | "medium" | "high";
export type Employment = "salary" | "self_employed" | "business_owner" | "student";

/** A user's financial profile; any field can be null when the data is unknown. */
export type Profile = {
  age_band: string | null;
  monthly_income: number | null;
  employment: Employment | null;
  risk_profile: RiskLevel | null;
  region: string | null;
  existing_products: number | null;
};

/** Fields the what-if editor can override. */
export type ProfileOverrides = Partial<Pick<Profile, "age_band" | "monthly_income" | "employment" | "risk_profile" | "existing_products">>;

/** One failed eligibility rule with a human-readable detail. */
export type FailedRule = { rule_id: string; detail: string };

/** The data a reason relies on; `source` says where (history, popularity, rule, catalog, profile). */
export type Evidence = Record<string, string | number | string[] | null> & { source: string };

export type Reason = { text: string; evidence: Evidence };

export type Product = {
  product_id: string;
  name: string;
  category: string;
  risk_level: RiskLevel;
  min_income: number;
  description: string;
};

/** A served product: catalog fields plus model score, reasons and cautions. */
export type RecommendedProduct = Product & {
  score: number;
  reasons: Reason[];
  cautions: Reason[];
};

/** One row of the teaching ranking WITHOUT the eligibility layer. */
export type UnfilteredRow = {
  product_id: string;
  name: string;
  score: number;
  eligible: boolean;
  failed_rules: FailedRule[];
};

export type RecommendationStrategy = "eligibility_then_ranking" | "cold_start_popular" | "no_eligible_products";

/** GET /v1/financial-products/recommendations/{user_id} response. */
export type RecommendationResponse = {
  strategy: RecommendationStrategy;
  user_id: string;
  request_id: string;
  model_version: string;
  profile: Profile | null;
  profile_source: "dataset" | "dataset+what_if" | "what_if" | "none";
  eligibility: {
    rules_version: string;
    total_products: number;
    eligible_products: number;
    ineligible: (Product & { failed_rules: FailedRule[] })[];
  };
  recommendations: RecommendedProduct[];
  unfiltered_ranking?: UnfilteredRow[];
};

export type UserProfile = Profile & { user_id: string };

export type InteractionEventType = "view" | "learn" | "apply";

/** GET /v1/users/{user_id}: profile, event counts and recent history. */
export type UserDetail = {
  user_id: string;
  profile: UserProfile | null;
  event_type_counts: Partial<Record<InteractionEventType, number>>;
  history: {
    product_id: string;
    name: string | null;
    event_type: InteractionEventType;
    timestamp: string;
    // True for events recorded after the current model was trained.
    after_training: boolean;
  }[];
};

export type StrategyMetrics = Record<string, number>;

/** GET /v1/model: metadata, top coefficients, pending feedback and (optional) evaluation. */
export type ModelInfo = {
  model_version: string;
  trained_at: string;
  label_rule: string;
  eligibility_rules_version: string;
  eligibility_rules: { rule_id: string; description: string }[];
  num_interactions: number;
  num_training_rows: number;
  pending_new_interactions: number;
  top_coefficients: { feature: string; label: string; coefficient: number }[];
  evaluation?: {
    k: number;
    cutoff: string;
    test_requests: number;
    applied_products_that_were_ineligible: number;
    mean_eligible_products_per_request: number;
    strategies: Record<string, StrategyMetrics>;
    eligibility_rate_by_risk_profile: Record<string, number>;
    eligibility_rate_by_income_quartile: Record<string, number>;
  };
};

export type ProfileOptions = { age_bands: string[]; employment_types: Employment[]; risk_profiles: RiskLevel[] };

/** POST /v1/financial-products/events body. */
export type FeedbackEvent = {
  user_id: string;
  product_id: string;
  event_type: "recommendation_click" | InteractionEventType;
  request_id?: string;
  recommendation_strategy?: string;
};

/** Error carrying the HTTP status (0 = network failure) so callers can treat 404 specially. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/**
 * fetch wrapper: prefixes the base URL, sends JSON headers, and turns network failures and
 * non-2xx responses into ApiError (using FastAPI's `detail` string when present).
 */
async function requestJson<ResponseBody>(path: string, requestOptions?: RequestInit): Promise<ResponseBody> {
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
    const errorMessage = typeof errorBody?.detail === "string" ? errorBody.detail : `Request failed (${response.status})`;
    throw new ApiError(errorMessage, response.status);
  }
  return response.json();
}

/** One function per endpoint, used as TanStack Query queryFn / mutationFn. */
export const api = {
  health: () => requestJson<{ status: string }>("/health"),

  profileOptions: () => requestJson<ProfileOptions>("/v1/profile/options"),

  modelInfo: () => requestJson<ModelInfo>("/v1/model"),

  retrainModel: () => requestJson<ModelInfo>("/v1/model/retrain", { method: "POST" }),

  searchUsers: (searchTerm: string, resultLimit = 8) =>
    requestJson<{ total: number; users: UserProfile[] }>(
      `/v1/users?search=${encodeURIComponent(searchTerm)}&limit=${resultLimit}`,
    ),

  userDetail: (userId: string) => requestJson<UserDetail>(`/v1/users/${encodeURIComponent(userId)}`),

  recommendations: (userId: string, profileOverrides: ProfileOverrides, resultLimit: number) => {
    // Always ask for the unfiltered comparison; only set overrides that have a value.
    const queryParameters = new URLSearchParams({ limit: String(resultLimit), include_unfiltered: "true" });
    for (const [field, value] of Object.entries(profileOverrides)) {
      if (value !== null && value !== undefined) queryParameters.set(field, String(value));
    }
    return requestJson<RecommendationResponse>(
      `/v1/financial-products/recommendations/${encodeURIComponent(userId)}?${queryParameters}`,
    );
  },

  recordEvent: (feedbackEvent: FeedbackEvent) =>
    requestJson<{ accepted: boolean; pending_new_interactions: number }>("/v1/financial-products/events", {
      method: "POST",
      body: JSON.stringify(feedbackEvent),
    }),
};
