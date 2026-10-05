/**
 * Typed client for the ShopSmart FastAPI backend. The types mirror the JSON returned by api/main.py.
 */

/** Backend base URL; set NEXT_PUBLIC_API_URL in .env.local (inlined into the browser bundle at build). */
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8002";

/** Which path produced the list: the personal hybrid model, or the cold-start popularity fallback. */
export type RecommendationStrategy = "adaptive_hybrid" | "popular_fallback";

/** Which part of the hybrid contributed most to a card ("popular" for fallback lists). */
export type RecommendationSource = "similar_products" | "shoppers_like_you" | "popular";

/**
 * Why a product was recommended. `item_id`/`name` is one of the shopper's own products and
 * `your_event` their strongest action on it.
 * - similar_product: the candidate shares `shared` (e.g. subcategory = `value`) with that product.
 * - co_interaction: `shopper_count` shoppers interacted with both that product and the candidate.
 */
export type RecommendationReason =
  | {
      type: "similar_product";
      item_id: string;
      name: string | null;
      your_event: "view" | "cart" | "purchase" | null;
      shared: "subcategory" | "brand" | "category" | "description";
      value: string | null;
    }
  | {
      type: "co_interaction";
      item_id: string;
      name: string | null;
      your_event: "view" | "cart" | "purchase" | null;
      shopper_count: number;
    };

/** Catalog fields the API attaches to items; all null when an item is missing from products.csv. */
export type ProductDetails = {
  name: string | null;
  category: string | null;
  subcategory: string | null;
  brand: string | null;
  price: number | null;
};

/** A catalog product as returned by the product search. */
export type Product = ProductDetails & { item_id: string };

/** GET /v1/products: matching products plus every category (for the filter). */
export type ProductSearchResponse = {
  total: number;
  categories: string[];
  products: Product[];
};

/** One recommended product, with the hybrid part it mainly came from and why. */
export type Recommendation = ProductDetails & {
  item_id: string;
  score: number;
  source: RecommendationSource;
  reasons: RecommendationReason[];
};

/** GET /v1/recommendations/user/{id}; the request id is echoed back with feedback events. */
export type RecommendationResponse = {
  strategy: RecommendationStrategy;
  user_id: string;
  recommendation_request_id: string;
  model_trained_at: string;
  recommendations: Recommendation[];
};

export type ShopperProfile = {
  user_id: string;
  segment: string;
  age_band: string;
  preferred_category: string;
  device_type: string;
  region: string;
};

export type InteractionEventType = "view" | "cart" | "purchase";

/** One past interaction; `pending` = recorded after the last training run, so not yet in the model. */
export type HistoryEvent = ProductDetails & {
  item_id: string;
  event_type: InteractionEventType;
  timestamp: number;
  pending: boolean;
};

/** GET /v1/users/{id}: profile (null for shoppers not in users.csv) plus recent history. */
export type ShopperDetail = {
  user_id: string;
  profile: ShopperProfile | null;
  known_to_model: boolean;
  event_count: number;
  event_type_counts: Partial<Record<InteractionEventType, number>>;
  history: HistoryEvent[];
};

export type ShopperSearchResponse = {
  total: number;
  users: ShopperProfile[];
};

/** Offline metrics keyed like "recall@10", "ndcg@10", "coverage". */
export type RankingMetrics = Record<string, number>;

/** GET /v1/model (and the retrain response): training metadata plus pending interaction count. */
export type ModelInfo = {
  model_type: string;
  trained_at: string;
  event_weights: Record<InteractionEventType, number>;
  num_events: number;
  num_users: number;
  num_items: number;
  sparsity: number;
  pending_new_interactions: number;
  evaluation?: {
    protocol: string;
    k: number;
    evaluated_users: number;
    hybrid?: RankingMetrics;
    item_item: RankingMetrics;
    popularity_baseline: RankingMetrics;
  };
};

/** Clicks are logged for analysis only; view/cart/purchase also become training data. */
export type FeedbackEventType = InteractionEventType | "recommendation_click";

/** Body of POST /v1/events. */
export type FeedbackEvent = {
  user_id: string;
  item_id: string;
  event_type: FeedbackEventType;
  recommendation_request_id?: string;
};

export type FeedbackResponse = {
  accepted: boolean;
  becomes_training_data: boolean;
  pending_new_interactions: number;
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
 * fetch() wrapper: prefixes the base URL, sends JSON, and turns network failures and non-2xx
 * responses into ApiError so TanStack Query exposes them as `error`.
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
    // fetch only rejects on network errors (server down, CORS), never on HTTP status codes.
    throw new ApiError(
      `Can't reach the API at ${API_BASE_URL}. Is uvicorn running?`,
      0,
    );
  }

  // Prefer FastAPI's `detail` string; fall back to a generic message if the body isn't usable JSON.
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

/** One function per backend endpoint, used as queryFn / mutationFn in the components. */
export const api = {
  health: () => requestJson<{ status: string }>("/health"),

  modelInfo: () => requestJson<ModelInfo>("/v1/model"),

  retrainModel: () =>
    requestJson<ModelInfo>("/v1/model/retrain", { method: "POST" }),

  searchShoppers: (searchTerm: string, resultLimit = 8) =>
    requestJson<ShopperSearchResponse>(
      `/v1/users?search=${encodeURIComponent(searchTerm)}&limit=${resultLimit}`,
    ),

  searchProducts: (searchTerm: string, category: string, resultLimit = 24) =>
    requestJson<ProductSearchResponse>(
      `/v1/products?search=${encodeURIComponent(searchTerm)}` +
        `&category=${encodeURIComponent(category)}&limit=${resultLimit}`,
    ),

  shopperDetail: (userId: string) =>
    requestJson<ShopperDetail>(`/v1/users/${encodeURIComponent(userId)}`),

  recommendations: (userId: string, resultLimit: number) =>
    requestJson<RecommendationResponse>(
      `/v1/recommendations/user/${encodeURIComponent(userId)}?limit=${resultLimit}`,
    ),

  recordEvent: (feedbackEvent: FeedbackEvent) =>
    requestJson<FeedbackResponse>("/v1/events", {
      method: "POST",
      body: JSON.stringify(feedbackEvent),
    }),
};
