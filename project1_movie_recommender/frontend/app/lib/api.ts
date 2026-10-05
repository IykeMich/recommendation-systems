/** Base URL of the FastAPI backend; override with NEXT_PUBLIC_API_BASE_URL (inlined at build time). */
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8001";

// Response shapes below mirror the JSON returned by ../backend/api/main.py.

/** One movie row from the Project 1 catalog. `genres` is pipe-separated, e.g. "Drama|Comedy". */
export type Movie = {
  item_id: string;
  title: string;
  genres: string;
  description: string;
  year: number;
  language: string;
};

/** GET /v1/movies: `total` counts all matches; `movies` holds only the first `limit`. */
export type MovieSearchResponse = {
  total: number;
  movies: Movie[];
};

/** A similar movie; `score` is TF-IDF cosine similarity (0..1). */
export type MovieRecommendation = {
  item_id: string;
  title: string;
  genres: string;
  score: number;
};

/** GET /v1/recommendations/content/{item_id} response. */
export type ContentRecommendationResponse = {
  strategy: string;
  source_item_id: string;
  recommendations: MovieRecommendation[];
};

/** Body for POST /v1/recommendations/content/query: a new movie that is not in the catalog. */
export type MovieQuery = {
  title: string;
  genres: string;
  description: string;
};

/** POST /v1/recommendations/content/query response; `source` echoes the described movie. */
export type MovieQueryRecommendationResponse = {
  strategy: string;
  source: Pick<MovieQuery, "title" | "genres">;
  recommendations: MovieRecommendation[];
};

/** GET /v1/retail/users/{user_id}: the users.csv row plus the shopper's interaction count. */
export type RetailUserProfile = {
  user_id: string;
  segment: string;
  age_band: string;
  preferred_category: string;
  device_type: string;
  region: string;
  event_count: number;
};

/** A recommended product. `score` is absent for cold-start (popularity) results. */
export type ProductRecommendation = {
  item_id: string;
  name: string;
  category: string;
  score?: number;
};

/** GET /v1/recommendations/user/{user_id}; `strategy` says which algorithm produced the list. */
export type UserRecommendationResponse = {
  strategy: "collaborative_item_item_baseline" | "popular_cold_start";
  user_id: string;
  recommendations: ProductRecommendation[];
};

/** Body for POST /v1/events (matches the backend's `EventIn` model; optional fields have defaults). */
export type RetailEvent = {
  user_id: string;
  item_id: string;
  event_type: string;
  event_value?: number;
  project?: string;
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
 * Fetch `path` from the API and parse the JSON body.
 * Normalises failures into ApiError: status 0 for network errors, otherwise the HTTP status + `detail`.
 */
async function requestJson<ResponseBody>(
  path: string,
  requestOptions?: RequestInit,
): Promise<ResponseBody> {
  let response: Response;
  // fetch() only rejects on network failure (server down, CORS), not on 4xx/5xx responses.
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...requestOptions,
      headers: { "Content-Type": "application/json", ...requestOptions?.headers },
    });
  } catch {
    throw new ApiError(
      `Can't reach the API at ${API_BASE_URL}. Is uvicorn running?`,
      0,
    );
  }

  // HTTP error: prefer FastAPI's `detail` string (e.g. "Unknown user_id"); the body may not be JSON.
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

/** Typed client: one method per backend endpoint. IDs are URL-encoded before going into paths. */
export const api = {
  health: () => requestJson<{ status: string }>("/health"),

  searchMovies: (searchTerm: string, resultLimit = 12) =>
    requestJson<MovieSearchResponse>(
      `/v1/movies?search=${encodeURIComponent(searchTerm)}&limit=${resultLimit}`,
    ),

  contentRecommendations: (movieId: string, resultLimit: number) =>
    requestJson<ContentRecommendationResponse>(
      `/v1/recommendations/content/${encodeURIComponent(movieId)}?limit=${resultLimit}`,
    ),

  queryContentRecommendations: (movieQuery: MovieQuery, resultLimit: number) =>
    requestJson<MovieQueryRecommendationResponse>(
      `/v1/recommendations/content/query?limit=${resultLimit}`,
      { method: "POST", body: JSON.stringify(movieQuery) },
    ),

  retailUserProfile: (userId: string) =>
    requestJson<RetailUserProfile>(
      `/v1/retail/users/${encodeURIComponent(userId)}`,
    ),

  userRecommendations: (userId: string, resultLimit: number) =>
    requestJson<UserRecommendationResponse>(
      `/v1/recommendations/user/${encodeURIComponent(userId)}?limit=${resultLimit}`,
    ),

  recordEvent: (retailEvent: RetailEvent) =>
    requestJson<{ accepted: boolean }>("/v1/events", {
      method: "POST",
      body: JSON.stringify(retailEvent),
    }),
};
