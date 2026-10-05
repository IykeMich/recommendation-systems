// Base URL of the FastAPI backend. NEXT_PUBLIC_* variables are inlined into the browser bundle at
// build time; without one we assume the local `uvicorn ... --port 8005` from the README.
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8005";

/** The policy's three possible outcomes for a risk score (decided server-side, never by the browser). */
export type Decision = "ALLOW" | "REVIEW" | "BLOCK";
/** What an analyst concluded after reviewing an alert. Stored separately from the model's decision. */
export type Outcome = "confirmed_fraud" | "legitimate";

/**
 * Body of POST /score. Mirrors the backend's `Transaction` model, which forbids extra fields,
 * so a client can't send its own risk_score. The server needs either `hour` or
 * `transaction_timestamp`, and either `velocity_1h` or `transaction_timestamp`.
 */
export type TransactionInput = {
  transaction_id: string;
  customer_id: string;
  amount: number;
  currency: string;
  country: string;
  device: string;
  merchant_category: string;
  hour?: number;
  velocity_1h?: number;
  transaction_timestamp?: string;
};

/**
 * The training statistics behind one reason. A reason is only shown when its bucket's training fraud
 * rate is at least 1.5x the overall rate (`lift`). `feature` is null for the fallback reason
 * "No single dominant risk signal", which has no evidence fields.
 */
export type ReasonEvidence = {
  feature: string | null;
  bucket?: string;
  training_fraud_rate?: number;
  overall_fraud_rate?: number;
  training_transactions?: number;
  lift?: number;
};

/**
 * The feature values the model actually scored, as computed by the server.
 * Customer fields compare the transaction with that customer's training-split history:
 * `customer_txn_count` = reference transactions, `amount_vs_customer_mean` = amount / their average
 * (null when there is no history), `customer_device_share` / `customer_country_share` = fraction of
 * their history on this device type / in this country (0 means new).
 * `velocity_source` says where velocity_1h came from: "client" = sent in the request,
 * "decision_store" = counted server-side from this customer's decisions in the previous hour
 * (only when a transaction_timestamp was sent without velocity_1h).
 */
export type FeatureSnapshot = {
  amount: number;
  hour: number;
  velocity_1h: number;
  country: string;
  device: string;
  merchant_category: string;
  customer_txn_count: number;
  customer_mean_amount: number | null;
  amount_vs_customer_mean: number | null;
  customer_device_share: number;
  customer_country_share: number;
  velocity_source: "client" | "decision_store";
};

/**
 * Response of POST /score. `reasons` are plain texts; `reason_evidence` pairs each text with its
 * training evidence. `context` holds true facts (e.g. "device type new for this customer") that are
 * not risk drivers in this data. `idempotent_replay` is true when this transaction_id was already
 * scored with the same payload, so the stored decision was returned instead of a new one.
 */
export type ScoreResult = {
  transaction_id: string;
  risk_score: number;
  decision: Decision;
  model_version: string;
  policy_version: string;
  reasons: string[];
  reason_evidence: { text: string; evidence: ReasonEvidence }[];
  context: string[];
  features: FeatureSnapshot;
  idempotent_replay: boolean;
};

/**
 * One row of GET /v1/decisions (the recent decisions table).
 * `source` is "replay" for held-out test transactions streamed via /v1/replay.
 * `synthetic_label` exists only for those replayed rows: the dataset's known fraud label (1 = fraud),
 * stored separately as if the true outcome arrived later. It lets the dashboard measure precision
 * without waiting for analysts. It is null for transactions scored through /score.
 */
export type DecisionRow = {
  transaction_id: string;
  customer_id: string;
  amount: number;
  risk_score: number;
  decision: Decision;
  created_at: string;
  transaction_timestamp: string | null;
  source: "api" | "replay";
  outcome: Outcome | null;
  synthetic_label: 0 | 1 | null;
  reasons: string[];
};

/**
 * A customer's training-period reference history. Customers absent from the training split are
 * `known: false`. `devices` / `countries` map each value to how many reference transactions used it.
 */
export type CustomerProfile =
  | { customer_id: string; known: false }
  | {
      customer_id: string;
      known: true;
      reference_transactions: number;
      mean_amount: number;
      devices: Record<string, number>;
      countries: Record<string, number>;
    };

/**
 * Full stored record from GET /v1/decisions/{id}, shown in the alert drawer. Unlike DecisionRow,
 * `reasons` carry their evidence, and it adds the original payload, versions, analyst outcome
 * and the customer's profile.
 */
export type DecisionDetail = Omit<DecisionRow, "reasons"> & {
  payload: TransactionInput;
  reasons: { text: string; evidence: ReasonEvidence }[];
  context: string[];
  features: FeatureSnapshot;
  model_version: string;
  policy_version: string;
  outcome_note: string | null;
  outcome_at: string | null;
  customer_profile: CustomerProfile;
};

/**
 * KPIs from GET /v1/stats, aggregated over the decision store.
 * "Flagged" means REVIEW or BLOCK. The *_labelled_* counts only include replayed rows that have a
 * synthetic label: `flagged_and_labelled_fraud / flagged_with_label` is the live precision.
 * `rejected_requests` counts /score calls that failed validation (422); the latest 5 are listed.
 */
export type Stats = {
  scored: number;
  review: number;
  block: number;
  average_risk: number;
  latest_ingestion: string | null;
  confirmed_fraud: number;
  confirmed_legitimate: number;
  flagged_and_labelled_fraud: number;
  flagged_with_label: number;
  allowed_but_labelled_fraud: number;
  rejected_requests: number;
  latest_rejections: { received_at: string; errors: { field: string; message: string }[] }[];
};

/** Counts of correct and incorrect predictions at one threshold (fraud = positive class). */
export type ConfusionMatrix = { true_negative: number; false_positive: number; false_negative: number; true_positive: number };

/** Classification metrics when every score >= `threshold` is treated as fraud; `flag_rate` = share flagged. */
export type ThresholdMetrics = {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  flag_rate: number;
  confusion_matrix: ConfusionMatrix;
};

/**
 * Score-quality metrics for one model on one data split. PR-AUC suits rare fraud better than ROC-AUC;
 * the Brier score measures calibration (lower is better).
 */
export type ScoreMetrics = {
  roc_auc: number;
  pr_auc: number;
  brier_score: number;
  at_review_threshold: ThresholdMetrics;
  at_block_threshold: ThresholdMetrics;
};

/**
 * Response of GET /v1/model: the training manifest (versions, data snapshot, feature lists),
 * metrics.json (every compared model on validation and test, plus decision mixes with and without
 * calibration), the threshold table and the training data-quality report.
 */
export type ModelInfo = {
  manifest: {
    model_name: string;
    model_version: string;
    selected_model: string;
    calibration: string;
    feature_schema_version: string;
    threshold_policy_version: string;
    trained_at: string;
    training_data_snapshot: { file: string; sha256: string; rows: number };
    features: { numeric: string[]; categorical: string[] };
  };
  metrics: {
    selected_model: string;
    served_model: string;
    selection_rule: string;
    split: { method: string } & Record<"train" | "validation" | "test", { rows: number; fraud_rate: number }>;
    models: Record<string, { validation: ScoreMetrics; test: ScoreMetrics }>;
    test_decision_mix: Record<Decision, number>;
    test_decision_mix_uncalibrated: Record<Decision, number>;
  };
  // One row per candidate threshold, with the confusion-matrix counts flattened into the row.
  threshold_table: ({ threshold: number; precision: number; recall: number; f1: number; flag_rate: number } & ConfusionMatrix)[];
  data_quality: Record<string, number | string | string[]>;
};

/** Active policy: score >= block_threshold → BLOCK, >= review_threshold → REVIEW, else ALLOW. */
export type Policy = { review_threshold: number; block_threshold: number; version: string };

/**
 * Result of GET /v1/policy/simulate: what a candidate policy would do on the validation set.
 * `decision_mix` holds fractions; `flagged` = metrics at the review threshold (REVIEW or BLOCK),
 * `blocked` = metrics at the block threshold.
 */
export type PolicySimulation = {
  validation_transactions: number;
  decision_mix: Record<Decision, number>;
  flagged: ThresholdMetrics;
  blocked: ThresholdMetrics;
};

/**
 * Result of POST /v1/replay. `duplicates_ignored` counts transactions that were already scored
 * (idempotent replays); `next_offset` is where the next batch should start in the test set.
 */
export type ReplayResult = {
  processed: number;
  duplicates_ignored: number;
  decisions: Record<Decision, number>;
  next_offset: number;
};

/** Error thrown for failed API calls. `status` is the HTTP status, or 0 when the API was unreachable. */
export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

/**
 * Calls the API and parses the JSON response. Every failure is turned into an ApiError with a
 * readable message, so panels can show `error.message` directly.
 */
async function requestJson<ResponseBody>(path: string, requestOptions?: RequestInit): Promise<ResponseBody> {
  let response: Response;
  // fetch only rejects on network failure (server down, CORS), not on HTTP error statuses.
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...requestOptions,
      headers: { "Content-Type": "application/json", ...requestOptions?.headers },
    });
  } catch {
    throw new ApiError(`Can't reach the API at ${API_BASE_URL}. Is uvicorn running?`, 0);
  }
  if (!response.ok) {
    // Flatten FastAPI's `detail` into one line: HTTPException sends a string (e.g. 409, 404);
    // the validation handler sends a list of {field, message} (422). Anything else gets a generic message.
    const errorBody = await response.json().catch(() => null);
    const detail = errorBody?.detail;
    const errorMessage = typeof detail === "string"
      ? detail
      : Array.isArray(detail)
        ? detail.map((error: { field?: string; message?: string }) => `${error.field || "request"}: ${error.message}`).join("; ")
        : `Request failed (${response.status})`;
    throw new ApiError(errorMessage, response.status);
  }
  return response.json();
}

/** Typed client: one function per backend endpoint (see the README's API table). */
export const api = {
  health: () => requestJson<{ status: string; model_version: string; policy_version: string }>("/health"),
  score: (transaction: TransactionInput) =>
    requestJson<ScoreResult>("/score", { method: "POST", body: JSON.stringify(transaction) }),
  // Streams `count` held-out test transactions, starting at `offset`, through the scoring path.
  replay: (count: number, offset: number) =>
    requestJson<ReplayResult>("/v1/replay", { method: "POST", body: JSON.stringify({ count, offset }) }),
  // A null decision means "all decisions"; otherwise the server filters by ALLOW/REVIEW/BLOCK.
  decisions: (decision: Decision | null, limit = 100) =>
    requestJson<{ decisions: DecisionRow[] }>(`/v1/decisions?limit=${limit}${decision ? `&decision=${decision}` : ""}`),
  decisionDetail: (transactionId: string) =>
    requestJson<DecisionDetail>(`/v1/decisions/${encodeURIComponent(transactionId)}`),
  recordOutcome: (transactionId: string, outcome: Outcome) =>
    requestJson<{ outcome: Outcome }>(`/v1/decisions/${encodeURIComponent(transactionId)}/outcome`, {
      method: "POST",
      body: JSON.stringify({ outcome }),
    }),
  stats: () => requestJson<Stats>("/v1/stats"),
  modelInfo: () => requestJson<ModelInfo>("/v1/model"),
  policy: () => requestJson<Policy>("/v1/policy"),
  // Saves new thresholds as a new policy version; affects new decisions only, no retraining.
  updatePolicy: (reviewThreshold: number, blockThreshold: number) =>
    requestJson<Policy>("/v1/policy", {
      method: "PUT",
      body: JSON.stringify({ review_threshold: reviewThreshold, block_threshold: blockThreshold }),
    }),
  // Read-only preview of a candidate policy on the validation set; nothing is saved.
  simulatePolicy: (reviewThreshold: number, blockThreshold: number) =>
    requestJson<PolicySimulation>(
      `/v1/policy/simulate?review_threshold=${reviewThreshold}&block_threshold=${blockThreshold}`,
    ),
};
