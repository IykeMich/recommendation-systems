"use client";

import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { api, ApiError, type Employment, type Profile, type ProfileOverrides, type RiskLevel, type UserDetail } from "../lib/api";
import { EVENT_LABELS, formatDateTime, formatMoney, INCOME_PRESETS, titleCase } from "../lib/format";

/** Used until /v1/profile/options loads (or if it fails). */
const FALLBACK_OPTIONS = {
  age_bands: ["18-24", "25-34", "35-44", "45-54", "55+"],
  employment_types: ["salary", "self_employed", "business_owner", "student"] as Employment[],
  risk_profiles: ["low", "medium", "high"] as RiskLevel[],
};

/**
 * Left panel: the what-if profile editor plus the user's activity history.
 * Inputs show the effective profile (dataset values with overrides applied, as returned by the API).
 */
export function ProfilePanel({
  userId,
  userDetailQuery,
  effectiveProfile,
  profileOverrides,
  onProfileOverridesChange,
}: {
  userId: string;
  userDetailQuery: UseQueryResult<UserDetail, Error>;
  effectiveProfile: Profile | null;
  profileOverrides: ProfileOverrides;
  onProfileOverridesChange: (nextOverrides: ProfileOverrides) => void;
}) {
  const profileOptionsQuery = useQuery({ queryKey: ["profile-options"], queryFn: api.profileOptions, staleTime: Infinity });
  const options = profileOptionsQuery.data ?? FALLBACK_OPTIONS;
  // The user-detail endpoint returns 404 for IDs not in the dataset (cold-start visitor).
  const isUnknownUser = userDetailQuery.error instanceof ApiError && userDetailQuery.error.status === 404;
  const userDetail = userDetailQuery.data;
  const datasetProfile = userDetail?.profile ?? null;
  const hasOverrides = Object.keys(profileOverrides).length > 0;

  /** Set or clear one override field and pass the new override object up to the page. */
  function setOverride<Field extends keyof ProfileOverrides>(field: Field, value: ProfileOverrides[Field]) {
    const nextOverrides = { ...profileOverrides };
    // Choosing the dataset's own value removes the override instead of duplicating it.
    if (value === undefined || value === null || (datasetProfile && datasetProfile[field] === value)) {
      delete nextOverrides[field];
    } else {
      nextOverrides[field] = value;
    }
    onProfileOverridesChange(nextOverrides);
  }

  // Prefer the server's merged profile; fall back to the dataset profile before it arrives.
  const shownProfile = effectiveProfile ?? datasetProfile;

  return (
    <aside className="panel profile-panel">
      <div className="user-heading">
        <span className="avatar" aria-hidden="true">{userId.slice(0, 1)}</span>
        <div>
          <h2>{userId}</h2>
          {isUnknownUser && !hasOverrides && <span className="tag tag-warning">Unknown visitor</span>}
          {hasOverrides && <span className="tag tag-accent">What-if profile</span>}
        </div>
      </div>

      {isUnknownUser && !hasOverrides && (
        <p className="muted small">
          No profile, so income and risk are unknown. Missing data is never treated as eligible, so only products
          with no income minimum and no high risk are offered, ranked by popularity. Fill in a what-if profile
          below to see how that changes.
        </p>
      )}

      {/* What-if editor: each control writes an override; a dot marks overridden fields. */}
      <section className="what-if" aria-label="Profile">
        <div className="section-row">
          <h3 className="section-title">Profile</h3>
          {hasOverrides && (
            <button type="button" className="link-button" onClick={() => onProfileOverridesChange({})}>
              Reset
            </button>
          )}
        </div>

        <label className="field">
          <span className="field-label">
            Monthly income
            {profileOverrides.monthly_income !== undefined && <span className="override-dot" title="What-if value" />}
          </span>
          <input
            className="text-input"
            type="number"
            min={0}
            step={10000}
            placeholder="unknown"
            value={shownProfile?.monthly_income !== null && shownProfile?.monthly_income !== undefined ? Math.round(shownProfile.monthly_income) : ""}
            onChange={(changeEvent) =>
              // An empty input removes the income override.
              setOverride("monthly_income", changeEvent.target.value === "" ? undefined : Number(changeEvent.target.value))
            }
          />
          <span className="presets">
            {INCOME_PRESETS.map((incomePreset) => (
              <button key={incomePreset} type="button" className="preset" onClick={() => setOverride("monthly_income", incomePreset)}>
                {incomePreset >= 1_000_000 ? `${incomePreset / 1_000_000}M` : `${incomePreset / 1000}k`}
              </button>
            ))}
          </span>
        </label>

        <fieldset className="field">
          <legend className="field-label">
            Risk profile
            {profileOverrides.risk_profile !== undefined && <span className="override-dot" title="What-if value" />}
          </legend>
          <div className="segmented">
            {options.risk_profiles.map((riskProfile) => (
              <button
                key={riskProfile}
                type="button"
                aria-pressed={shownProfile?.risk_profile === riskProfile}
                onClick={() => setOverride("risk_profile", riskProfile)}
              >
                {riskProfile}
              </button>
            ))}
          </div>
        </fieldset>

        <div className="field-pair">
          <label className="field">
            <span className="field-label">
              Employment
              {profileOverrides.employment !== undefined && <span className="override-dot" title="What-if value" />}
            </span>
            <select
              className="select"
              value={shownProfile?.employment ?? ""}
              onChange={(changeEvent) => setOverride("employment", (changeEvent.target.value || undefined) as Employment | undefined)}
            >
              <option value="">unknown</option>
              {options.employment_types.map((employmentType) => (
                <option key={employmentType} value={employmentType}>{titleCase(employmentType)}</option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="field-label">
              Age
              {profileOverrides.age_band !== undefined && <span className="override-dot" title="What-if value" />}
            </span>
            <select
              className="select"
              value={shownProfile?.age_band ?? ""}
              onChange={(changeEvent) => setOverride("age_band", changeEvent.target.value || undefined)}
            >
              <option value="">unknown</option>
              {options.age_bands.map((ageBand) => (
                <option key={ageBand} value={ageBand}>{ageBand}</option>
              ))}
            </select>
          </label>
        </div>

        {datasetProfile && (
          <p className="muted tiny">
            Dataset profile: income {formatMoney(datasetProfile.monthly_income)}, {datasetProfile.risk_profile} risk,{" "}
            {titleCase(datasetProfile.employment)}, {datasetProfile.region}, {datasetProfile.existing_products} existing products.
          </p>
        )}
      </section>

      {/* Activity: event counts and recent history; events after training are tagged "new". */}
      {userDetail && (
        <section>
          <h3 className="section-title">Activity</h3>
          <div className="event-counts">
            {(["view", "learn", "apply"] as const).map((eventType) => (
              <div key={eventType} className={`event-count event-${eventType}`}>
                <strong>{userDetail.event_type_counts[eventType] ?? 0}</strong>
                <span>{EVENT_LABELS[eventType].toLowerCase()}</span>
              </div>
            ))}
          </div>
          <ol className="history-list">
            {userDetail.history.map((historyEvent) => (
              <li key={`${historyEvent.product_id}-${historyEvent.event_type}-${historyEvent.timestamp}`}>
                <span className={`event-tag event-${historyEvent.event_type}`}>{EVENT_LABELS[historyEvent.event_type]}</span>
                <span className="history-name" title={historyEvent.product_id}>{historyEvent.name ?? historyEvent.product_id}</span>
                {historyEvent.after_training ? (
                  <span className="tag tag-warning" title="Recorded after the model was trained">new</span>
                ) : (
                  <time className="muted tiny">{formatDateTime(historyEvent.timestamp)}</time>
                )}
              </li>
            ))}
          </ol>
        </section>
      )}
    </aside>
  );
}
