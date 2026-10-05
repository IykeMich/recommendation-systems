"use client";

import { useQuery } from "@tanstack/react-query";
import { api, type DeviceType, type RecommendationContext, type RecommendationRules } from "../lib/api";
import { currentLocalContextTime, DAY_NAMES, daypartForHour, formatHour } from "../lib/format";

// Used until /v1/context/options loads (or if it fails); mirrors src/config.py.
const FALLBACK_DEVICE_TYPES: DeviceType[] = ["mobile", "tablet", "web"];
const FALLBACK_REGIONS = ["Lagos", "Abuja", "Port Harcourt", "Ibadan", "Enugu"];

/**
 * Controls for the request context (device, region, hour, day) and the business-rule toggles.
 * Controlled component: the page owns the state and receives every change.
 */
export function ContextControls({
  context,
  onContextChange,
  rules,
  onRulesChange,
}: {
  context: RecommendationContext;
  onContextChange: (nextContext: RecommendationContext) => void;
  rules: RecommendationRules;
  onRulesChange: (nextRules: RecommendationRules) => void;
}) {
  // Allowed values never change while the API runs, so fetch once and never treat as stale.
  const contextOptionsQuery = useQuery({
    queryKey: ["context-options"],
    queryFn: api.contextOptions,
    staleTime: Infinity,
  });
  const deviceTypes = contextOptionsQuery.data?.device_types ?? FALLBACK_DEVICE_TYPES;
  const regions = contextOptionsQuery.data?.regions ?? FALLBACK_REGIONS;

  /** Merge a partial change into the current context and pass the result up. */
  function updateContext(contextChanges: Partial<RecommendationContext>) {
    onContextChange({ ...context, ...contextChanges });
  }

  return (
    <section className="panel context-controls" aria-label="Recommendation context">
      <div className="context-grid">
        <fieldset className="control">
          <legend>Device</legend>
          <div className="segmented">
            {deviceTypes.map((deviceType) => (
              <button
                key={deviceType}
                type="button"
                aria-pressed={context.device_type === deviceType}
                onClick={() => updateContext({ device_type: deviceType })}
              >
                {deviceType}
              </button>
            ))}
          </div>
        </fieldset>

        <label className="control">
          <span className="control-label">Region</span>
          <select
            className="select"
            value={context.region}
            onChange={(changeEvent) => updateContext({ region: changeEvent.target.value })}
          >
            {regions.map((regionName) => (
              <option key={regionName} value={regionName}>
                {regionName}
              </option>
            ))}
          </select>
        </label>

        <label className="control hour-control">
          <span className="control-label">
            Time <strong>{formatHour(context.hour)}</strong>
            <span className={`daypart daypart-${daypartForHour(context.hour)}`}>
              {daypartForHour(context.hour)}
            </span>
          </span>
          <input
            type="range"
            min={0}
            max={23}
            value={context.hour}
            onChange={(changeEvent) => updateContext({ hour: Number(changeEvent.target.value) })}
            aria-valuetext={`${formatHour(context.hour)}, ${daypartForHour(context.hour)}`}
          />
          <span className="range-ticks" aria-hidden="true">
            <span>00</span>
            <span>06</span>
            <span>12</span>
            <span>18</span>
            <span>23</span>
          </span>
        </label>

        <fieldset className="control">
          <legend>Day</legend>
          <div className="segmented days">
            {DAY_NAMES.map((dayName, dayIndex) => (
              <button
                key={dayName}
                type="button"
                aria-pressed={context.day_of_week === dayIndex}
                onClick={() => updateContext({ day_of_week: dayIndex })}
              >
                {dayName}
              </button>
            ))}
          </div>
        </fieldset>

        <div className="control now-control">
          <button
            type="button"
            className="button secondary"
            onClick={() => updateContext(currentLocalContextTime())}
          >
            Use current time
          </button>
        </div>
      </div>

      {/* Rule toggles map to enforce_region, enforce_channel and max_per_category (3 or off). */}
      <div className="rules" aria-label="Business rules">
        <span className="rules-title">Rules</span>
        <label className="toggle">
          <input
            type="checkbox"
            checked={rules.enforce_region}
            onChange={(changeEvent) => onRulesChange({ ...rules, enforce_region: changeEvent.target.checked })}
          />
          Only offers available in {context.region}
        </label>
        <label className="toggle">
          <input
            type="checkbox"
            checked={rules.enforce_channel}
            onChange={(changeEvent) => onRulesChange({ ...rules, enforce_channel: changeEvent.target.checked })}
          />
          Only offers that work on {context.device_type}
        </label>
        <label className="toggle">
          <input
            type="checkbox"
            checked={rules.max_per_category !== null}
            onChange={(changeEvent) =>
              onRulesChange({ ...rules, max_per_category: changeEvent.target.checked ? 3 : null })
            }
          />
          Max 3 offers per category
        </label>
      </div>
    </section>
  );
}
