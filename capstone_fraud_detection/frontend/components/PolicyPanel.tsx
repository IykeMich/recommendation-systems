"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import { formatPercent } from "../lib/format";
import { useDebouncedValue } from "../lib/useDebouncedValue";
import { PanelError } from "./ui";

/**
 * Policy panel: threshold sliders, a live preview of their effect on the validation set, and a button
 * to apply them as a new policy version. This is "the model estimates risk; the policy decides":
 * thresholds change without retraining, and stored decisions keep their original policy version.
 */
export function PolicyPanel() {
  const queryClient = useQueryClient();
  // ["policy"]: the active thresholds. Uses the default 30 s staleTime; no polling.
  const policyQuery = useQuery({ queryKey: ["policy"], queryFn: api.policy });
  // Local, unsaved slider values. null until the active policy has loaded.
  const [thresholds, setThresholds] = useState<{ review: number; block: number } | null>(null);

  // Seed the sliders from the loaded policy, once. After that the user's edits are kept, so
  // background refetches of ["policy"] never overwrite slider positions.
  useEffect(() => {
    if (policyQuery.data && thresholds === null) {
      setThresholds({ review: policyQuery.data.review_threshold, block: policyQuery.data.block_threshold });
    }
  }, [policyQuery.data, thresholds]);

  // Wait 200 ms after the slider stops moving before simulating, rather than one request per pixel.
  const debouncedThresholds = useDebouncedValue(thresholds, 200);
  // ["policy-simulation", thresholds]: one cached result per threshold pair. Disabled when the pair is
  // invalid (review > block), which the API would reject. Returning `previous` as placeholder keeps the
  // last result on screen while the next one loads, so the preview doesn't flicker.
  const simulationQuery = useQuery({
    queryKey: ["policy-simulation", debouncedThresholds],
    queryFn: () => api.simulatePolicy(debouncedThresholds!.review, debouncedThresholds!.block),
    enabled: debouncedThresholds !== null && debouncedThresholds.review <= debouncedThresholds.block,
    placeholderData: (previous) => previous,
  });
  // PUT /v1/policy with the current slider values. On success, write the returned policy straight into
  // the ["policy"] cache (no refetch needed) and refetch ["health"], whose header shows the policy version.
  const applyMutation = useMutation({
    mutationFn: () => api.updatePolicy(thresholds!.review, thresholds!.block),
    onSuccess: (policy) => {
      queryClient.setQueryData(["policy"], policy);
      queryClient.invalidateQueries({ queryKey: ["health"] });
    },
  });

  const policy = policyQuery.data;
  // isChanged: the sliders differ from the active policy (enables Apply and shows Reset).
  // isInvalid: review above block would leave no REVIEW band, so the server forbids it.
  const isChanged = policy && thresholds && (thresholds.review !== policy.review_threshold || thresholds.block !== policy.block_threshold);
  const isInvalid = thresholds !== null && thresholds.review > thresholds.block;
  const simulation = simulationQuery.data;

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Policy</h2>
        {policy && <code className="version">{policy.version}</code>}
      </div>
      <p className="muted small">The model estimates risk; the policy decides. Changing thresholds needs no retraining and only affects new decisions.</p>
      <PanelError error={policyQuery.error} />
      {thresholds && (
        <>
          {/* One slider per threshold; each change replaces only that zone's value. */}
          {(["review", "block"] as const).map((zone) => (
            <label key={zone} className="slider">
              <span className="slider-label">
                {zone === "review" ? "REVIEW at or above" : "BLOCK at or above"} <strong>{thresholds[zone].toFixed(2)}</strong>
              </span>
              <input type="range" min={0} max={1} step={0.01} value={thresholds[zone]}
                onChange={(changeEvent) => setThresholds({ ...thresholds, [zone]: Number(changeEvent.target.value) })} />
            </label>
          ))}
          {isInvalid && <p className="error-text small">The review threshold must not exceed the block threshold.</p>}
          {/* Validation-set preview: decision mix bar plus the trade-offs an analyst cares about. */}
          {simulation && !isInvalid && (
            <div className="simulation">
              <p className="muted tiny">On {simulation.validation_transactions.toLocaleString()} validation transactions:</p>
              <div className="mix-bar" aria-label="Decision mix">
                {(["ALLOW", "REVIEW", "BLOCK"] as const).map((decision) => (
                  <span key={decision} className={`mix-${decision.toLowerCase()}`} style={{ width: `${simulation.decision_mix[decision] * 100}%` }}
                    title={`${decision} ${formatPercent(simulation.decision_mix[decision])}`} />
                ))}
              </div>
              {/* "flagged" = metrics at the review threshold (REVIEW + BLOCK); "blocked" = at the block one. */}
              <dl className="simulation-stats">
                <dt>Review + block</dt><dd>{formatPercent(simulation.decision_mix.REVIEW + simulation.decision_mix.BLOCK)}</dd>
                <dt>Blocked</dt><dd>{formatPercent(simulation.decision_mix.BLOCK)}</dd>
                <dt>Fraud caught (recall)</dt><dd>{formatPercent(simulation.flagged.recall)}</dd>
                <dt>Flags that are fraud (precision)</dt><dd>{formatPercent(simulation.flagged.precision)}</dd>
                <dt>Legitimate blocked</dt><dd>{simulation.blocked.confusion_matrix.false_positive}</dd>
                <dt>Fraud missed</dt><dd>{simulation.flagged.confusion_matrix.false_negative}</dd>
              </dl>
            </div>
          )}
          <div className="policy-actions">
            <button type="button" className="button" disabled={!isChanged || isInvalid || applyMutation.isPending} onClick={() => applyMutation.mutate()}>
              {applyMutation.isPending ? "Applying…" : "Apply policy"}
            </button>
            {/* Reset moves the sliders back to the active policy's thresholds. */}
            {isChanged && policy && (
              <button type="button" className="button secondary"
                onClick={() => setThresholds({ review: policy.review_threshold, block: policy.block_threshold })}>
                Reset
              </button>
            )}
          </div>
          <PanelError error={applyMutation.error} />
        </>
      )}
    </section>
  );
}
