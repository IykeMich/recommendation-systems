"use client";

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import { PanelError } from "./ui";

/**
 * "Stream test transactions" panel: simulates a Kinesis-style stream by sending batches of held-out
 * test transactions to POST /v1/replay. Replayed rows carry their synthetic label, which powers the
 * "Flagged that were fraud" KPI. "Resend previous" demonstrates idempotency.
 */
export function ReplayPanel() {
  const queryClient = useQueryClient();
  const [batchSize, setBatchSize] = useState(250);
  // Position in the test set for the next batch. Kept only in component state, so it restarts at 0
  // on page reload (re-sent rows are then returned as stored decisions, not scored twice).
  const [nextOffset, setNextOffset] = useState(0);
  // On success, advance the offset and refetch the decisions table and KPIs so new rows appear at once.
  const replayMutation = useMutation({
    mutationFn: () => api.replay(batchSize, nextOffset),
    onSuccess: (result) => {
      setNextOffset(result.next_offset);
      queryClient.invalidateQueries({ queryKey: ["decisions"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
  });
  const result = replayMutation.data;
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Stream test transactions</h2>
      </div>
      <p className="muted small">
        Pushes held-out test transactions through the same scoring path, the way a Kinesis consumer would.
        Resending a batch shows idempotency: duplicates are ignored.
      </p>
      <div className="replay-controls">
        <label className="field">
          <span>Batch size</span>
          <select className="input" value={batchSize} onChange={(changeEvent) => setBatchSize(Number(changeEvent.target.value))}>
            {[50, 250, 1000, 3000].map((size) => <option key={size} value={size}>{size}</option>)}
          </select>
        </label>
        <button type="button" className="button" disabled={replayMutation.isPending} onClick={() => replayMutation.mutate()}>
          {replayMutation.isPending ? "Streaming…" : `Send next ${batchSize}`}
        </button>
        {/* Steps the offset back one batch (uses the current batch size); the next send repeats those rows. */}
        <button type="button" className="button secondary" disabled={replayMutation.isPending || nextOffset === 0}
          onClick={() => setNextOffset(Math.max(0, nextOffset - batchSize))} title="Step back so the next send repeats the previous batch">
          Resend previous
        </button>
      </div>
      <PanelError error={replayMutation.error} />
      {/* Summary of the last batch; duplicates are rows the store had already scored. */}
      {result && (
        <p className="small" role="status">
          Processed {result.processed}: {result.decisions.ALLOW} allow, {result.decisions.REVIEW} review, {result.decisions.BLOCK} block
          {result.duplicates_ignored > 0 && <> · <strong>{result.duplicates_ignored} duplicates returned stored decisions</strong></>}
        </p>
      )}
    </section>
  );
}
