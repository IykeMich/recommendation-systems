"use client";

import { useState, type FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api, type TransactionInput } from "../lib/api";
import { COUNTRIES, DEVICES, formatPercent, MERCHANT_CATEGORIES, randomTransactionId } from "../lib/format";
import { DecisionBadge, PanelError, RiskBar } from "./ui";

// One-click examples. The second matches the dataset's planted fraud pattern (hours 1–4, velocity >= 6);
// the third looks unusual (a large amount in a country customer F00009 has never used) but amount
// and country carry no signal in this data, so it is still allowed.
const PRESETS: { label: string; values: Partial<TransactionInput> }[] = [
  { label: "Daytime purchase", values: { hour: 14, velocity_1h: 2, amount: 1200, merchant_category: "food" } },
  { label: "03:00, high velocity", values: { hour: 3, velocity_1h: 9, amount: 1500, merchant_category: "travel" } },
  { label: "Large, new country", values: { customer_id: "F00009", hour: 15, velocity_1h: 1, amount: 40000, country: "US", merchant_category: "electronics" } },
];

/**
 * Default form values with a fresh transaction ID. It sends hour and velocity_1h directly (no
 * timestamp), so the server uses them as given ("client" velocity source).
 */
function newTransaction(): TransactionInput {
  return {
    transaction_id: randomTransactionId(), customer_id: "F00001", amount: 1500, currency: "NGN",
    country: "NG", device: "android", merchant_category: "travel", hour: 3, velocity_1h: 9,
  };
}

/**
 * "Score a transaction" form. Posts to /score and shows the decision, risk score and reasons.
 * The transaction ID is the idempotency key: resubmitting the same ID and payload returns the stored
 * decision; the same ID with a changed payload is rejected with 409.
 */
export function ScorePanel({ onScored }: { onScored: (transactionId: string) => void }) {
  const queryClient = useQueryClient();
  // Passing the function (not its result) means newTransaction runs only on the first render.
  const [transaction, setTransaction] = useState<TransactionInput>(newTransaction);
  // On success: refresh the decisions table and KPIs, then open the new decision in the drawer.
  // On error: still refresh ["stats"], because a 422 rejection is counted in "Rejected requests".
  const scoreMutation = useMutation({
    mutationFn: api.score,
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ["decisions"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      onScored(result.transaction_id);
    },
    onError: () => queryClient.invalidateQueries({ queryKey: ["stats"] }),
  });

  /** Type-safe setter for one form field: the value's type must match the chosen field. */
  function update<Field extends keyof TransactionInput>(field: Field, value: TransactionInput[Field]) {
    setTransaction((previous) => ({ ...previous, [field]: value }));
  }

  /** Stops the browser's full-page form submit and sends the transaction through the mutation instead. */
  function submit(submitEvent: FormEvent) {
    submitEvent.preventDefault();
    scoreMutation.mutate(transaction);
  }

  const result = scoreMutation.data;
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Score a transaction</h2>
      </div>
      {/* Presets: start from defaults with a new ID, then overlay the preset's values. */}
      <div className="presets">
        {PRESETS.map((preset) => (
          <button key={preset.label} type="button" className="pill"
            onClick={() => setTransaction({ ...newTransaction(), ...preset.values })}>
            {preset.label}
          </button>
        ))}
      </div>
      <form className="score-form" onSubmit={submit}>
        <label className="field wide">
          <span>Transaction ID <span className="muted">(idempotency key)</span></span>
          <span className="inline">
            <input className="input" value={transaction.transaction_id} onChange={(changeEvent) => update("transaction_id", changeEvent.target.value)} />
            <button type="button" className="button secondary" onClick={() => update("transaction_id", randomTransactionId())}>New ID</button>
          </span>
        </label>
        <label className="field">
          <span>Customer</span>
          <input className="input" value={transaction.customer_id} onChange={(changeEvent) => update("customer_id", changeEvent.target.value)} />
        </label>
        <label className="field">
          <span>Amount (NGN)</span>
          <input className="input" type="number" step="0.01" value={transaction.amount}
            onChange={(changeEvent) => update("amount", Number(changeEvent.target.value))} />
        </label>
        <label className="field">
          <span>Hour</span>
          <input className="input" type="number" min={0} max={23} value={transaction.hour ?? ""}
            onChange={(changeEvent) => update("hour", Number(changeEvent.target.value))} />
        </label>
        <label className="field">
          <span>Velocity (1h)</span>
          <input className="input" type="number" min={1} value={transaction.velocity_1h ?? ""}
            onChange={(changeEvent) => update("velocity_1h", Number(changeEvent.target.value))} />
        </label>
        <label className="field">
          <span>Country</span>
          <select className="input" value={transaction.country} onChange={(changeEvent) => update("country", changeEvent.target.value)}>
            {COUNTRIES.map((country) => <option key={country}>{country}</option>)}
          </select>
        </label>
        <label className="field">
          <span>Device</span>
          <select className="input" value={transaction.device} onChange={(changeEvent) => update("device", changeEvent.target.value)}>
            {DEVICES.map((device) => <option key={device}>{device}</option>)}
          </select>
        </label>
        <label className="field wide">
          <span>Merchant category</span>
          <select className="input" value={transaction.merchant_category} onChange={(changeEvent) => update("merchant_category", changeEvent.target.value)}>
            {MERCHANT_CATEGORIES.map((category) => <option key={category}>{category}</option>)}
          </select>
        </label>
        <button type="submit" className="button wide" disabled={scoreMutation.isPending}>
          {scoreMutation.isPending ? "Scoring…" : "Score transaction"}
        </button>
      </form>
      <PanelError error={scoreMutation.error} />
      {/* Result of the last successful score */}
      {result && (
        <div className="score-result" role="status">
          <div className="score-result-top">
            <DecisionBadge decision={result.decision} />
            <RiskBar riskScore={result.risk_score} />
          </div>
          {result.idempotent_replay && (
            <p className="notice small">Already scored: returned the stored decision, no duplicate alert.</p>
          )}
          {/* Reasons with their training evidence (the fallback reason has none). */}
          <ul className="reasons compact">
            {result.reason_evidence.map((reason) => (
              <li key={reason.text}>
                {reason.text}
                {reason.evidence.training_fraud_rate !== undefined && (
                  <span className="evidence"> · {formatPercent(reason.evidence.training_fraud_rate)} fraud in training</span>
                )}
              </li>
            ))}
          </ul>
          {result.context.length > 0 && <p className="muted small">Context: {result.context.join(" · ")}</p>}
          <p className="muted tiny">{result.model_version} · {result.policy_version}</p>
        </div>
      )}
    </section>
  );
}
