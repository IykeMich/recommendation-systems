import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Project details · Fraud Ops",
  description:
    "What the Fraud Detection & Prevention capstone does, how it works, its measured results on synthetic data, and the tech behind it.",
};

/*
 * Static "Project details" page for recruiters, HR and hiring managers. It is a server component
 * with no API calls, so it renders even when the backend is down. Every number below is copied
 * from backend/artifacts/metrics.json and data_quality.json (model fraud-20260930142557).
 */

const STACK_BADGES = ["Python", "scikit-learn", "FastAPI", "Next.js", "TypeScript", "AWS"];

const FEATURES = [
  "Score a single payment transaction and get an ALLOW, REVIEW or BLOCK decision with a risk score and plain-English reasons.",
  "Stream batches of held-out test transactions through the same scoring path, as if they arrived live.",
  "Work an alert queue: filter decisions, open any alert to see its evidence and the customer's profile, and record an analyst outcome.",
  "Move the review and block thresholds with sliders, preview the impact on validation data, then apply a new policy version without retraining.",
  "Inspect the model card: model comparison, threshold table, training data quality and rejected requests.",
];

const PIPELINE = [
  { title: "Data and trust checks", text: "20,000 synthetic transactions are checked for duplicates, missing values and invalid amounts before anything else runs." },
  { title: "Features and model", text: "Leakage-safe features (time of day, transaction velocity, customer profile) feed four candidate models. The winner is picked on validation data and calibrated so a score reads as a real probability." },
  { title: "Policy and API", text: "A FastAPI service computes features and the score server-side, then a separate, versioned policy turns the score into a decision. Retries are idempotent." },
  { title: "Operations dashboard", text: "This Next.js app polls the API to show live KPIs, alerts, explanations and policy controls." },
];

// Served model (gradient boosting, calibrated) on the held-out test set, policy-v1 (review 0.55 / block 0.85).
const RESULTS = [
  { value: "0.9435", label: "ROC-AUC on held-out test data" },
  { value: "0.635", label: "PR-AUC on held-out test data", note: "Synthetic-rule ceiling: 0.632" },
  { value: "89.3%", label: "Fraud caught (recall) at the review threshold" },
  { value: "69.4%", label: "Precision of flagged transactions" },
  { value: "0.0297", label: "Brier score (calibration; lower is better)" },
  { value: "10.0%", label: "Test transactions sent to review", note: "0% auto-blocked" },
];

const HIGHLIGHTS = [
  "Model and policy are separate: the model estimates risk, a versioned policy decides. Thresholds change from the dashboard without retraining, and every decision records the model and policy version it was made under.",
  "Calibration changed the outcome: the uncalibrated model pushed suspicious scores to about 0.95, sending them all to BLOCK. Isotonic calibration routes them to analyst REVIEW instead, which is the defensible result for this data.",
  "Honest evaluation: selection used validation PR-AUC and the test set was scored once. The README reports that the models sit at the ceiling of the dataset's planted rule rather than claiming fraud skill.",
  "The browser is untrusted: the API rejects unknown fields (a client can't send its own risk score), computes features itself, and turns retried transaction IDs into the stored decision instead of a duplicate alert.",
  "Reasons are evidence-backed: an explanation is shown only when its bucket had at least 1.5× the overall training fraud rate, and the evidence is returned with it.",
];

const STACK_GROUPS = [
  { name: "ML", items: ["Python", "pandas", "scikit-learn (gradient boosting, random forest, logistic regression)", "isotonic calibration", "Jupyter"] },
  { name: "Backend", items: ["FastAPI", "Pydantic", "SQLite decision store", "pytest (39 tests)"] },
  { name: "Frontend", items: ["Next.js", "React", "TypeScript", "TanStack Query"] },
  { name: "Cloud (designed, not deployed)", items: ["AWS SAM", "API Gateway", "Lambda", "Kinesis", "SageMaker", "DynamoDB", "CloudWatch", "S3"] },
];

const RUN_COMMANDS = `cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.train          # optional: rebuild artifacts/
python -m pytest -q
uvicorn api.main:app --reload --port 8005

cd ../frontend               # second terminal
npm install && cp .env.example .env.local && npm run dev
# → http://localhost:3005`;

export default function ProjectDetailsPage() {
  return (
    <div className="page about-page">
      <header className="site-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">!</span>
          <div>
            <h1>Project details</h1>
            <p className="muted small">Fraud Detection &amp; Prevention capstone</p>
          </div>
        </div>
        <Link href="/" className="header-link">← Back to the app</Link>
      </header>

      <section className="panel about-hero">
        <h2 className="about-title">Fraud Ops: a fraud-risk scoring platform, end to end</h2>
        <p className="about-lead">
          A working system that scores payment transactions for fraud risk, decides whether to allow, review or
          block them, explains why, and gives analysts a dashboard to act on the alerts.
        </p>
        <ul className="badge-row" aria-label="Tech stack">
          {STACK_BADGES.map((badge) => (
            <li key={badge} className="tech-badge">{badge}</li>
          ))}
        </ul>
      </section>

      <p className="disclaimer" role="note">
        Built on synthetic transactions with a learning-only fraud label. It is not a live financial control
        system, and the results below measure the pipeline, not real-world fraud performance.
      </p>

      <div className="about-grid">
        <section className="panel">
          <h2 className="section-title">The problem</h2>
          <p>
            Payment providers lose money to fraud, but blocking too eagerly frustrates genuine customers and
            floods analysts with false alarms. The business needs a risk score it can trust, a decision rule it
            can tune without retraining, and a clear reason for every alert.
          </p>
        </section>

        <section className="panel">
          <h2 className="section-title">What it does</h2>
          <ul className="about-list">
            {FEATURES.map((feature) => <li key={feature}>{feature}</li>)}
          </ul>
        </section>
      </div>

      <section className="panel about-section">
        <h2 className="section-title">How it works</h2>
        <ol className="pipeline">
          {PIPELINE.map((step, index) => (
            <li key={step.title}>
              <span className="pipeline-step" aria-hidden="true">{index + 1}</span>
              <div>
                <h3>{step.title}</h3>
                <p className="muted">{step.text}</p>
              </div>
            </li>
          ))}
        </ol>
        <p className="small muted">
          <strong>AWS design:</strong> API Gateway and a Kinesis stream feed two Lambda functions, which call a
          SageMaker endpoint for the score, apply the policy from environment variables, and write decisions
          idempotently to DynamoDB, with CloudWatch logs and alarms. It is defined in an AWS SAM template and
          tested locally; it has not been deployed.
        </p>
      </section>

      <section className="about-section">
        <h2 className="section-title">
          Results <span className="muted">· served model on the held-out test set (3,001 transactions, 234 fraud)</span>
        </h2>
        <div className="kpis about-stats">
          {RESULTS.map((result) => (
            <div key={result.label} className="kpi">
              <span className="kpi-label">{result.label}</span>
              <strong className="kpi-value">{result.value}</strong>
              {result.note && <span className="tiny muted">{result.note}</span>}
            </div>
          ))}
        </div>
        <p className="small muted">
          Calibrated gradient boosting, policy-v1 (review at 0.55, block at 0.85). The labels follow one planted
          synthetic rule, so the strong models sit at that rule&apos;s ceiling; with only 234 fraud cases,
          differences of a few PR-AUC points are noise.
        </p>
      </section>

      <div className="about-grid">
        <section className="panel">
          <h2 className="section-title">Engineering highlights</h2>
          <ul className="about-list">
            {HIGHLIGHTS.map((highlight) => <li key={highlight}>{highlight}</li>)}
          </ul>
        </section>

        <section className="panel">
          <h2 className="section-title">Tech stack</h2>
          <dl className="stack-groups">
            {STACK_GROUPS.map((group) => (
              <div key={group.name}>
                <dt>{group.name}</dt>
                <dd>
                  <ul className="badge-row">
                    {group.items.map((item) => <li key={item} className="tech-badge">{item}</li>)}
                  </ul>
                </dd>
              </div>
            ))}
          </dl>
        </section>
      </div>

      <section className="panel about-section">
        <h2 className="section-title">Run it locally</h2>
        <pre className="payload code-block"><code>{RUN_COMMANDS}</code></pre>
        <p className="small muted">The API listens on port 8005 (Swagger UI at <code>/docs</code>); the dashboard on port 3005.</p>
      </section>

      <p className="about-footer">
        <Link href="/" className="header-link">← Back to the app</Link>
      </p>
    </div>
  );
}
