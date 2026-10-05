import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Project details · ShopSmart Offers",
  description:
    "Context-aware offer recommender: what it does, how it works, measured results and the tech stack behind it.",
};

const STACK_BADGES = ["Python", "scikit-learn", "FastAPI", "Next.js", "TypeScript", "TanStack Query", "AWS SageMaker"];

const FEATURES = [
  "Search for any of 1,000 shoppers and see their profile and offer history (impressions, clicks, redemptions).",
  "Set the shopping context (device, region, hour of day, day of week) and get a fresh Top-10 list of offers.",
  "Pin a context, change something, and see which offers moved up, moved down or are new.",
  "Switch the business rules (region eligibility, device eligibility, max 3 offers per category) and see how many offers each rule removes.",
  "Click or redeem an offer: the shopper’s history updates at once, and “Retrain model” updates the model itself.",
  "Read why each offer was ranked: per-offer reasons, the model’s strongest signals and its offline evaluation.",
];

const PIPELINE = [
  {
    title: "Data",
    body: "Offer catalog, shopper profiles and a log of impressions, clicks and redemptions. A click or redemption counts as a positive.",
  },
  {
    title: "Candidates & features",
    body: "Business rules filter the 120 offers down to the eligible ones for this region and device. Features describe the shopper, the offer and the context, using only events before the request.",
  },
  {
    title: "Ranker",
    body: "A scikit-learn logistic-regression model scores every eligible offer; the top 10 are returned with readable reasons.",
  },
  {
    title: "API",
    body: "A FastAPI service serves recommendations, shopper search, feedback events, model info and on-demand retraining.",
  },
  {
    title: "UI",
    body: "A Next.js app with TanStack Query. Every context value is part of the query key, so each change fetches the right list.",
  },
];

// Exact values from backend/artifacts/metrics.json and metadata.json (time-aware holdout, K = 10).
const STATS = [
  { value: "0.6356", label: "Recall@10, contextual ranker", note: "Share of held-out clicks/redeems found in the Top-10" },
  { value: "0.0948", label: "Recall@10, popularity baseline", note: "The “just show bestsellers” approach" },
  { value: "≈6.7×", label: "Better than popularity", note: "0.6356 ÷ 0.0948 on Recall@10" },
  { value: "0.3145", label: "NDCG@10, contextual ranker", note: "Rewards putting the right offer near the top" },
  { value: "92.5%", label: "Catalog coverage", note: "Share of the 120 offers that appear in some Top-10" },
  { value: "601", label: "Test requests", note: "Clicks/redeems in the latest 20% of events" },
];

const HIGHLIGHTS = [
  "Inspected the data before modelling: there was no “accepted” label, so the label rule (click or redeem = positive) is explicit and documented.",
  "Point-in-time features: training, evaluation and the API share one feature builder that only counts earlier events, so there is no training/serving skew.",
  "Evaluated honestly on a time-based holdout. The finding: context features add no measurable lift on this dataset (Recall@10 0.6356 with context vs 0.6589 without). This is reported, not hidden.",
  "Context still matters through candidate rules: changing region or device changes which offers are eligible, and the UI shows that funnel step by step.",
  "Measured a real product trade-off: capping 3 offers per category raises diversity from 0.17 to 0.42 but drops Recall@10 to 0.2596.",
];

const STACK_GROUPS = [
  { title: "Machine learning", items: ["Python", "pandas", "NumPy", "scikit-learn", "joblib", "Jupyter"] },
  { title: "Backend", items: ["FastAPI", "Pydantic", "Uvicorn", "pytest"] },
  { title: "Frontend", items: ["Next.js 16", "React 19", "TypeScript", "TanStack Query"] },
  { title: "Cloud (AWS design)", items: ["SageMaker", "Lambda", "API Gateway", "Kinesis", "S3", "Glue / Athena"] },
];

const RUN_COMMANDS = `cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.train                          # train + evaluate
uvicorn api.main:app --reload --port 8003    # API

cd ../frontend
npm install && cp .env.example .env.local
npm run dev                                  # http://localhost:3003`;

/** Static "Project details" page for portfolio visitors. No API calls, so it renders without the backend. */
export default function AboutPage() {
  return (
    <div className="page about-page">
      <header className="site-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">%</span>
          <div>
            <h1>ShopSmart Offers</h1>
            <p className="muted small">Project details</p>
          </div>
        </div>
        <Link href="/" className="details-link">
          ← Back to the app
        </Link>
      </header>

      <section className="panel about-hero">
        <h2 className="about-title">Context-Aware Offer Recommender</h2>
        <p className="about-lead">
          Picks the 10 discount offers a shopper is most likely to use <em>right now</em>, based on their
          history, the offer itself and the current context: device, region, hour and day.
        </p>
        <ul className="about-badges" aria-label="Tech stack">
          {STACK_BADGES.map((badge) => (
            <li key={badge} className="strategy-badge">
              {badge}
            </li>
          ))}
        </ul>
      </section>

      <section className="about-section">
        <h2 className="section-title">The problem</h2>
        <div className="panel">
          <p>
            Retailers run many offers at once, but showing every shopper the same promotions wastes screen space
            and discount budget. The same shopper may also want different offers on a phone in the evening than on
            a laptop at lunchtime, and some offers are only valid in certain regions or channels. The goal is to
            rank only the offers a shopper can actually use, in the order they are most likely to engage with.
          </p>
        </div>
      </section>

      <section className="about-section">
        <h2 className="section-title">What it does</h2>
        <ul className="panel about-list">
          {FEATURES.map((feature) => (
            <li key={feature}>{feature}</li>
          ))}
        </ul>
      </section>

      <section className="about-section">
        <h2 className="section-title">How it works</h2>
        <ol className="about-steps">
          {PIPELINE.map((step, index) => (
            <li key={step.title} className="panel">
              <span className="about-step-number" aria-hidden="true">
                {index + 1}
              </span>
              <div>
                <h3>{step.title}</h3>
                <p className="muted small">{step.body}</p>
              </div>
            </li>
          ))}
        </ol>
        <p className="muted small about-note">
          <strong>On AWS:</strong> a SageMaker training job and endpoint run the same <code>src/</code> code that is
          tested locally. API Gateway calls a Lambda that queries the endpoint, and feedback goes through a second
          Lambda into Kinesis, then S3 and Glue/Athena for retraining. The browser never holds AWS credentials.
        </p>
      </section>

      <section className="about-section">
        <h2 className="section-title">Results</h2>
        <div className="about-stats">
          {STATS.map((stat) => (
            <div key={stat.label} className="panel about-stat">
              <strong>{stat.value}</strong>
              <span>{stat.label}</span>
              <span className="muted tiny">{stat.note}</span>
            </div>
          ))}
        </div>
        <p className="muted small about-note">
          Time-aware holdout: trained on the earliest 80% of events, then all 120 offers were ranked for each of the
          601 clicks/redeems in the latest 20%. Data: Project 2’s 1,000 shoppers, plus 120 synthetic offers and 6,961
          interactions generated to match them, so these numbers show the method works, not real business impact.
        </p>
      </section>

      <section className="about-section">
        <h2 className="section-title">Engineering highlights</h2>
        <ul className="panel about-list">
          {HIGHLIGHTS.map((highlight) => (
            <li key={highlight}>{highlight}</li>
          ))}
        </ul>
      </section>

      <section className="about-section">
        <h2 className="section-title">Tech stack</h2>
        <div className="about-stack">
          {STACK_GROUPS.map((group) => (
            <div key={group.title} className="panel">
              <h3 className="section-title">{group.title}</h3>
              <ul className="reasons">
                {group.items.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <section className="about-section">
        <h2 className="section-title">Run it locally</h2>
        <pre className="panel about-code">
          <code>{RUN_COMMANDS}</code>
        </pre>
      </section>
    </div>
  );
}
