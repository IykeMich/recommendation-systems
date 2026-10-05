import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Project details · ShopSmart",
  description:
    "ShopSmart in plain language: an explainable product recommender built with Python, FastAPI, Next.js and an AWS deployment design. Problem, features, pipeline, results and tech stack.",
};

// Static page: no API calls, so it renders even when the backend is down.
// Every number below is copied from backend/artifacts/metrics.json, model_metadata.json or the README.

const HERO_BADGES = ["Python", "scikit-learn", "FastAPI", "Next.js", "TypeScript", "AWS SageMaker AI"];

const FEATURES = [
  {
    title: "Pick any shopper",
    body: "Search 1,000 synthetic shoppers or use a quick pick, including a brand-new guest with no history.",
  },
  {
    title: "See personalised picks",
    body: "Get a ranked list of products for that shopper, each tagged “Similar products” or “Shoppers like you”.",
  },
  {
    title: "Ask “Why?”",
    body: "Every card explains itself, e.g. “37 shoppers who engaged with Doritos (you bought it) also engaged with this”.",
  },
  {
    title: "Act like a shopper",
    body: "View, add to cart or buy a recommended product, or any product found with catalog search.",
  },
  {
    title: "Retrain and watch it adapt",
    body: "Press Retrain model to fold the new actions in; cards then show ▲/▼/NEW rank movement.",
  },
  {
    title: "Check the model",
    body: "A model panel shows when it was trained, the data size and its offline scores against a popularity baseline.",
  },
];

const PIPELINE = [
  {
    title: "Data",
    body: "598 real-brand products plus 27,830 synthetic views, carts and purchases, weighted 1 / 3 / 6 by intent.",
  },
  {
    title: "Model",
    body: "Two signals: product similarity (TF-IDF on category, brand and description) and “shoppers like you” (item-item cosine similarity). The blend shifts from 80% content for new shoppers to 80% collaborative once they have a longer history.",
  },
  {
    title: "API",
    body: "A FastAPI service returns the top products with a reason for each, logs feedback events and can retrain and hot-swap the model.",
  },
  {
    title: "UI",
    body: "This Next.js storefront, using TanStack Query for data fetching and caching.",
  },
];

const RESULTS = [
  { value: "0.4286", label: "Recall@10 on held-out data", note: "vs 0.0904 for a popularity baseline" },
  { value: "0.2799", label: "NDCG@10 (ranking quality)", note: "vs 0.252 for item-item only" },
  { value: "0.246", label: "Recall@10 for new shoppers", note: "first 3 products only; 0.104 item-item only" },
  { value: "99.16%", label: "Catalog coverage", note: "vs 3.01% for popularity" },
];

const HIGHLIGHTS = [
  "Explainable by design: every recommendation carries a human-readable reason, phrased honestly as “N shoppers who engaged with X also engaged with this”.",
  "Leak-free evaluation: the holdout removes every event for the hidden shopper–product pair, so a leftover view can’t give the answer away.",
  "Decisions backed by data: excluding other pack sizes of owned products was tested and rejected, because it dropped Recall@10 from 0.43 to 0.13.",
  "A closed feedback loop: impressions, clicks and purchases are logged separately, so “the system recommended it” stays distinct from “they found it themselves”.",
  "Cold start handled: unknown shoppers get a popularity fallback, and new shoppers lean on content similarity until they have enough history.",
];

const STACK = [
  { group: "ML", items: ["Python", "pandas", "NumPy", "SciPy", "scikit-learn", "Jupyter"] },
  { group: "Backend", items: ["FastAPI", "Uvicorn", "pytest"] },
  { group: "Frontend", items: ["Next.js", "React", "TypeScript", "TanStack Query"] },
  { group: "Cloud", items: ["SageMaker AI", "S3", "Lambda", "API Gateway", "Kinesis", "Vercel"] },
];

const RUN_LOCALLY = `# Backend (from backend/)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.train
uvicorn api.main:app --reload --port 8002

# Frontend (from frontend/)
npm install
cp .env.example .env.local
npm run dev        # http://localhost:3002`;

/** "Project details" page for recruiters and hiring managers: plain language first, depth second. */
export default function AboutPage() {
  return (
    <div className="page about-page">
      <header className="panel about-hero">
        <Link href="/" className="back-link small">
          ← Back to the app
        </Link>
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            S
          </span>
          <div>
            <h1>ShopSmart</h1>
            <p className="muted small">Project details</p>
          </div>
        </div>
        <p className="about-pitch">
          An online-store recommender that suggests products each shopper is likely to want, and
          explains why it picked every one.
        </p>
        <ul className="badge-row" aria-label="Tech stack">
          {HERO_BADGES.map((badge) => (
            <li key={badge} className="product-category">
              {badge}
            </li>
          ))}
        </ul>
      </header>

      <section className="panel" aria-labelledby="problem-title">
        <h2 id="problem-title" className="section-title">
          The problem
        </h2>
        <p>
          Online stores carry far more products than any shopper will scroll through, so showing
          the right few items is what turns a visit into a sale. Simple “best sellers” lists treat
          everyone the same, and black-box recommendations are hard for a business to trust.
          ShopSmart shows a personalised list for each shopper, with a plain reason for every pick.
        </p>
      </section>

      <section className="panel" aria-labelledby="features-title">
        <h2 id="features-title" className="section-title">
          What it does
        </h2>
        <ul className="about-grid">
          {FEATURES.map((feature) => (
            <li key={feature.title} className="about-card">
              <h3>{feature.title}</h3>
              <p className="muted small">{feature.body}</p>
            </li>
          ))}
        </ul>
      </section>

      <section className="panel" aria-labelledby="how-title">
        <h2 id="how-title" className="section-title">
          How it works
        </h2>
        <ol className="pipeline">
          {PIPELINE.map((step, stepIndex) => (
            <li key={step.title}>
              <span className="avatar" aria-hidden="true">
                {stepIndex + 1}
              </span>
              <div>
                <h3>{step.title}</h3>
                <p className="muted small">{step.body}</p>
              </div>
            </li>
          ))}
        </ol>
        <p className="small">
          <strong>AWS deployment design:</strong> the model trains and serves on Amazon SageMaker
          AI from an S3 bundle. API Gateway routes recommendation requests through a Lambda
          function to the SageMaker endpoint, and feedback events through a second Lambda into
          Kinesis and S3 for retraining. The browser never holds AWS credentials.
        </p>
      </section>

      <section className="panel" aria-labelledby="results-title">
        <h2 id="results-title" className="section-title">
          Results
        </h2>
        <dl className="stat-tiles">
          {RESULTS.map((result) => (
            <div key={result.label} className="stat-tile">
              <dt className="small">{result.label}</dt>
              <dd className="stat-value">{result.value}</dd>
              <dd className="muted tiny">{result.note}</dd>
            </div>
          ))}
        </dl>
        <p className="muted small">
          Offline test: each shopper’s latest cart or purchase was hidden, and the model had to
          rank it in its top 10 (973 shoppers evaluated). A Recall@10 of 0.4286 means the hidden
          product appeared in the top 10 for about 43% of shoppers. The shoppers and their
          behaviour are <strong>synthetic</strong> (generated to match a real-brand catalog of
          598 products), so real-world behaviour would be the true test.
        </p>
      </section>

      <section className="panel" aria-labelledby="highlights-title">
        <h2 id="highlights-title" className="section-title">
          Engineering highlights
        </h2>
        <ul className="about-list">
          {HIGHLIGHTS.map((highlight) => (
            <li key={highlight}>{highlight}</li>
          ))}
        </ul>
      </section>

      <section className="panel" aria-labelledby="stack-title">
        <h2 id="stack-title" className="section-title">
          Tech stack
        </h2>
        <dl className="stack-groups">
          {STACK.map((stackGroup) => (
            <div key={stackGroup.group}>
              <dt className="small">
                <strong>{stackGroup.group}</strong>
              </dt>
              <dd className="badge-row">
                {stackGroup.items.map((item) => (
                  <span key={item} className="product-category">
                    {item}
                  </span>
                ))}
              </dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="panel" aria-labelledby="run-title">
        <h2 id="run-title" className="section-title">
          Run it locally
        </h2>
        <pre className="code-block">
          <code>{RUN_LOCALLY}</code>
        </pre>
        <p className="muted small">
          API docs at <code>http://localhost:8002/docs</code>. Tests:{" "}
          <code>python -m pytest -q</code> from <code>backend/</code>.
        </p>
      </section>

      <p className="small">
        <Link href="/" className="back-link">
          ← Back to the app
        </Link>
      </p>
    </div>
  );
}
