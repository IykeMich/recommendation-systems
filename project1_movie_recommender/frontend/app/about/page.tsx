import type { Metadata } from "next";
import Link from "next/link";

/** Page <title> and meta description for the recruiter-facing project summary. */
export const metadata: Metadata = {
  title: "Project details · RecLab Movie Recommender",
  description:
    "Plain-language overview of the RecLab movie recommender: the problem, what it does, how it works, results and tech stack.",
};

const TECH_BADGES = ["Python", "scikit-learn", "FastAPI", "Next.js", "TypeScript", "AWS"];

const FEATURES = [
  "Search an 800-movie catalog by title, genre or ID and pick any movie as a starting point.",
  "See the most similar movies, ranked, with a similarity score bar on every card.",
  "Click any recommendation to explore from that movie next, browsing the catalog by similarity.",
  "Describe a movie that doesn't exist yet (title, genres, plot) and instantly get the closest catalog matches: the cold-start case.",
  "Switch to the Retail tab to pick a shopper, see their profile and get personalised product picks, with a popularity fallback for brand-new visitors.",
  "Send “view” and “add to cart” events from the UI, the same contract a production event pipeline would use.",
];

const PIPELINE = [
  {
    title: "Data",
    body: "A movie catalog (genres + plot description) and a retail interaction log, stored as CSV files.",
  },
  {
    title: "Model",
    body: "Each movie's genres and description become a TF-IDF vector; cosine similarity between vectors ranks “movies like this one”.",
  },
  {
    title: "API",
    body: "A FastAPI service precomputes the model at start-up and serves recommendations, catalog search and events as JSON.",
  },
  {
    title: "UI",
    body: "A Next.js + TanStack Query playground calls the API, caches results and shows loading, empty and error states.",
  },
];

/** Every number below is taken from the project's own data files or API (see README.md). */
const STATS = [
  { value: "800", label: "Movies in the catalog" },
  { value: "2,826", label: "TF-IDF vocabulary terms" },
  { value: "9,058", label: "Movie interactions in the dataset" },
  { value: "500", label: "Retail products" },
  { value: "1,000", label: "Retail shoppers" },
  { value: "29,165", label: "Retail interaction events" },
  { value: "7", label: "API endpoints" },
];

const HIGHLIGHTS = [
  "Handles item cold start: a brand-new movie is projected into the existing TF-IDF space with the already-fitted vectorizer (no retraining), so it can be matched against the catalog immediately.",
  "Transparent, explainable baselines: content similarity for movies and an item-item collaborative baseline for retail, with a popularity fallback for shoppers who have no history.",
  "A stable API contract (documented in docs/api-contract.md) lets the same frontend talk to the local FastAPI app or the AWS deployment unchanged.",
  "The browser never holds AWS credentials: it calls the application API, which calls SageMaker on its behalf.",
  "Self-contained backend: all data, model code and AWS templates live inside backend/, so deploying that one folder never leaves data behind.",
];

const STACK = [
  { group: "ML", items: ["scikit-learn (TF-IDF, cosine similarity)", "pandas", "joblib model artifact"] },
  { group: "Backend", items: ["Python", "FastAPI", "Pydantic", "Uvicorn"] },
  { group: "Frontend", items: ["Next.js (App Router)", "React", "TypeScript", "TanStack Query"] },
  { group: "Cloud", items: ["AWS SAM", "API Gateway (HTTP API)", "Lambda", "SageMaker endpoint"] },
];

const RUN_LOCALLY = `# API (terminal 1)
cd project1_movie_recommender/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8001

# Frontend (terminal 2)
cd project1_movie_recommender/frontend
npm install && cp .env.example .env.local
npm run dev        # http://localhost:3001`;

/** Static "Project details" page for recruiters and hiring managers; makes no API calls. */
export default function AboutPage() {
  return (
    <div className="page about">
      <header className="site-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            R
          </span>
          <div>
            <h1>RecLab</h1>
            <p className="muted small">Project details</p>
          </div>
        </div>
      </header>

      <main className="about-sections">
        <section className="panel about-hero">
          <span className="eyebrow">Project 1 · Recommendation systems</span>
          <h2>RecLab Movie Recommender</h2>
          <p className="about-lead">
            A working “movies like this one” recommender that suggests similar films from their
            genres and plot, even for a brand-new movie nobody has watched yet.
          </p>
          <ul className="chips" aria-label="Tech stack">
            {TECH_BADGES.map((badge) => (
              <li key={badge} className="chip">
                {badge}
              </li>
            ))}
          </ul>
          <p>
            <Link href="/" className="button about-cta">
              ← Back to the app
            </Link>
          </p>
        </section>

        <section className="panel about-section">
          <h3>The problem</h3>
          <p>
            Streaming and retail catalogs are too large to browse, and people leave when they
            can’t find something they like. New titles are hardest of all: with no viewing history,
            classic “people who watched this also watched” systems can’t recommend them. This
            project recommends from what an item <em>is</em> (its genres and description), so even
            a brand-new title can be recommended on day one.
          </p>
        </section>

        <section className="panel about-section">
          <h3>What it does</h3>
          <ul className="about-list">
            {FEATURES.map((feature) => (
              <li key={feature}>{feature}</li>
            ))}
          </ul>
        </section>

        <section className="panel about-section">
          <h3>How it works</h3>
          <ol className="about-steps">
            {PIPELINE.map((step, stepIndex) => (
              <li key={step.title}>
                <span className="about-step-number" aria-hidden="true">
                  {stepIndex + 1}
                </span>
                <div>
                  <strong>{step.title}</strong>
                  <p className="muted">{step.body}</p>
                </div>
              </li>
            ))}
          </ol>
          <p className="muted small">
            <strong>AWS design:</strong> the same API contract is deployed with AWS SAM as an API
            Gateway HTTP API in front of a Lambda function, which forwards requests to a SageMaker
            endpoint hosting the trained TF-IDF model.
          </p>
        </section>

        <section className="panel about-section">
          <h3>Results &amp; key facts</h3>
          <ul className="about-stats">
            {STATS.map((stat) => (
              <li key={stat.label} className="card">
                <span className="about-stat-value">{stat.value}</span>
                <span className="muted small">{stat.label}</span>
              </li>
            ))}
          </ul>
          <p className="muted small">
            Figures come from the project’s data files. The datasets are synthetic sample data
            built for learning; the project has no offline accuracy evaluation, so no accuracy
            metrics are claimed.
          </p>
        </section>

        <section className="panel about-section">
          <h3>Engineering highlights</h3>
          <ul className="about-list">
            {HIGHLIGHTS.map((highlight) => (
              <li key={highlight}>{highlight}</li>
            ))}
          </ul>
        </section>

        <section className="panel about-section">
          <h3>Tech stack</h3>
          <div className="about-stack">
            {STACK.map((stackGroup) => (
              <div key={stackGroup.group}>
                <span className="eyebrow">{stackGroup.group}</span>
                <ul className="chips">
                  {stackGroup.items.map((item) => (
                    <li key={item} className="chip">
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        <section className="panel about-section">
          <h3>Run it locally</h3>
          <pre className="about-code">
            <code>{RUN_LOCALLY}</code>
          </pre>
        </section>
      </main>
    </div>
  );
}
