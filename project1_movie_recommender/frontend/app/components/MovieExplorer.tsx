"use client";

import { useState, type FormEvent } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import {
  api,
  type Movie,
  type MovieQuery,
  type MovieRecommendation,
} from "../lib/api";
import { useDebouncedValue } from "../lib/useDebouncedValue";
import {
  GenreChips,
  LimitSelect,
  ScoreBar,
  SkeletonCards,
  StatusMessage,
} from "./ui";

/**
 * The movie recommendations are based on. Only id/title/genres are required, because a
 * clicked recommendation (MovieRecommendation) lacks description/year/language.
 */
type SelectedMovie = Pick<Movie, "item_id" | "title" | "genres"> &
  Partial<Pick<Movie, "description" | "year" | "language">>;

/**
 * What recommendations are based on: a catalog movie, or a new movie the user described
 * (not in the catalog, so it is sent as text and projected into the TF-IDF space).
 */
type RecommendationSource =
  | { kind: "catalog"; movie: SelectedMovie }
  | { kind: "described"; query: MovieQuery };

/** Movie selected on first load so the page shows recommendations immediately. */
const INITIAL_MOVIE: SelectedMovie = {
  item_id: "M00001",
  title: "The Shawshank Redemption",
  genres: "Drama",
};

/** Sidebar modes: browse the catalog, or describe a new movie in a form. */
const SIDEBAR_MODES = [
  { modeId: "catalog", label: "Catalog" },
  { modeId: "describe", label: "Describe new" },
] as const;

type SidebarModeId = (typeof SIDEBAR_MODES)[number]["modeId"];

const EMPTY_QUERY: MovieQuery = { title: "", genres: "", description: "" };

/**
 * Movies tab: search the catalog (or describe a new movie), and see content-based "similar movies".
 */
export function MovieExplorer() {
  const [sidebarMode, setSidebarMode] = useState<SidebarModeId>("catalog");
  const [searchTerm, setSearchTerm] = useState("");
  const [draftQuery, setDraftQuery] = useState<MovieQuery>(EMPTY_QUERY);
  const [source, setSource] = useState<RecommendationSource>({
    kind: "catalog",
    movie: INITIAL_MOVIE,
  });
  const [resultLimit, setResultLimit] = useState(10);
  // Debounce typing so we don't send a search request on every keystroke.
  const debouncedSearchTerm = useDebouncedValue(searchTerm.trim());

  // Catalog search, cached per search term. keepPreviousData keeps old results on screen while
  // the next term loads (avoids flicker). An empty term returns the first movies in the catalog.
  const movieSearchQuery = useQuery({
    queryKey: ["movies", debouncedSearchTerm],
    queryFn: () => api.searchMovies(debouncedSearchTerm),
    placeholderData: keepPreviousData,
  });

  // Content-based recs; the key includes the source (movie ID or described text) + limit, so changing
  // either fetches (or reuses cache). A described movie is POSTed as text instead of looked up by ID.
  const recommendationsQuery = useQuery({
    queryKey:
      source.kind === "catalog"
        ? ["content-recommendations", source.movie.item_id, resultLimit]
        : ["content-query-recommendations", source.query, resultLimit],
    // Both responses share `recommendations`, which is all the result list needs.
    queryFn: (): Promise<{ recommendations: MovieRecommendation[] }> =>
      source.kind === "catalog"
        ? api.contentRecommendations(source.movie.item_id, resultLimit)
        : api.queryContentRecommendations(source.query, resultLimit),
  });

  // Results arrive sorted by similarity, so the first score is the max used to scale the score bars.
  const recommendations = recommendationsQuery.data?.recommendations ?? [];
  const topScore = recommendations[0]?.score ?? 1;

  // Clicking a recommendation makes it the new source movie ("keep exploring") and scrolls to the top.
  function selectRecommendation(recommendation: MovieRecommendation) {
    selectCatalogMovie(recommendation);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function selectCatalogMovie(movie: SelectedMovie) {
    setSource({ kind: "catalog", movie });
  }

  // Submitting the form makes the described movie the source. Genres are typed comma-separated
  // and normalised to the catalog's pipe-separated format ("Drama, Comedy" -> "Drama|Comedy").
  function submitDescribedMovie(submitEvent: FormEvent<HTMLFormElement>) {
    submitEvent.preventDefault();
    const genres = draftQuery.genres
      .split(/[,|]/)
      .map((genreName) => genreName.trim())
      .filter(Boolean)
      .join("|");
    setSource({
      kind: "described",
      query: {
        title: draftQuery.title.trim(),
        genres,
        description: draftQuery.description.trim(),
      },
    });
  }

  const canSubmitQuery = Boolean(draftQuery.genres.trim() || draftQuery.description.trim());
  const selectedItemId = source.kind === "catalog" ? source.movie.item_id : null;

  return (
    <div className="explorer">
      {/* Sidebar: search box, match count and clickable search results */}
      <aside className="panel sidebar">
        <div className="quick-picks" role="group" aria-label="Recommendation source">
          {SIDEBAR_MODES.map((sidebarModeOption) => (
            <button
              key={sidebarModeOption.modeId}
              type="button"
              className="pill"
              aria-pressed={sidebarMode === sidebarModeOption.modeId}
              onClick={() => setSidebarMode(sidebarModeOption.modeId)}
            >
              {sidebarModeOption.label}
            </button>
          ))}
        </div>

        {sidebarMode === "describe" ? (
          // New-movie form: the movie is not added to the catalog, only used as a query.
          <form className="query-form" onSubmit={submitDescribedMovie}>
            <p className="muted small">
              Describe a movie that isn&apos;t in the catalog and get similar catalog movies.
            </p>
            <label className="field-label" htmlFor="query-title">
              Title <span className="muted small">(optional)</span>
            </label>
            <input
              id="query-title"
              className="text-input"
              placeholder="e.g. Starfall"
              value={draftQuery.title}
              onChange={(changeEvent) =>
                setDraftQuery({ ...draftQuery, title: changeEvent.target.value })
              }
              autoComplete="off"
            />
            <label className="field-label" htmlFor="query-genres">
              Genres
            </label>
            <input
              id="query-genres"
              className="text-input"
              placeholder="e.g. Sci-Fi, Thriller"
              value={draftQuery.genres}
              onChange={(changeEvent) =>
                setDraftQuery({ ...draftQuery, genres: changeEvent.target.value })
              }
              autoComplete="off"
            />
            <label className="field-label" htmlFor="query-description">
              Description
            </label>
            <textarea
              id="query-description"
              className="text-input"
              placeholder="What is the movie about?"
              value={draftQuery.description}
              onChange={(changeEvent) =>
                setDraftQuery({ ...draftQuery, description: changeEvent.target.value })
              }
            />
            <button type="submit" className="button" disabled={!canSubmitQuery}>
              Get recommendations
            </button>
          </form>
        ) : (
          <>
          <label className="field-label" htmlFor="movie-search">
            Find a movie
          </label>
          <input
            id="movie-search"
            className="text-input"
            type="search"
            placeholder="Title, genre or ID (e.g. M00042)"
            value={searchTerm}
            onChange={(changeEvent) => setSearchTerm(changeEvent.target.value)}
            autoComplete="off"
          />

          <div className="search-meta">
            {movieSearchQuery.data &&
              `${movieSearchQuery.data.total} match${movieSearchQuery.data.total === 1 ? "" : "es"}`}
            {movieSearchQuery.isFetching && <span className="spinner" aria-label="Searching" />}
          </div>

          {movieSearchQuery.isError && (
            <StatusMessage tone="error" title="Search unavailable">
              {movieSearchQuery.error.message}
            </StatusMessage>
          )}

          {movieSearchQuery.data?.movies.length === 0 && (
            <StatusMessage tone="empty" title="No movies found">
              Try a genre like “Drama” or a title word like “River”.
            </StatusMessage>
          )}

          <ul className="option-list">
            {movieSearchQuery.data?.movies.map((movie) => (
              <li key={movie.item_id}>
                <button
                  type="button"
                  className="option"
                  aria-pressed={movie.item_id === selectedItemId}
                  onClick={() => selectCatalogMovie(movie)}
                >
                  <span className="option-title">{movie.title}</span>
                  <span className="option-meta">
                    {movie.item_id} · {movie.year} · {movie.genres.replaceAll("|", ", ")}
                  </span>
                </button>
              </li>
            ))}
          </ul>
          </>
        )}
      </aside>

      <section className="results">
        {/* Source movie card ("Because you watched" / "Because you described") */}
        {source.kind === "catalog" ? (
          <div className="panel source-card">
            <span className="eyebrow">Because you watched</span>
            <h2>{source.movie.title}</h2>
            <GenreChips genres={source.movie.genres} />
            {source.movie.description && (
              <p className="muted">{source.movie.description}</p>
            )}
            <p className="meta-row">
              <code>{source.movie.item_id}</code>
              {source.movie.year && <span>{source.movie.year}</span>}
              {source.movie.language && <span>{source.movie.language}</span>}
            </p>
          </div>
        ) : (
          <div className="panel source-card">
            <span className="eyebrow">Because you described</span>
            <h2>{source.query.title || "Untitled movie"}</h2>
            <GenreChips genres={source.query.genres} />
            {source.query.description && (
              <p className="muted">{source.query.description}</p>
            )}
            <p className="meta-row">
              <span className="badge">New — not in catalog</span>
            </p>
          </div>
        )}

        <div className="results-header">
          <div>
            <h3>Similar movies</h3>
            <p className="muted small">
              {source.kind === "catalog"
                ? "TF-IDF over genres and description, ranked by cosine similarity."
                : "Your text projected into the catalog's TF-IDF space, ranked by cosine similarity."}
            </p>
          </div>
          <LimitSelect resultLimit={resultLimit} onResultLimitChange={setResultLimit} />
        </div>

        {/* Recommendation results: loading skeleton, error, or ranked cards */}
        {recommendationsQuery.isPending && <SkeletonCards cardCount={6} />}

        {recommendationsQuery.isError && (
          <StatusMessage tone="error" title="Couldn't load recommendations">
            {recommendationsQuery.error.message}
          </StatusMessage>
        )}

        {recommendationsQuery.isSuccess && (
          <ol className="card-grid">
            {recommendations.map((recommendation, rankIndex) => (
              <li key={recommendation.item_id}>
                <button
                  type="button"
                  className="card card-button"
                  onClick={() => selectRecommendation(recommendation)}
                  title="Explore recommendations for this movie"
                >
                  <span className="rank">#{rankIndex + 1}</span>
                  <span className="card-title">{recommendation.title}</span>
                  <GenreChips genres={recommendation.genres} />
                  <ScoreBar
                    score={recommendation.score}
                    maxScore={topScore}
                    label={`${Math.round(recommendation.score * 100)}% similar`}
                  />
                </button>
              </li>
            ))}
          </ol>
        )}
      </section>
    </div>
  );
}
