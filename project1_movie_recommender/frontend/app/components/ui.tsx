import type { ReactNode } from "react";

/** Horizontal bar showing `score` relative to `maxScore`. */
export function ScoreBar({
  score,
  maxScore,
  label,
}: {
  score: number;
  maxScore: number;
  label: string;
}) {
  // Scale to the top result; a 4% minimum keeps tiny scores visible. Guard against divide-by-zero.
  const fillPercent = maxScore > 0 ? Math.max(4, (score / maxScore) * 100) : 0;
  return (
    <div className="score">
      <div
        className="score-track"
        role="meter"
        aria-label="Relevance score"
        aria-valuemin={0}
        aria-valuemax={maxScore}
        aria-valuenow={score}
      >
        <div className="score-fill" style={{ width: `${fillPercent}%` }} />
      </div>
      <span className="score-label">{label}</span>
    </div>
  );
}

/** Renders a pipe-separated genre string ("Drama|Comedy") as individual chips. */
export function GenreChips({ genres }: { genres: string }) {
  const genreList = genres.split("|").filter(Boolean);
  return (
    <div className="chips">
      {genreList.map((genreName) => (
        <span key={genreName} className="chip">
          {genreName}
        </span>
      ))}
    </div>
  );
}

/** Error or empty-state box; errors use role="alert" so screen readers announce them. */
export function StatusMessage({
  tone,
  title,
  children,
}: {
  tone: "error" | "empty";
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className={`status status-${tone}`} role={tone === "error" ? "alert" : "status"}>
      <strong>{title}</strong>
      {children && <p>{children}</p>}
    </div>
  );
}

/** Grey placeholder cards shown while recommendations load. */
export function SkeletonCards({ cardCount }: { cardCount: number }) {
  return (
    <div className="card-grid" aria-busy="true" aria-label="Loading recommendations">
      {Array.from({ length: cardCount }, (_, cardIndex) => (
        <div key={cardIndex} className="card skeleton">
          <div className="skeleton-line wide" />
          <div className="skeleton-line" />
          <div className="skeleton-line short" />
        </div>
      ))}
    </div>
  );
}

/** "Show 5 / 10 / 20" dropdown controlling how many recommendations are requested. */
export function LimitSelect({
  resultLimit,
  onResultLimitChange,
}: {
  resultLimit: number;
  onResultLimitChange: (nextLimit: number) => void;
}) {
  return (
    <label className="limit-select">
      Show
      <select
        value={resultLimit}
        onChange={(changeEvent) => onResultLimitChange(Number(changeEvent.target.value))}
      >
        {[5, 10, 20].map((limitOption) => (
          <option key={limitOption} value={limitOption}>
            {limitOption}
          </option>
        ))}
      </select>
    </label>
  );
}
