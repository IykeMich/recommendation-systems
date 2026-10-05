"""SQLite decision store.

transaction_id is the idempotency key: a retried transaction returns the stored
decision instead of creating a second alert (guide section 18). Locally this
plays the role DynamoDB (conditional put) plays in the AWS design.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

# decisions: one row per transaction_id (the PRIMARY KEY enforces idempotency); JSON columns are
# stored as text. The (customer_id, transaction_timestamp) index serves the velocity query.
# rejections: one row per malformed /score request, for the data-quality panel.
SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    transaction_id TEXT PRIMARY KEY,
    payload_hash TEXT NOT NULL,
    payload TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    amount REAL NOT NULL,
    risk_score REAL NOT NULL,
    decision TEXT NOT NULL,
    reasons TEXT NOT NULL,
    context TEXT NOT NULL,
    features TEXT NOT NULL,
    model_version TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    transaction_timestamp TEXT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    synthetic_label INTEGER,
    outcome TEXT,
    outcome_note TEXT,
    outcome_at TEXT
);
CREATE INDEX IF NOT EXISTS decisions_customer ON decisions (customer_id, transaction_timestamp);
CREATE INDEX IF NOT EXISTS decisions_created ON decisions (created_at);
CREATE TABLE IF NOT EXISTS rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at TEXT NOT NULL,
    errors TEXT NOT NULL
);
"""

# Columns holding lists/dicts, serialised with json.dumps on write and parsed on read.
JSON_COLUMNS = ("payload", "reasons", "context", "features")


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string (seconds precision); ISO strings sort chronologically."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DuplicateTransaction(Exception):
    """Same transaction_id, different payload: a conflict, not a retry."""


class DecisionStore:
    """Thin wrapper over a SQLite file: decisions, analyst outcomes, rejections and stats."""
    def __init__(self, path):
        """Create the parent folder and the tables/indexes if they do not exist yet."""
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        """Open a fresh connection per operation; commit on success and always close.

        If the body raises, commit is skipped, so a failed write is not saved.
        """
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _to_dict(row) -> dict:
        """Turn a sqlite3.Row into a dict and decode its JSON columns."""
        record = dict(row)
        for column in JSON_COLUMNS:
            record[column] = json.loads(record[column])
        return record

    def get(self, transaction_id: str):
        """Return the stored decision for transaction_id, or None."""
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM decisions WHERE transaction_id = ?", (transaction_id,)).fetchone()
        return self._to_dict(row) if row else None

    def insert(self, record: dict) -> tuple:
        """Insert a decision; returns (stored record, was_replay)."""
        values = {**record, **{column: json.dumps(record[column]) for column in JSON_COLUMNS}}
        try:
            with self.connect() as connection:
                connection.execute(
                    f"INSERT INTO decisions ({', '.join(values)}) VALUES ({', '.join('?' for _ in values)})",
                    tuple(values.values()),
                )
            return record, False
        except sqlite3.IntegrityError:
            # PRIMARY KEY clash: another request stored this transaction_id first (e.g. a concurrent
            # retry that passed the API's earlier get()). Same payload hash -> replay of the stored
            # decision; different payload -> a genuine conflict.
            existing = self.get(record["transaction_id"])
            if existing["payload_hash"] != record["payload_hash"]:
                raise DuplicateTransaction(record["transaction_id"])
            return existing, True

    def recent_customer_count(self, customer_id: str, since_iso: str, until_iso: str) -> int:
        """Count this customer's stored decisions with since_iso <= timestamp < until_iso.

        Used for server-side velocity. Comparing ISO strings works because all are UTC in the
        same format; rows stored without a timestamp (NULL) are never counted.
        """
        with self.connect() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM decisions WHERE customer_id = ? AND transaction_timestamp >= ? AND transaction_timestamp < ?",
                (customer_id, since_iso, until_iso),
            ).fetchone()[0]

    def list(self, decision=None, limit: int = 50) -> list:
        """Newest decisions first, optionally filtered by decision type; rowid breaks created_at ties."""
        query, parameters = "SELECT * FROM decisions", []
        if decision:
            query += " WHERE decision = ?"
            parameters.append(decision)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        parameters.append(limit)
        with self.connect() as connection:
            return [self._to_dict(row) for row in connection.execute(query, parameters).fetchall()]

    def set_outcome(self, transaction_id: str, outcome: str, note):
        """Record an analyst outcome; returns the updated record, or None if the ID is unknown."""
        with self.connect() as connection:
            updated = connection.execute(
                "UPDATE decisions SET outcome = ?, outcome_note = ?, outcome_at = ? WHERE transaction_id = ?",
                (outcome, note, utc_now(), transaction_id),
            ).rowcount
        return self.get(transaction_id) if updated else None

    def record_rejection(self, errors: list) -> None:
        """Store the validation errors of one rejected request."""
        with self.connect() as connection:
            connection.execute("INSERT INTO rejections (received_at, errors) VALUES (?, ?)", (utc_now(), json.dumps(errors)))

    def stats(self) -> dict:
        """Dashboard totals computed in SQL plus rejection counts."""
        with self.connect() as connection:
            # In SQLite a comparison is 1 or 0, so SUM(condition) counts matching rows. SUM over an
            # empty table is NULL, hence the `or 0` below. synthetic_label is only set for replayed
            # test rows, giving a rough "flagged vs labelled fraud" check.
            totals = connection.execute("""
                SELECT COUNT(*) AS scored,
                       SUM(decision = 'REVIEW') AS review,
                       SUM(decision = 'BLOCK') AS block,
                       AVG(risk_score) AS average_risk,
                       MAX(created_at) AS latest_ingestion,
                       SUM(outcome = 'confirmed_fraud') AS confirmed_fraud,
                       SUM(outcome = 'legitimate') AS confirmed_legitimate,
                       SUM(decision != 'ALLOW' AND synthetic_label = 1) AS flagged_and_labelled_fraud,
                       SUM(decision != 'ALLOW' AND synthetic_label IS NOT NULL) AS flagged_with_label,
                       SUM(decision = 'ALLOW' AND synthetic_label = 1) AS allowed_but_labelled_fraud
                FROM decisions
            """).fetchone()
            rejections = connection.execute("SELECT COUNT(*) FROM rejections").fetchone()[0]
            latest_rejections = connection.execute(
                "SELECT received_at, errors FROM rejections ORDER BY id DESC LIMIT 5").fetchall()
        return {
            **{key: (totals[key] or 0) for key in totals.keys() if key not in ("average_risk", "latest_ingestion")},
            "average_risk": round(totals["average_risk"] or 0.0, 4),
            "latest_ingestion": totals["latest_ingestion"],
            "rejected_requests": rejections,
            "latest_rejections": [{"received_at": row["received_at"], "errors": json.loads(row["errors"])} for row in latest_rejections],
        }
