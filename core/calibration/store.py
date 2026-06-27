"""
calibration/store.py
SQLite persistence for calibration state with versioned history.

Every calibration is appended as a new row (never overwritten), so you get a
full audit trail of how parameters drift over time. The latest calibration per
(ticker, model) is retrieved for warm-starting the next run.

Schema
------
calibrations(
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT,
    model         TEXT,
    timestamp     REAL,
    spot          REAL,
    r             REAL,
    params_json   TEXT,
    rmse          REAL,
    max_error     REAL,
    n_points      INTEGER,
    arb_free      INTEGER,
    warm_started  INTEGER,
    elapsed_sec   REAL
)
"""

import json
import sqlite3
import time
from pathlib import Path

from .calibrator import CalibrationResult


_SCHEMA = """
CREATE TABLE IF NOT EXISTS calibrations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    timestamp     REAL    NOT NULL,
    spot          REAL,
    r             REAL,
    params_json   TEXT    NOT NULL,
    rmse          REAL,
    max_error     REAL,
    n_points      INTEGER,
    arb_free      INTEGER,
    warm_started  INTEGER,
    elapsed_sec   REAL
);
CREATE INDEX IF NOT EXISTS idx_ticker_model
    ON calibrations(ticker, model, timestamp DESC);
"""


class CalibrationStore:
    """SQLite-backed versioned store for calibration results."""

    def __init__(self, db_path: str = "calibrations.db"):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)

    # ── write ─────────────────────────────────────────────────────────────────

    def save(self, ticker: str, result: CalibrationResult,
             spot: float = None, r: float = None) -> int:
        """Append a calibration result. Returns the new row id."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO calibrations
                   (ticker, model, timestamp, spot, r, params_json,
                    rmse, max_error, n_points, arb_free, warm_started, elapsed_sec)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ticker, result.model, result.timestamp, spot, r,
                 json.dumps(result.params), result.rmse, result.max_error,
                 result.n_points, int(result.arb_free),
                 int(result.warm_started), result.elapsed_sec),
            )
            return cur.lastrowid

    # ── read ──────────────────────────────────────────────────────────────────

    def latest_params(self, ticker: str, model: str) -> dict | None:
        """Most recent fitted parameters for (ticker, model), or None."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT params_json FROM calibrations
                   WHERE ticker=? AND model=?
                   ORDER BY timestamp DESC LIMIT 1""",
                (ticker, model),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def history(self, ticker: str, model: str, limit: int = 100) -> list[dict]:
        """
        Full calibration history for (ticker, model), newest first.
        Each entry includes params + quality metrics + timestamp.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM calibrations
                   WHERE ticker=? AND model=?
                   ORDER BY timestamp DESC LIMIT ?""",
                (ticker, model, limit),
            ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["params"] = json.loads(d.pop("params_json"))
            d["arb_free"] = bool(d["arb_free"])
            d["warm_started"] = bool(d["warm_started"])
            out.append(d)
        return out

    def param_timeseries(self, ticker: str, model: str,
                         param_name: str) -> tuple[list[float], list[float]]:
        """
        Time series of one parameter for drift analysis.
        Returns (timestamps, values) in chronological order.
        """
        hist = self.history(ticker, model, limit=10_000)
        hist = sorted(hist, key=lambda h: h["timestamp"])
        ts   = [h["timestamp"] for h in hist]
        vals = [h["params"].get(param_name) for h in hist]
        return ts, vals

    def tickers(self) -> list[str]:
        """All distinct tickers in the store."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM calibrations ORDER BY ticker"
            ).fetchall()
        return [r[0] for r in rows]

    def clear(self, ticker: str = None):
        """Delete history (all, or for one ticker). Use with care."""
        with sqlite3.connect(self.db_path) as conn:
            if ticker is None:
                conn.execute("DELETE FROM calibrations")
            else:
                conn.execute("DELETE FROM calibrations WHERE ticker=?", (ticker,))
