from __future__ import annotations

from pathlib import Path
import json
import sqlite3
from typing import Any, Iterable


def db_path(path: str | Path | None = None) -> Path:
    p = Path(path) if path else Path("data/derivatives_engine.sqlite")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str | Path | None = None) -> Path:
    p = db_path(path)
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    with connect(p) as conn:
        conn.executescript(schema)
    return p


def upsert_ohlcv(rows: Iterable[dict[str, Any]], path: str | Path | None = None) -> int:
    rows = list(rows)
    if not rows:
        return 0
    with connect(path) as conn:
        conn.executemany(
            """
            INSERT INTO ohlcv(symbol, source, date, open, high, low, close, adj_close, volume)
            VALUES(:symbol, :source, :date, :open, :high, :low, :close, :adj_close, :volume)
            ON CONFLICT(symbol, source, date) DO UPDATE SET
              open=excluded.open,
              high=excluded.high,
              low=excluded.low,
              close=excluded.close,
              adj_close=excluded.adj_close,
              volume=excluded.volume
            """,
            rows,
        )
    return len(rows)


def insert_market_snapshot(
    symbol: str,
    source: str,
    observed_at: str,
    price: float,
    currency: str | None = None,
    raw: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> int:
    with connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO market_snapshots(symbol, source, observed_at, price, currency, raw_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (symbol, source, observed_at, float(price), currency, json.dumps(raw or {})),
        )
        return int(cur.lastrowid)


def insert_rate_curve(
    currency: str,
    tenors: dict[str, float],
    source: str,
    observed_at: str,
    path: str | Path | None = None,
) -> int:
    rows = [
        {"currency": currency, "tenor": tenor, "rate": float(rate), "source": source, "observed_at": observed_at}
        for tenor, rate in tenors.items()
    ]
    if not rows:
        return 0
    with connect(path) as conn:
        conn.executemany(
            """
            INSERT INTO rates(currency, tenor, rate, source, observed_at)
            VALUES(:currency, :tenor, :rate, :source, :observed_at)
            """,
            rows,
        )
    return len(rows)


def insert_option_chain(
    symbol: str,
    source: str,
    expiry: str,
    raw: dict[str, Any],
    path: str | Path | None = None,
) -> int:
    with connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO option_chains(symbol, source, expiry, raw_json)
            VALUES (?, ?, ?, ?)
            """,
            (symbol, source, expiry, json.dumps(raw)),
        )
        return int(cur.lastrowid)


def insert_pricing_run(product_type: str, request: dict[str, Any], response: dict[str, Any], path: str | Path | None = None) -> int:
    with connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO pricing_runs(product_type, request_json, response_json)
            VALUES (?, ?, ?)
            """,
            (product_type, json.dumps(request), json.dumps(response)),
        )
        return int(cur.lastrowid)


def insert_workflow(name: str, product_type: str, workflow: dict[str, Any], path: str | Path | None = None) -> int:
    with connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO workflows(name, product_type, workflow_json)
            VALUES (?, ?, ?)
            """,
            (name, product_type, json.dumps(workflow)),
        )
        return int(cur.lastrowid)


def get_workflow(workflow_id: int, path: str | Path | None = None) -> dict[str, Any] | None:
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["workflow"] = json.loads(data.pop("workflow_json"))
    return data


def latest_ohlcv(symbol: str, source: str | None = None, limit: int = 5, path: str | Path | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM ohlcv WHERE symbol = ?"
    params: list[Any] = [symbol]
    if source:
        sql += " AND source = ?"
        params.append(source)
    sql += " ORDER BY date DESC LIMIT ?"
    params.append(limit)
    with connect(path) as conn:
        return [dict(r) for r in conn.execute(sql, params)]
