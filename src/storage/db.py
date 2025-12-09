# FILE: src/storage/db.py
from pathlib import Path
import sqlite3
from typing import Optional, Dict, Any

DB_PATH = Path("artifacts/events.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    ip TEXT,
    ua TEXT,
    method TEXT,
    path TEXT,
    status INTEGER,
    size INTEGER,
    content_score REAL,
    content_thr REAL,
    session_score REAL,
    session_thr REAL,
    fused_score REAL,
    fused_thr REAL,
    is_anomaly INTEGER,
    raw_line TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_utc);
CREATE INDEX IF NOT EXISTS idx_events_anom ON events(is_anomaly, ts_utc);
CREATE INDEX IF NOT EXISTS idx_events_ip ON events(ip, ts_utc);
"""

def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()

def insert_event(row: Dict[str, Any]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO events (
                ts_utc, ip, ua, method, path, status, size,
                content_score, content_thr, session_score, session_thr,
                fused_score, fused_thr, is_anomaly, raw_line
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            row.get("ts_utc"),
            row.get("ip"),
            row.get("ua"),
            row.get("method"),
            row.get("path"),
            row.get("status"),
            row.get("size"),
            row.get("content_score"),
            row.get("content_thr"),
            row.get("session_score"),
            row.get("session_thr"),
            row.get("fused_score"),
            row.get("fused_thr"),
            1 if row.get("is_anomaly") else 0,
            row.get("raw_line"),
        ))
        conn.commit()

def recent_events(limit: int = 100):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("""
            SELECT ts_utc, ip, method, path, status, size,
                   content_score, session_score, fused_score,
                   is_anomaly
            FROM events
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        return rows

def stats_counts():
    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        anoms = conn.execute("SELECT COUNT(*) FROM events WHERE is_anomaly=1").fetchone()[0]
        return {"total": total, "anomalies": anoms}
