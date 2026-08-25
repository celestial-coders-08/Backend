"""Labeled-examples store and threshold retrain logic."""
import json
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "feedback.db")
THRESHOLDS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "thresholds.json")


def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS labeled_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bank_id TEXT NOT NULL,
            ledger_id TEXT,
            label TEXT NOT NULL,
            resolved_by TEXT DEFAULT 'human',
            rationale TEXT,
            timestamp TEXT NOT NULL,
            batch_id TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT,
            config_version INTEGER,
            match_rate_pct REAL,
            precision REAL,
            recall REAL,
            f1 REAL,
            timestamp TEXT
        )
    """)
    for column in ("total_records", "match_count", "exception_count", "fuzzy_count", "ai_count"):
        try:
            conn.execute(f"ALTER TABLE run_history ADD COLUMN {column} INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS retrain_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_version INTEGER,
            before_match_rate REAL,
            after_match_rate REAL,
            before_f1 REAL,
            after_f1 REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn


def save_resolution(bank_id: str, ledger_id: str | None, label: str, rationale: str = "", batch_id: str = ""):
    """label = 'match' | 'no_match'"""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO labeled_examples (bank_id, ledger_id, label, resolved_by, rationale, timestamp, batch_id) VALUES (?,?,?,?,?,?,?)",
        (bank_id, ledger_id, label, "human", rationale, datetime.utcnow().isoformat(), batch_id)
    )
    conn.commit()
    conn.close()


def get_labeled_examples(limit: int = 50) -> list:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT bank_id, ledger_id, label, rationale FROM labeled_examples ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [{"bank_id": r[0], "ledger_id": r[1], "label": r[2], "rationale": r[3]} for r in rows]


def save_run_history(batch_id: str, match_rate: float, precision: float, recall: float, f1: float, tier_counts=None, total_records=0):
    cfg = _load_cfg()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO run_history (batch_id, config_version, match_rate_pct, precision, recall, f1, timestamp, total_records, match_count, exception_count, fuzzy_count, ai_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (batch_id, cfg.get("version", 1), match_rate, precision, recall, f1, datetime.utcnow().isoformat(),
         total_records, (tier_counts or {}).get("exact", 0) + (tier_counts or {}).get("fuzzy", 0) + (tier_counts or {}).get("fuzzy_split_payment", 0),
         (tier_counts or {}).get("exception", 0), (tier_counts or {}).get("fuzzy", 0) + (tier_counts or {}).get("fuzzy_split_payment", 0),
         (tier_counts or {}).get("ai_adjudicated", 0))
    )
    conn.commit()
    conn.close()


def get_run_history() -> list:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT batch_id, config_version, match_rate_pct, precision, recall, f1, timestamp, total_records, match_count, exception_count, fuzzy_count, ai_count FROM run_history ORDER BY id"
    ).fetchall()
    conn.close()
    return [
        {"batch_id": r[0], "config_version": r[1], "match_rate_pct": r[2],
         "precision": r[3], "recall": r[4], "f1": r[5], "timestamp": r[6],
         "total_records": r[7], "match_count": r[8], "exception_count": r[9],
         "fuzzy_count": r[10], "ai_count": r[11]}
        for r in rows
    ]


def save_retrain_history(config_version, before, after):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO retrain_history (config_version, before_match_rate, after_match_rate, before_f1, after_f1, timestamp) VALUES (?,?,?,?,?,?)",
        (config_version, before["match_rate_pct"], after["match_rate_pct"], before["f1"], after["f1"], datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_retrain_history() -> list:
    conn = _get_conn()
    rows = conn.execute("SELECT config_version, before_match_rate, after_match_rate, before_f1, after_f1, timestamp FROM retrain_history ORDER BY id").fetchall()
    conn.close()
    return [{"config_version": r[0], "before_match_rate_pct": r[1], "after_match_rate_pct": r[2],
             "before_f1": r[3], "after_f1": r[4], "timestamp": r[5]} for r in rows]


def _load_cfg() -> dict:
    with open(THRESHOLDS_PATH) as f:
        return json.load(f)


def _save_cfg(cfg: dict):
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def retrain_thresholds(min_labels: int = 5) -> dict:
    """Recompute thresholds from labeled examples. Returns new config."""
    examples = get_labeled_examples(limit=200)
    matches = [e for e in examples if e["label"] == "match"]

    if len(examples) < min_labels:
        return {"status": "insufficient_labels", "n": len(examples), "required": min_labels}

    cfg = _load_cfg()
    before = _evaluate_fixed_batch()
    cfg["version"] = cfg.get("version", 1) + 1
    cfg["updated_at"] = datetime.utcnow().isoformat()
    cfg["updated_from_n_labels"] = len(examples)

    # Simple statistical adjustment: slightly relax amount tolerance based on confirmed matches
    match_ratio = len(matches) / max(len(examples), 1)
    if match_ratio > 0.7:
        cfg["amount_tolerance_pct"] = min(0.05, cfg["amount_tolerance_pct"] * 1.1)
        cfg["date_window_days"] = min(7, cfg["date_window_days"] + 1)
    elif match_ratio < 0.4:
        cfg["amount_tolerance_pct"] = max(0.01, cfg["amount_tolerance_pct"] * 0.9)
        cfg["date_window_days"] = max(2, cfg["date_window_days"] - 1)

    # Select top few-shot examples (most recent)
    cfg["few_shot_examples"] = [e["bank_id"] for e in examples[:5]]

    _save_cfg(cfg)
    after = _evaluate_fixed_batch()
    save_retrain_history(cfg["version"], before, after)
    return {"status": "retrained", "new_version": cfg["version"], "n_labels": len(examples),
            "before": before, "after": after, "delta": {
                "match_rate_pct": round(after["match_rate_pct"] - before["match_rate_pct"], 4),
                "f1": round(after["f1"] - before["f1"], 4),
            }}


def _evaluate_fixed_batch() -> dict:
    """Score the fixed held-out batch without using its ground truth in matching."""
    batch_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test_bench")
    paths = [os.path.join(batch_dir, name) for name in
             ("bank_transactions.csv", "ledger_entries.csv", "ground_truth.json")]
    if not all(os.path.exists(path) for path in paths):
        return {"match_rate_pct": 0.0, "f1": 0.0}
    import pandas as pd
    from engine.pipeline import run_pipeline
    from scoring.scorer import score_results
    with open(paths[2]) as f:
        ground_truth = json.load(f)
    result = run_pipeline(pd.read_csv(paths[0]), pd.read_csv(paths[1]))
    scoring = score_results(result["matches"], result["exceptions"], ground_truth)
    return {"match_rate_pct": result["stats"]["match_rate_pct"], "f1": scoring["f1"]}
