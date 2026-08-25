"""Pipeline orchestrator — runs all 3 layers in sequence."""
import json
import os
import pandas as pd
from .exact_match import run_exact_match
from .fuzzy_match import run_fuzzy_match
from .split_match import run_split_match
from .ai_adjudicate import run_ai_adjudication


def _load_thresholds():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "config", "thresholds.json")
    with open(path) as f:
        return json.load(f)


def run_pipeline(bank_df: pd.DataFrame, ledger_df: pd.DataFrame,
                 few_shot_examples: list = None, progress_callback=None):
    """
    Run the full 3-layer pipeline.
    progress_callback(stage: str, detail: str) is called for SSE streaming.
    Returns: {matches, exceptions, stats}
    """
    cfg = _load_thresholds()
    if few_shot_examples is None:
        few_shot_examples = []

    def emit(stage, detail):
        if progress_callback:
            progress_callback(stage, detail)

    # ── Layer 1: Exact match ──────────────────────────────────────────────
    emit("exact_match", f"Scanning {len(bank_df)} bank records vs {len(ledger_df)} ledger entries…")
    exact_matches, unmatched_bank, unmatched_ledger = run_exact_match(bank_df, ledger_df)
    emit("exact_match_done", f"{len(exact_matches)} matched by exact key — {len(unmatched_bank)} remaining")

    # ── Layer 2: Fuzzy match ──────────────────────────────────────────────
    emit("fuzzy_match", f"Running fuzzy match on {len(unmatched_bank)} unresolved records…")
    fuzzy_matches, ai_candidates, fuzzy_exceptions, unmatched_bank2, unmatched_ledger2 = run_fuzzy_match(
        unmatched_bank, unmatched_ledger, cfg
    )
    emit("fuzzy_match_done",
         f"{len(fuzzy_matches)} matched by fuzzy — {len(ai_candidates)} sent to AI adjudication")

    # ── Layer 2.5: Split-payment matching ─────────────────────────────────
    emit("split_match", f"Checking {len(unmatched_bank2)} unmatched bank records for split payments…")
    split_matches, split_exceptions, split_bank_ids, split_ledger_ids = run_split_match(
        unmatched_bank2, unmatched_ledger2, cfg
    )
    ai_candidates = [c for c in ai_candidates if c["bank_row"]["bank_id"] not in split_bank_ids]
    fuzzy_exceptions = [e for e in fuzzy_exceptions if e["bank_id"] not in split_bank_ids]
    emit("split_match_done", f"{len(split_matches)} records resolved by split matching")

    # ── Layer 3: AI adjudication ──────────────────────────────────────────
    ai_matches, ai_exceptions = [], []
    if ai_candidates:
        emit("ai_adjudication", f"Sending {len(ai_candidates)} ambiguous pair(s) to Groq AI…")
        ai_matches, ai_exceptions = run_ai_adjudication(ai_candidates, few_shot_examples)
        emit("ai_adjudication_done",
             f"{len(ai_matches)} matched by AI — {len(ai_exceptions)} unresolved")
    else:
        emit("ai_adjudication_skip", "No ambiguous pairs — skipping AI layer")

    adjudicated_ids = {
        record.get("bank_id")
        for record in ai_matches + ai_exceptions
        if record.get("bank_id")
    }
    for candidate in ai_candidates:
        bank_row = candidate["bank_row"]
        if bank_row["bank_id"] not in adjudicated_ids:
            ai_exceptions.append({
                "bank_id": bank_row["bank_id"],
                "amount": float(bank_row.get("amount", 0)),
                "date": bank_row.get("date"),
                "reason_code": "AI_UNRESOLVED",
                "explanation": (
                    f"AI_UNRESOLVED: bank {bank_row['bank_id']} amount ${float(bank_row.get('amount', 0)):.2f} "
                    f"on {bank_row.get('date')} did not receive an adjudication result."
                ),
                "candidates": [candidate["ledger_row"].get("ledger_id")],
            })

    # ── Combine ───────────────────────────────────────────────────────────
    all_matches = exact_matches + fuzzy_matches + split_matches + ai_matches
    all_exceptions = fuzzy_exceptions + split_exceptions + ai_exceptions

    total = len(bank_df)
    resolved = len([m for m in all_matches if m["resolved"]])
    match_rate = round(resolved / total * 100, 1) if total else 0

    emit("scoring", f"Scoring complete — {match_rate}% match rate ({resolved}/{total})")

    exception_record_ids = {
        bank_id
        for exception in all_exceptions
        for bank_id in exception.get("bank_ids", [exception.get("bank_id")])
        if bank_id
    }
    tier_counts = {
        "exact": len(exact_matches),
        "fuzzy": len(fuzzy_matches),
        "fuzzy_split_payment": len(split_matches),
        "ai_adjudicated": len(ai_matches),
        "exception": len(exception_record_ids),
    }
    counted_records = sum(tier_counts.values())
    assert counted_records == total, (
        f"Pipeline accounting mismatch: tiers={counted_records}, total_records={total}"
    )

    return {
        "matches": all_matches,
        "exceptions": all_exceptions,
        "stats": {
            "total_bank_records": total,
            "total_records": total,
            "total_ledger_records": len(ledger_df),
            "resolved": resolved,
            "match_rate_pct": match_rate,
            "tier_counts": tier_counts,
            "config_version": cfg.get("version", 1),
        },
    }
