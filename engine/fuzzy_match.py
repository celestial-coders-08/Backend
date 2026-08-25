"""Layer 2: Fuzzy match using amount tolerance, date window, and description similarity."""
import json
import math
import os
from collections import defaultdict
from datetime import datetime, date

import pandas as pd
from rapidfuzz import fuzz


def _load_thresholds():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "config", "thresholds.json")
    with open(path) as f:
        return json.load(f)


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _parse_date(d_val) -> date:
    d_str = str(d_val) if d_val is not None else "1970-01-01"
    try:
        parts = d_str.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return date(1970, 1, 1)


def _prepare_record(row: dict) -> dict:
    d_str = str(row.get("date", "1970-01-01"))
    return {
        "raw": row,
        "amount": _safe_float(row.get("amount", 0)),
        "date_str": d_str,
        "date_obj": _parse_date(d_str),
        "desc": str(row.get("description", "")),
        "company_lower": str(row.get("company", "")).lower(),
    }


def _date_diff(d1: str, d2: str) -> int:
    return abs((_parse_date(d1) - _parse_date(d2)).days)


def _confidence_score_prep(b_row: dict, l_row: dict, cfg) -> float:
    amount_tol = cfg["amount_tolerance_pct"]
    amount_tol_abs = cfg.get("amount_tolerance_abs", 0.0)
    date_win = cfg["date_window_days"]

    b_amt = b_row["amount"]
    l_amt = l_row["amount"]

    # Amount score
    if b_amt == 0 and l_amt == 0:
        amt_score = 1.0
    else:
        denom = max(abs(b_amt), abs(l_amt), 1.0)
        difference = abs(b_amt - l_amt)
        allowed_difference = max(denom * amount_tol, amount_tol_abs)
        amt_score = max(0.0, 1.0 - difference / max(allowed_difference, 1e-6)) if difference <= allowed_difference else 0.0

    # Date score
    day_diff = abs((b_row["date_obj"] - l_row["date_obj"]).days)
    date_score = max(0.0, 1.0 - day_diff / date_win) if day_diff <= date_win else 0.0

    # Description similarity
    desc_score = fuzz.token_sort_ratio(b_row["desc"], l_row["desc"]) / 100.0

    # Company match bonus
    b_co = b_row["company_lower"]
    l_co = l_row["company_lower"]
    co_score = 1.0 if b_co == l_co else fuzz.ratio(b_co, l_co) / 100.0

    # Weighted composite
    score = 0.35 * amt_score + 0.30 * date_score + 0.20 * desc_score + 0.15 * co_score
    return round(score, 4)


def _confidence_score(bank_row, ledger_row, cfg) -> float:
    """0–1 confidence that bank_row and ledger_row are the same transaction."""
    b_prep = _prepare_record(bank_row if isinstance(bank_row, dict) else bank_row.to_dict())
    l_prep = _prepare_record(ledger_row if isinstance(ledger_row, dict) else ledger_row.to_dict())
    return _confidence_score_prep(b_prep, l_prep, cfg)


def _build_ledger_index(ledger_preps, cfg):
    if len(ledger_preps) <= 2000:
        return {"all": ledger_preps, "by_bucket": None}

    amount_values = [abs(r["amount"]) for r in ledger_preps]
    median_amount = sorted(amount_values)[len(amount_values) // 2] if amount_values else 1.0
    bucket_size = max(10.0, median_amount * max(cfg.get("amount_tolerance_pct", 0.03) * 2.5, 0.05))

    by_bucket = defaultdict(list)
    for row in ledger_preps:
        amt = row["amount"]
        bucket = round(amt / bucket_size) * bucket_size
        by_bucket[bucket].append(row)

    return {"all": ledger_preps, "by_bucket": by_bucket, "bucket_size": bucket_size}


def _candidate_ledger_rows_prep(bank_prep, ledger_index, cfg):
    if ledger_index["by_bucket"] is None:
        return ledger_index["all"]

    target_amt = bank_prep["amount"]
    target_date_obj = bank_prep["date_obj"]
    amount_tol = max(
        1.0,
        abs(target_amt) * cfg.get("amount_tolerance_pct", 0.03) * 2.5,
        cfg.get("amount_tolerance_abs", 0.0) * 2.5,
    )
    bucket_size = ledger_index["bucket_size"]
    lower_bucket = round((target_amt - amount_tol) / bucket_size) * bucket_size
    upper_bucket = round((target_amt + amount_tol) / bucket_size) * bucket_size

    candidates = []
    for bucket in range(int(lower_bucket), int(upper_bucket) + 1, max(1, int(bucket_size))):
        candidates.extend(ledger_index["by_bucket"].get(float(bucket), []))

    if not candidates:
        candidates = ledger_index["all"]

    date_win_limit = cfg.get("date_window_days", 5) + 2
    filtered = []
    for row in candidates:
        if abs((target_date_obj - row["date_obj"]).days) <= date_win_limit:
            filtered.append(row)

    return filtered[:80]


def run_fuzzy_match(bank_df: pd.DataFrame, ledger_df: pd.DataFrame, cfg=None):
    """
    Returns:
        auto_matches  – confidence >= high threshold → matched
        ai_candidates – list of {bank_row, ledger_row, confidence} for AI adjudication
        exceptions    – confidence < low threshold → NO_CANDIDATE exception
        unmatched_bank – still unresolved bank rows
        unmatched_ledger – still unresolved ledger rows
    """
    if cfg is None:
        cfg = _load_thresholds()

    high = cfg["confidence_auto_match"]
    low = cfg["confidence_ai_band_low"]

    auto_matches = []
    ai_candidates = []
    exceptions = []
    matched_bank_ids = set()
    matched_ledger_ids = set()

    bank_records = bank_df.to_dict("records")
    ledger_records = ledger_df.to_dict("records")

    bank_preps = [_prepare_record(r) for r in bank_records]
    ledger_preps = [_prepare_record(r) for r in ledger_records]

    ledger_index = _build_ledger_index(ledger_preps, cfg)

    for b_prep in bank_preps:
        brow = b_prep["raw"]
        bank_id = brow["bank_id"]
        best_score = -1
        best_lprep = None

        for l_prep in _candidate_ledger_rows_prep(b_prep, ledger_index, cfg):
            lrow = l_prep["raw"]
            ledger_id = lrow["ledger_id"]
            if ledger_id in matched_ledger_ids:
                continue

            score = _confidence_score_prep(b_prep, l_prep, cfg)
            if score > best_score:
                best_score = score
                best_lprep = l_prep

        if best_score >= high and best_lprep is not None:
            best_lrow = best_lprep["raw"]
            auto_matches.append({
                "bank_id": bank_id,
                "ledger_id": best_lrow["ledger_id"],
                "tier": "fuzzy",
                "confidence": best_score,
                "rationale": None,
                "resolved": True,
            })
            matched_bank_ids.add(bank_id)
            matched_ledger_ids.add(best_lrow["ledger_id"])
        elif best_score >= low and best_lprep is not None:
            best_lrow = best_lprep["raw"]
            ai_candidates.append({
                "bank_row": brow,
                "ledger_row": best_lrow,
                "confidence": best_score,
            })
        else:
            exceptions.append({
                "bank_id": bank_id,
                "amount": float(brow["amount"]),
                "date": brow.get("date"),
                "reason_code": "NO_CANDIDATE",
                "explanation": (
                    f"NO_CANDIDATE: no ledger entry within {cfg.get('date_window_days', 5)} days "
                    f"for ${float(brow['amount']):.2f} payment from {brow.get('company', 'unknown')}; "
                    f"fee tolerance is ${cfg.get('amount_tolerance_abs', 0.0):.2f}."
                ),
                "candidates": [],
            })

    unmatched_bank = bank_df[~bank_df["bank_id"].isin(matched_bank_ids)].copy()
    unmatched_ledger = ledger_df[~ledger_df["ledger_id"].isin(matched_ledger_ids)].copy()
    return auto_matches, ai_candidates, exceptions, unmatched_bank, unmatched_ledger

