"""Match one ledger entry to exactly two bank settlements by summed amount."""
from itertools import combinations

from rapidfuzz import fuzz

from .fuzzy_match import _parse_date, _safe_float


def _amount_tolerance(amount, cfg):
    return max(
        abs(amount) * cfg.get("amount_tolerance_pct", 0.03),
        cfg.get("amount_tolerance_abs", 0.0),
    )


def _similar_counterparty(bank_row, ledger_row):
    bank_company = str(bank_row.get("company", ""))
    ledger_company = str(ledger_row.get("company", ""))
    return fuzz.token_set_ratio(bank_company, ledger_company) >= 70


def run_split_match(bank_df, ledger_df, cfg):
    """Return split matches, split exceptions, and IDs consumed by this stage."""
    matches = []
    exceptions = []
    consumed_bank_ids = set()
    consumed_ledger_ids = set()
    available_banks = [row for row in bank_df.to_dict("records")]

    for ledger_row in ledger_df.to_dict("records"):
        ledger_id = ledger_row["ledger_id"]
        ledger_amount = _safe_float(ledger_row.get("amount"))
        ledger_date = _parse_date(ledger_row.get("date"))
        date_window = cfg.get("date_window_days", 5)
        candidates = [
            row for row in available_banks
            if row["bank_id"] not in consumed_bank_ids
            and abs((_parse_date(row.get("date")) - ledger_date).days) <= date_window
            and _similar_counterparty(row, ledger_row)
        ]

        valid = []
        tolerance = _amount_tolerance(ledger_amount, cfg)
        candidates = [
            row for row in candidates
            if abs(_safe_float(row.get("amount"))) <= abs(ledger_amount) + tolerance
        ]
        for first, second in combinations(candidates, 2):
            total = _safe_float(first.get("amount")) + _safe_float(second.get("amount"))
            difference = abs(total - ledger_amount)
            if difference <= tolerance:
                valid.append({
                    "bank_ids": [first["bank_id"], second["bank_id"]],
                    "ledger_id": ledger_id,
                    "bank_amounts": [_safe_float(first.get("amount")), _safe_float(second.get("amount"))],
                    "sum": round(total, 2),
                    "difference": round(difference, 2),
                })

        if len(valid) == 1:
            candidate = valid[0]
            for bank_id in candidate["bank_ids"]:
                matches.append({
                    "bank_id": bank_id,
                    "ledger_id": ledger_id,
                    "tier": "fuzzy_split_payment",
                    "confidence": 1.0,
                    "rationale": (
                        f"Two settlements ${candidate['bank_amounts'][0]:.2f} + "
                        f"${candidate['bank_amounts'][1]:.2f} = ${candidate['sum']:.2f}; "
                        f"ledger amount ${ledger_amount:.2f}, difference ${candidate['difference']:.2f}."
                    ),
                    "resolved": True,
                })
                consumed_bank_ids.add(bank_id)
            consumed_ledger_ids.add(ledger_id)
        elif len(valid) > 1:
            candidate_ids = {bank_id for candidate in valid for bank_id in candidate["bank_ids"]}
            exceptions.append({
                "bank_id": sorted(candidate_ids)[0],
                "bank_ids": sorted(candidate_ids),
                "reason_code": "AMBIGUOUS_SPLIT_CANDIDATES",
                "explanation": (
                    f"Found {len(valid)} valid two-record combinations for ledger {ledger_id} "
                    f"amount ${ledger_amount:.2f} within ${tolerance:.2f} tolerance; no combination selected."
                ),
                "candidates": valid,
            })
            consumed_bank_ids.update(candidate_ids)
            consumed_ledger_ids.add(ledger_id)

    return matches, exceptions, consumed_bank_ids, consumed_ledger_ids