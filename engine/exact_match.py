"""Layer 1: Exact match on (amount, date, reference)."""
import pandas as pd


def run_exact_match(bank_df: pd.DataFrame, ledger_df: pd.DataFrame):
    """
    Returns:
        matches  – list of {bank_id, ledger_id, tier, confidence, rationale, resolved}
        unmatched_bank – DataFrame of unmatched bank rows
        unmatched_ledger – DataFrame of unmatched ledger rows
    """
    matches = []
    matched_bank_ids = set()
    matched_ledger_ids = set()

    # Build a lookup dict from ledger: key = (amount, date, reference) → ledger_id
    ledger_records = ledger_df.to_dict("records")
    ledger_lookup = {}
    for lrow in ledger_records:
        key = (float(lrow["amount"]), str(lrow["date"]), str(lrow["reference"]))
        ledger_lookup[key] = lrow["ledger_id"]

    bank_records = bank_df.to_dict("records")
    for brow in bank_records:
        key = (float(brow["amount"]), str(brow["date"]), str(brow["reference"]))
        if key in ledger_lookup:
            lid = ledger_lookup[key]
            matches.append({
                "bank_id": brow["bank_id"],
                "ledger_id": lid,
                "tier": "exact",
                "confidence": 1.0,
                "rationale": None,
                "resolved": True,
            })
            matched_bank_ids.add(brow["bank_id"])
            matched_ledger_ids.add(lid)

    unmatched_bank = bank_df[~bank_df["bank_id"].isin(matched_bank_ids)].copy()
    unmatched_ledger = ledger_df[~ledger_df["ledger_id"].isin(matched_ledger_ids)].copy()
    return matches, unmatched_bank, unmatched_ledger
