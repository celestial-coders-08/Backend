"""
Six categories of deliberate messiness injected into the synthetic dataset.
Each returns a (bank_row_patch, ledger_row_patch, ground_truth_type) tuple.
"""
import random
import string


def _rand_str(n=6):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def late_settlement(bank_row, ledger_row):
    """Bank date is 3-7 days after ledger date."""
    from datetime import timedelta
    days = random.randint(3, 7)
    ledger_row = ledger_row.copy()
    ledger_row["date"] = (bank_row["date"] - timedelta(days=days)).strftime("%Y-%m-%d")
    return bank_row, ledger_row, "late_settlement"


def fee_deduction(bank_row, ledger_row):
    """Bank amount is slightly less than ledger (fee taken)."""
    ledger_row = ledger_row.copy()
    ledger_row["amount"] = round(bank_row["amount"] + random.uniform(1.5, 8.0), 2)
    return bank_row, ledger_row, "fee_deduction"


def split_payment(bank_rows, ledger_row):
    """Two bank rows sum to one ledger row."""
    half = round(ledger_row["amount"] / 2, 2)
    bank_rows[0]["amount"] = half
    bank_rows[1]["amount"] = round(ledger_row["amount"] - half, 2)
    return bank_rows, ledger_row, "split_payment"


def duplicate_transaction(bank_row, ledger_row):
    """Bank row has a near-duplicate (same amount, 1 day diff)."""
    from datetime import timedelta
    dup = bank_row.copy()
    dup["bank_id"] = "B" + _rand_str(3)
    dup["date"] = (bank_row["date"] + timedelta(days=1)).strftime("%Y-%m-%d")
    return bank_row, dup, ledger_row, "duplicate"


def garbled_description(bank_row, ledger_row):
    """Description is truncated / uppercased differently."""
    bank_row = bank_row.copy()
    desc = bank_row.get("description", "")
    bank_row["description"] = desc[:max(4, len(desc) - 5)].upper() + "..."
    return bank_row, ledger_row, "garbled_description"


def true_orphan(bank_row):
    """No ledger match exists for this bank row."""
    return bank_row, None, "true_orphan"
