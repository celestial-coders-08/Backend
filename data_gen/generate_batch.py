"""
Synthetic data generator.
Usage: python generate_batch.py --n 50 --seed 42 --hard_case_ratio 0.25
Outputs: data/bank_transactions.csv, data/ledger_entries.csv, data/ground_truth.json
"""
import argparse
import csv
import json
import os
import random
from datetime import date, timedelta

COMPANIES = [
    "Acme Corp", "GlobalTech Ltd", "Sunrise Ventures", "BlueStar Inc",
    "Quantum Dynamics", "PeakFlow Solutions", "Meridian Trading", "NexGen Systems",
    "Coastal Partners", "Summit Analytics", "Vortex Holdings", "Ironclad Finance",
    "Silverline Media", "Prism Consultants", "Cascade Industries",
]

DESCRIPTIONS = [
    "Invoice payment", "Service fee", "Software subscription", "Consulting retainer",
    "Hardware purchase", "Maintenance contract", "License renewal", "Vendor payment",
    "Contractor payment", "Cloud services", "Marketing services", "Legal fees",
    "Audit services", "Insurance premium", "Utility payment",
]


def random_date(base: date, spread: int = 30) -> date:
    return base + timedelta(days=random.randint(0, spread))


def generate_batch(n: int = 50, seed: int = 42, hard_case_ratio: float = 0.25, out_dir: str = "data"):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    base_date = date(2026, 6, 1)
    n_hard = max(1, int(n * hard_case_ratio))
    n_clean = n - n_hard

    bank_rows = []
    ledger_rows = []
    ground_truth = []

    bank_counter = 1
    ledger_counter = 1

    # ── Clean 1:1 matches ─────────────────────────────────────────────────
    for _ in range(n_clean):
        bid = f"B{bank_counter:03d}"
        lid = f"L{ledger_counter:03d}"
        company = random.choice(COMPANIES)
        desc = random.choice(DESCRIPTIONS)
        amount = round(random.uniform(50, 5000), 2)
        d = random_date(base_date).strftime("%Y-%m-%d")

        bank_rows.append({
            "bank_id": bid, "company": company, "description": desc,
            "amount": amount, "date": d, "reference": f"REF-{bid}",
        })
        ledger_rows.append({
            "ledger_id": lid, "company": company, "description": desc,
            "amount": amount, "date": d, "reference": f"REF-{bid}",
        })
        ground_truth.append({"bank_id": bid, "ledger_id": lid, "type": "clean"})
        bank_counter += 1
        ledger_counter += 1

    # ── Hard cases ─────────────────────────────────────────────────────────
    hard_categories = ["late_settlement", "fee_deduction", "split_payment",
                       "garbled_description", "true_orphan", "duplicate"]
    category_cycle = (hard_categories * ((n_hard // len(hard_categories)) + 1))[:n_hard]
    random.shuffle(category_cycle)

    for cat in category_cycle:
        company = random.choice(COMPANIES)
        desc = random.choice(DESCRIPTIONS)
        amount = round(random.uniform(100, 3000), 2)
        d = random_date(base_date)
        bid = f"B{bank_counter:03d}"
        lid = f"L{ledger_counter:03d}"

        if cat == "late_settlement":
            days_drift = random.randint(3, 7)
            ledger_date = (d - timedelta(days=days_drift)).strftime("%Y-%m-%d")
            bank_rows.append({"bank_id": bid, "company": company, "description": desc,
                               "amount": amount, "date": d.strftime("%Y-%m-%d"), "reference": f"REF-{bid}"})
            ledger_rows.append({"ledger_id": lid, "company": company, "description": desc,
                                 "amount": amount, "date": ledger_date, "reference": f"REF-{bid}"})
            ground_truth.append({"bank_id": bid, "ledger_id": lid, "type": "late_settlement"})
            bank_counter += 1; ledger_counter += 1

        elif cat == "fee_deduction":
            fee = round(random.uniform(1.5, 8.0), 2)
            ledger_amount = round(amount + fee, 2)
            bank_rows.append({"bank_id": bid, "company": company, "description": desc,
                               "amount": amount, "date": d.strftime("%Y-%m-%d"), "reference": f"REF-{bid}"})
            ledger_rows.append({"ledger_id": lid, "company": company, "description": desc,
                                 "amount": ledger_amount, "date": d.strftime("%Y-%m-%d"), "reference": f"REF-{bid}"})
            ground_truth.append({"bank_id": bid, "ledger_id": lid, "type": "fee_deduction"})
            bank_counter += 1; ledger_counter += 1

        elif cat == "split_payment":
            bid2 = f"B{bank_counter + 1:03d}"
            half1 = round(amount / 2, 2)
            half2 = round(amount - half1, 2)
            bank_rows.append({"bank_id": bid, "company": company, "description": desc + " (pt1)",
                               "amount": half1, "date": d.strftime("%Y-%m-%d"), "reference": f"REF-{lid}"})
            bank_rows.append({"bank_id": bid2, "company": company, "description": desc + " (pt2)",
                               "amount": half2, "date": d.strftime("%Y-%m-%d"), "reference": f"REF-{lid}"})
            ledger_rows.append({"ledger_id": lid, "company": company, "description": desc,
                                 "amount": amount, "date": d.strftime("%Y-%m-%d"), "reference": f"REF-{lid}"})
            ground_truth.append({"bank_id": bid, "ledger_id": lid, "type": "split_payment",
                                  "group": [bid, bid2]})
            bank_counter += 2; ledger_counter += 1

        elif cat == "garbled_description":
            garbled = desc[:max(4, len(desc) - 5)].upper() + "..."
            bank_rows.append({"bank_id": bid, "company": company, "description": garbled,
                               "amount": amount, "date": d.strftime("%Y-%m-%d"), "reference": f"REF-{bid}"})
            ledger_rows.append({"ledger_id": lid, "company": company, "description": desc,
                                 "amount": amount, "date": d.strftime("%Y-%m-%d"), "reference": f"REF-{bid}"})
            ground_truth.append({"bank_id": bid, "ledger_id": lid, "type": "garbled_description"})
            bank_counter += 1; ledger_counter += 1

        elif cat == "true_orphan":
            bank_rows.append({"bank_id": bid, "company": company, "description": desc,
                               "amount": amount, "date": d.strftime("%Y-%m-%d"), "reference": f"REF-{bid}"})
            ground_truth.append({"bank_id": bid, "ledger_id": None, "type": "true_orphan"})
            bank_counter += 1

        elif cat == "duplicate":
            d2 = (d + timedelta(days=1)).strftime("%Y-%m-%d")
            bid2 = f"B{bank_counter + 1:03d}"
            bank_rows.append({"bank_id": bid, "company": company, "description": desc,
                               "amount": amount, "date": d.strftime("%Y-%m-%d"), "reference": f"REF-{bid}"})
            bank_rows.append({"bank_id": bid2, "company": company, "description": desc,
                               "amount": amount, "date": d2, "reference": f"REF-{bid}"})
            ledger_rows.append({"ledger_id": lid, "company": company, "description": desc,
                                 "amount": amount, "date": d.strftime("%Y-%m-%d"), "reference": f"REF-{bid}"})
            ground_truth.append({"bank_id": bid, "ledger_id": lid, "type": "duplicate"})
            ground_truth.append({"bank_id": bid2, "ledger_id": None, "type": "true_orphan"})
            bank_counter += 2; ledger_counter += 1

    # ── Write files ────────────────────────────────────────────────────────
    def write_csv(rows, filepath, fieldnames):
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(bank_rows, os.path.join(out_dir, "bank_transactions.csv"),
              ["bank_id", "company", "description", "amount", "date", "reference"])
    write_csv(ledger_rows, os.path.join(out_dir, "ledger_entries.csv"),
              ["ledger_id", "company", "description", "amount", "date", "reference"])

    gt_payload = {"batch_id": f"seed-{seed}", "mappings": ground_truth}
    with open(os.path.join(out_dir, "ground_truth.json"), "w") as f:
        json.dump(gt_payload, f, indent=2)

    print(f"Generated {len(bank_rows)} bank rows, {len(ledger_rows)} ledger rows → {out_dir}/")
    return gt_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard_case_ratio", type=float, default=0.25)
    args = parser.parse_args()
    generate_batch(args.n, args.seed, args.hard_case_ratio)
