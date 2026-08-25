"""Precision/recall/F1 scoring against ground truth, outside the matching engine."""
import argparse
import json
from typing import Dict, List

CATEGORIES = [
    "clean", "late_settlement", "fee_deduction", "split_payment",
    "garbled_description", "duplicate_orphan", "true_orphan_bank",
    "true_orphan_ledger",
]


def _category(value):
    return {"true_orphan": "true_orphan_bank", "duplicate": "duplicate_orphan"}.get(value, value or "unknown")


def score_results(matches: List[Dict], exceptions: List[Dict], ground_truth: Dict) -> Dict:
    """Score every ground-truth bank mapping, including records omitted by output."""
    gt_map = {
        m["bank_id"]: {"ledger_id": m.get("ledger_id"), "type": _category(m.get("type"))}
        for m in ground_truth.get("mappings", []) if "bank_id" in m
    }
    category_stats = {cat: {"tp": 0, "fp": 0, "fn": 0} for cat in CATEGORIES}
    exception_map = {}
    for exception in exceptions:
        ids = exception.get("bank_ids", [exception.get("bank_id")])
        for bank_id in ids:
            if bank_id:
                exception_map[bank_id] = exception
    false_match_records = []
    false_exception_records = []
    true_positives = false_positives = false_negatives = 0

    for match in matches:
        bank_id = match.get("bank_id")
        gt = gt_map.get(bank_id)
        category = gt["type"] if gt else "unknown"
        category_stats.setdefault(category, {"tp": 0, "fp": 0, "fn": 0})
        if gt and gt["ledger_id"] == match.get("ledger_id"):
            true_positives += 1
            category_stats[category]["tp"] += 1
        else:
            false_positives += 1
            category_stats[category]["fp"] += 1
            false_match_records.append({"bank_id": bank_id, "predicted_ledger": match.get("ledger_id"),
                                        "expected_ledger": gt.get("ledger_id") if gt else None, "type": category})

    for bank_id, gt in gt_map.items():
        category = gt["type"]
        category_stats.setdefault(category, {"tp": 0, "fp": 0, "fn": 0})
        expected = gt["ledger_id"]
        if not any(match.get("bank_id") == bank_id for match in matches) and expected is not None:
            false_negatives += 1
            category_stats[category]["fn"] += 1
            false_exception_records.append({"bank_id": bank_id, "expected_ledger": expected, "type": category,
                                             "reason_code": exception_map.get(bank_id, {}).get("reason_code", "MISSING_OUTPUT")})

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 1.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    for stats in category_stats.values():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        stats["precision"] = round(tp / (tp + fp), 4) if tp + fp else 1.0
        stats["recall"] = round(tp / (tp + fn), 4) if tp + fn else 1.0
        stats["f1"] = round(2 * stats["precision"] * stats["recall"] /
                            (stats["precision"] + stats["recall"]), 4) if stats["precision"] + stats["recall"] else 0.0

    has_ground_truth = bool(gt_map)
    if not has_ground_truth:
        precision = recall = f1 = None

    return {
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "scoring_available": has_ground_truth,
        "true_positives": true_positives, "false_positives": false_positives,
        "false_negatives": false_negatives, "predicted_matches": len(matches),
        "ground_truth_records": len(gt_map), "category_breakdown": category_stats,
        "false_match_records": false_match_records, "false_exception_records": false_exception_records,
    }


def print_score_table(result: Dict):
    print("Category                 Precision  Recall    F1      TP  FP  FN")
    print("-" * 68)
    for category, stats in result["category_breakdown"].items():
        print(f"{category:<24} {stats['precision']:<10.4f} {stats['recall']:<8.4f} {stats['f1']:<7.4f} "
              f"{stats['tp']:>3} {stats['fp']:>3} {stats['fn']:>3}")
    print(f"Overall                  {result['precision']:<10.4f} {result['recall']:<8.4f} {result['f1']:<7.4f} "
          f"{result['true_positives']:>3} {result['false_positives']:>3} {result['false_negatives']:>3}")


def main():
    parser = argparse.ArgumentParser(description="Score saved pipeline output against ground truth")
    parser.add_argument("--results", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    with open(args.results) as f:
        output = json.load(f)
    with open(args.ground_truth) as f:
        ground_truth = json.load(f)
    result = score_results(output.get("matches", []), output.get("exceptions", []), ground_truth)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print_score_table(result)


if __name__ == "__main__":
    main()
