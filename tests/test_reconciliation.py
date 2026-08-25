import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from engine.split_match import run_split_match
from engine.ai_adjudicate import adjudicate_pair
from engine.pipeline import run_pipeline
from scoring.scorer import score_results


CFG = {"amount_tolerance_pct": 0.03, "amount_tolerance_abs": 35.0, "date_window_days": 10}


def bank_rows(*amounts):
    return pd.DataFrame([
        {"bank_id": f"B{i}", "company": "Acme Corp", "description": "Settlement",
         "amount": amount, "date": "2026-06-05", "reference": f"R{i}"}
        for i, amount in enumerate(amounts, 1)
    ])


def ledger_row(amount):
    return pd.DataFrame([{
        "ledger_id": "L1", "company": "Acme Corp", "description": "Invoice payment",
        "amount": amount, "date": "2026-06-05", "reference": "L1",
    }])


class SplitMatchTests(unittest.TestCase):
    def test_clean_split(self):
        matches, exceptions, bank_ids, ledger_ids = run_split_match(bank_rows(40, 60), ledger_row(100), CFG)
        self.assertEqual({m["bank_id"] for m in matches}, {"B1", "B2"})
        self.assertTrue(all(m["tier"] == "fuzzy_split_payment" for m in matches))
        self.assertEqual(exceptions, [])
        self.assertEqual(bank_ids, {"B1", "B2"})
        self.assertEqual(ledger_ids, {"L1"})

    def test_ambiguous_split_lists_all_candidates(self):
        matches, exceptions, bank_ids, ledger_ids = run_split_match(bank_rows(40, 60, 50, 50), ledger_row(100), CFG)
        self.assertEqual(matches, [])
        self.assertEqual(exceptions[0]["reason_code"], "AMBIGUOUS_SPLIT_CANDIDATES")
        self.assertGreaterEqual(len(exceptions[0]["candidates"]), 2)
        self.assertEqual(bank_ids, {"B1", "B2", "B3", "B4"})
        self.assertEqual(ledger_ids, {"L1"})

    def test_no_match_falls_through(self):
        matches, exceptions, bank_ids, ledger_ids = run_split_match(bank_rows(10, 20), ledger_row(100), CFG)
        self.assertEqual(matches, [])
        self.assertEqual(exceptions, [])
        self.assertEqual(bank_ids, set())
        self.assertEqual(ledger_ids, set())

    def test_scorer_counts_unreported_expected_match(self):
        result = score_results([], [], {"mappings": [{"bank_id": "B1", "ledger_id": "L1", "type": "clean"}]})
        self.assertEqual(result["false_negatives"], 1)
        self.assertEqual(result["category_breakdown"]["clean"]["recall"], 0.0)

    def test_scorer_precision_counts_all_emitted_matches(self):
        result = score_results(
            [{"bank_id": "B1", "ledger_id": "L1"}, {"bank_id": "B1", "ledger_id": "L2"}],
            [], {"mappings": [{"bank_id": "B1", "ledger_id": "L1", "type": "clean"}]},
        )
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["precision"], 0.5)

    def test_pipeline_tier_counts_equal_total(self):
        rows = [{"bank_id": "B1", "company": "Acme", "description": "Payment", "amount": 10,
                 "date": "2026-06-01", "reference": "R1"}]
        ledger = [{"ledger_id": "L1", "company": "Acme", "description": "Payment", "amount": 10,
                   "date": "2026-06-01", "reference": "R1"}]
        result = run_pipeline(pd.DataFrame(rows), pd.DataFrame(ledger))
        self.assertEqual(sum(result["stats"]["tier_counts"].values()), result["stats"]["total_records"])

    @patch("engine.ai_adjudicate._get_client")
    def test_ai_retries_json_parse_once(self, get_client):
        responses = [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"match": true, "confidence": 0.9, "r'))]),
                     SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"match": true, "confidence": 0.9, "rationale": "Same payment."}'))])]
        create = Mock(side_effect=lambda **kwargs: responses.pop(0))
        completion = SimpleNamespace(create=create)
        get_client.return_value = SimpleNamespace(chat=SimpleNamespace(completions=completion))
        result = adjudicate_pair({"bank_id": "B1"}, {"ledger_id": "L1"})
        self.assertFalse(result["error"])
        self.assertEqual(result["rationale"], "Same payment.")
        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args.kwargs["max_tokens"], 512)
        self.assertEqual(create.call_args.kwargs["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
