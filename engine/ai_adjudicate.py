"""Layer 3: AI adjudication using Groq API with structured JSON output."""
import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        _client = Groq(api_key=api_key)
    return _client


def _build_prompt(bank_row: dict, ledger_row: dict, few_shot_examples: list) -> str:
    few_shot_text = ""
    for ex in few_shot_examples[:3]:  # cap at 3 examples
        few_shot_text += f"""
Example:
Bank: {json.dumps(ex.get('bank_row', {}))}
Ledger: {json.dumps(ex.get('ledger_row', {}))}
Decision: {json.dumps({"match": ex.get("label") == "match", "confidence": ex.get("confidence", 0.7), "rationale": ex.get("rationale", "")})}
"""

    return f"""You are a financial reconciliation assistant. Your job is to decide whether a bank transaction and a ledger entry refer to the same underlying financial event.

{few_shot_text}

Now decide for this pair:
Bank transaction: {json.dumps(bank_row)}
Ledger entry: {json.dumps(ledger_row)}

Respond ONLY with valid JSON in exactly this schema:
{{
  "match": true | false,
  "confidence": 0.0 to 1.0,
  "rationale": "one concise sentence explaining your decision"
}}

Rules:
- Small amount differences (<5%) are often fee deductions — lean toward match if other fields align
- Date differences of 3–7 days are common settlement delays — lean toward match if amount/company align
- Truncated/garbled descriptions that share the same root word count as matching
- Set confidence < 0.6 if you are genuinely uncertain; this will route to human review
- NEVER fabricate a match you are not reasonably confident about"""


def adjudicate_pair(bank_row: dict, ledger_row: dict, few_shot_examples: list = None) -> dict:
    """
    Returns dict with keys: match (bool), confidence (float), rationale (str)
    On any parse error, returns {"match": False, "confidence": 0.0, "rationale": "LLM_PARSE_ERROR", "error": True}
    """
    if few_shot_examples is None:
        few_shot_examples = []

    prompt = _build_prompt(bank_row, ledger_row, few_shot_examples)
    client = _get_client()

    CANDIDATE_MODELS = ["groq/compound-mini", "groq/compound", "qwen/qwen3.6-27b", "openai/gpt-oss-20b"]
    last_err = None

    parse_retry_used = False
    for model_name in CANDIDATE_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()

            # Parse structured JSON
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError(f"No valid JSON object found in response: {raw[:80]}")

            result = json.loads(raw[start:end])

            if not isinstance(result.get("match"), bool):
                raise ValueError("'match' field must be boolean")
            if not isinstance(result.get("confidence"), (int, float)):
                raise ValueError("'confidence' field must be numeric")

            return {
                "match": bool(result["match"]),
                "confidence": float(result["confidence"]),
                "rationale": str(result.get("rationale", "")),
                "error": False,
            }

        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            if parse_retry_used:
                break
            parse_retry_used = True
            continue
        except Exception as e:
            last_err = e
            continue

    return {
        "match": False,
        "confidence": 0.0,
        "rationale": f"LLM_PARSE_ERROR: {str(last_err)[:100]}",
        "error": True,
    }


from concurrent.futures import ThreadPoolExecutor


def run_ai_adjudication(ai_candidates: list, few_shot_examples: list = None):
    """
    Process a list of {bank_row, ledger_row, confidence} candidates.
    Returns:
        ai_matches – list of match records
        ai_exceptions – list of exception records
    """
    if few_shot_examples is None:
        few_shot_examples = []

    if not ai_candidates:
        return [], []

    def _process_candidate(candidate):
        bank_row = candidate["bank_row"]
        ledger_row = candidate["ledger_row"]
        result = adjudicate_pair(bank_row, ledger_row, few_shot_examples)
        return bank_row, ledger_row, result

    workers = min(8, len(ai_candidates))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        adjudicated_results = list(executor.map(_process_candidate, ai_candidates))

    ai_matches = []
    ai_exceptions = []

    for bank_row, ledger_row, result in adjudicated_results:
        context = (
            f"Bank {bank_row.get('bank_id')} ${float(bank_row.get('amount', 0)):.2f} "
            f"on {bank_row.get('date')} vs ledger {ledger_row.get('ledger_id')} "
            f"${float(ledger_row.get('amount', 0)):.2f} on {ledger_row.get('date')}."
        )
        if result["error"]:
            ai_exceptions.append({
                "bank_id": bank_row["bank_id"],
                "amount": float(bank_row.get("amount", 0)),
                "date": bank_row.get("date"),
                "reason_code": "LLM_PARSE_ERROR",
                "explanation": f"{context} {result['rationale']}",
                "candidates": [ledger_row.get("ledger_id")],
            })
        elif result["match"] and result["confidence"] >= 0.5:
            ai_matches.append({
                "bank_id": bank_row["bank_id"],
                "ledger_id": ledger_row["ledger_id"],
                "tier": "ai_adjudicated",
                "confidence": result["confidence"],
                "rationale": result["rationale"],
                "resolved": True,
            })
        else:
            reason = "LOW_CONFIDENCE_LLM" if not result["error"] else "LLM_PARSE_ERROR"
            ai_exceptions.append({
                "bank_id": bank_row["bank_id"],
                "amount": float(bank_row.get("amount", 0)),
                "date": bank_row.get("date"),
                "reason_code": reason,
                "explanation": f"{context} {result['rationale']}",
                "candidates": [ledger_row.get("ledger_id")],
            })

    return ai_matches, ai_exceptions
