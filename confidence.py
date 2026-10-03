"""
Trust layer for the Document Intelligence pipeline (Day 4).

Computes a confidence score for every extracted field from OBSERVABLE signals --
not the LLM's self-reported confidence, which isn't a calibrated probability.

Signals combined:
1. evidence_present   - did the field come with a real source-text span?
2. no_warnings        - did this field trigger a validation warning (missing/type/etc)?
3. cross_check        - for total_amount: does subtotal + tax_amount ~= total_amount?
4. source_quality     - was this document digitally extracted (reliable) or OCR'd (noisier)?

These combine into a 0-1 confidence score per field. Fields below a chosen threshold
are routed to human review; fields above are auto-accepted.

If evaluation_report.json exists (from evaluate.py), this script also CALIBRATES the
score against real correctness -- showing, at each threshold, what % of fields get
auto-accepted (coverage) and what % of those auto-accepted fields were actually wrong
(accepted-error-rate). This is the evidence a confidence score is actually meaningful,
not just a made-up number.

Run:
    python confidence.py
"""

import json
from pathlib import Path

NUMERIC_FIELDS = {"subtotal", "tax_amount", "total_amount"}

# Weights for combining signals into one score. Must sum to 1.0.
WEIGHTS = {
    "evidence_present": 0.35,
    "no_warnings": 0.30,
    "cross_check": 0.15,
    "source_quality": 0.20,
}

REVIEW_THRESHOLDS = [0.5, 0.7, 0.85]  # candidate thresholds to report coverage/error tradeoffs for


def cross_check_total(extracted_fields: dict) -> float:
    """
    Returns 1.0 if subtotal + tax_amount ~= total_amount (within 1%), 0.5 if the check
    isn't applicable (missing values to check), 0.0 if it fails outright.
    """
    def val(field):
        obj = extracted_fields.get(field, {})
        return obj.get("value") if isinstance(obj, dict) else None

    subtotal = val("subtotal")
    tax = val("tax_amount")
    total = val("total_amount")

    if subtotal is None or tax is None or total is None:
        return 0.5  # can't check -- neutral, not penalized

    try:
        expected = float(subtotal) + float(tax)
        actual = float(total)
        if actual == 0:
            return 0.5
        pct_diff = abs(expected - actual) / abs(actual)
        return 1.0 if pct_diff < 0.01 else 0.0
    except (TypeError, ValueError):
        return 0.5


def score_field(field_name: str, field_obj: dict, warnings: list, is_digital: bool,
                 cross_check_score: float) -> dict:
    """Compute the confidence score and its component signals for one field."""
    value = field_obj.get("value") if isinstance(field_obj, dict) else None
    evidence = field_obj.get("evidence") if isinstance(field_obj, dict) else None

    if value is None:
        # A confidently-null field: high confidence it's correctly absent,
        # UNLESS a warning specifically flags this field as a problem.
        field_warned = any(field_name in w for w in warnings)
        return {
            "value": None,
            "confidence": 0.1 if field_warned else 0.9,
            "signals": {"note": "field is null"},
        }

    evidence_present = 1.0 if evidence else 0.0
    field_warned = any(field_name in w for w in warnings)
    no_warnings = 0.0 if field_warned else 1.0
    source_quality = 1.0 if is_digital else 0.6  # OCR documents get a lower base signal

    # Cross-check only meaningfully applies to total_amount
    cc = cross_check_score if field_name == "total_amount" else 0.5

    score = (
        WEIGHTS["evidence_present"] * evidence_present +
        WEIGHTS["no_warnings"] * no_warnings +
        WEIGHTS["cross_check"] * cc +
        WEIGHTS["source_quality"] * source_quality
    )

    return {
        "value": value,
        "confidence": round(score, 3),
        "signals": {
            "evidence_present": evidence_present,
            "no_warnings": no_warnings,
            "cross_check": cc,
            "source_quality": source_quality,
        },
    }


def score_document(extraction_result: dict, ocr_result: dict) -> dict:
    """Score every field in one document's extraction."""
    extracted_fields = extraction_result.get("extracted_fields", {})
    warnings = extraction_result.get("warnings", [])
    is_digital = ocr_result.get("is_digital", True) if ocr_result else True

    cc_score = cross_check_total(extracted_fields)

    field_scores = {}
    for field_name, field_obj in extracted_fields.items():
        if field_name == "line_items":
            continue  # array field, scored separately if needed -- skip for now
        field_scores[field_name] = score_field(field_name, field_obj, warnings, is_digital, cc_score)

    return {
        "document_id": extraction_result.get("document_id"),
        "document_type": extraction_result.get("document_type"),
        "is_digital": is_digital,
        "field_scores": field_scores,
    }


def apply_thresholds(all_scored_docs: list, thresholds: list) -> dict:
    """
    For each threshold, compute what fraction of fields would be auto-accepted (coverage)
    and, among those, what fraction actually have low review-worthiness (just from the
    score itself -- calibration against real ground truth happens separately if available).
    """
    results = {}
    for threshold in thresholds:
        total_fields = 0
        auto_accepted = 0
        for doc in all_scored_docs:
            for field_name, fscore in doc["field_scores"].items():
                total_fields += 1
                if fscore["confidence"] >= threshold:
                    auto_accepted += 1

        results[threshold] = {
            "total_fields": total_fields,
            "auto_accepted": auto_accepted,
            "coverage": round(auto_accepted / total_fields, 3) if total_fields else None,
            "flagged_for_review": total_fields - auto_accepted,
        }
    return results


def calibrate_against_ground_truth(all_scored_docs: list, evaluation_report_path: str, thresholds: list) -> dict:
    """
    If evaluate.py's output is available, check whether high-confidence fields were
    ACTUALLY more likely to be correct. This is what makes the confidence score meaningful
    rather than decorative.
    """
    if not Path(evaluation_report_path).exists():
        return None

    with open(evaluation_report_path, "r", encoding="utf-8") as f:
        eval_report = json.load(f)

    # Build lookup: (document_id, field) -> outcome ("correct", "incorrect", "missed", "hallucination", "true_negative")
    outcome_lookup = {}
    for r in eval_report.get("raw_results", []):
        outcome_lookup[(r["document_id"], r["field"])] = r["outcome"]

    calibration = {}
    for threshold in thresholds:
        accepted_correct = 0
        accepted_wrong = 0
        rejected_correct = 0
        rejected_wrong = 0

        for doc in all_scored_docs:
            doc_id = doc["document_id"]
            for field_name, fscore in doc["field_scores"].items():
                outcome = outcome_lookup.get((doc_id, field_name))
                if outcome is None:
                    continue  # no ground truth for this doc/field
                is_wrong = outcome in ("incorrect", "missed", "hallucination")
                is_accepted = fscore["confidence"] >= threshold

                if is_accepted and not is_wrong:
                    accepted_correct += 1
                elif is_accepted and is_wrong:
                    accepted_wrong += 1
                elif not is_accepted and not is_wrong:
                    rejected_correct += 1
                else:
                    rejected_wrong += 1

        total_accepted = accepted_correct + accepted_wrong
        total_evaluated = accepted_correct + accepted_wrong + rejected_correct + rejected_wrong

        calibration[threshold] = {
            "coverage": round(total_accepted / total_evaluated, 3) if total_evaluated else None,
            "accepted_error_rate": round(accepted_wrong / total_accepted, 3) if total_accepted else None,
            "review_capture_rate": round(rejected_wrong / (accepted_wrong + rejected_wrong), 3)
                                    if (accepted_wrong + rejected_wrong) else None,
            "accepted_correct": accepted_correct,
            "accepted_wrong": accepted_wrong,
            "rejected_correct": rejected_correct,
            "rejected_wrong": rejected_wrong,
        }

    return calibration


def run_confidence_scoring(
    extractions_folder: str = "extractions",
    ocr_outputs_folder: str = "outputs",
    evaluation_report_path: str = "evaluation_report.json",
    output_path: str = "confidence_report.json",
):
    extractions_dir = Path(extractions_folder)
    ocr_dir = Path(ocr_outputs_folder)

    all_scored_docs = []
    for extraction_path in sorted(extractions_dir.glob("*.json")):
        with open(extraction_path, "r", encoding="utf-8") as f:
            extraction_result = json.load(f)

        if extraction_result.get("extraction_status") == "PARSE_FAILED":
            continue  # nothing to score if extraction itself failed

        ocr_path = ocr_dir / extraction_path.name
        ocr_result = None
        if ocr_path.exists():
            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_result = json.load(f)

        scored = score_document(extraction_result, ocr_result)
        all_scored_docs.append(scored)

    threshold_summary = apply_thresholds(all_scored_docs, REVIEW_THRESHOLDS)
    calibration = calibrate_against_ground_truth(all_scored_docs, evaluation_report_path, REVIEW_THRESHOLDS)

    report = {
        "documents_scored": len(all_scored_docs),
        "thresholds_tested": REVIEW_THRESHOLDS,
        "threshold_summary": threshold_summary,
        "calibration_against_ground_truth": calibration,
        "per_document_scores": all_scored_docs,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print readable summary
    print(f"\n{'='*60}")
    print("CONFIDENCE SCORING SUMMARY")
    print(f"{'='*60}")
    print(f"Documents scored: {len(all_scored_docs)}")

    print(f"\nTHRESHOLD ANALYSIS (score-based only):")
    for t, s in threshold_summary.items():
        print(f"  threshold={t}: coverage={s['coverage']}, "
              f"auto_accepted={s['auto_accepted']}/{s['total_fields']}")

    if calibration:
        print(f"\nCALIBRATION AGAINST GROUND TRUTH (from {evaluation_report_path}):")
        for t, c in calibration.items():
            print(f"  threshold={t}: coverage={c['coverage']}, "
                  f"accepted_error_rate={c['accepted_error_rate']}, "
                  f"review_capture_rate={c['review_capture_rate']}")
        print("\n  -> accepted_error_rate = of fields we auto-accepted, what fraction were actually wrong")
        print("  -> review_capture_rate = of all actually-wrong fields, what fraction did we correctly flag for review")
    else:
        print(f"\nNo {evaluation_report_path} found -- run evaluate.py first to see calibration against real accuracy.")

    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    run_confidence_scoring()