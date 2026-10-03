"""
Evaluation harness for the Document Intelligence pipeline.

Compares each document's extracted_fields (from extraction.py output
in extractions/*.json) against hand-labeled ground truth (ground_truth.xlsx),
computing field-level precision, recall, F1, and hallucination rate.

Run:
    python evaluate.py

Requires:
    pip install openpyxl
"""

import json
import re
import csv
from pathlib import Path
from datetime import datetime
import openpyxl

# Maps ground-truth spreadsheet column names -> schema field names used in extractions/*.json
# (only needed where they differ)
GT_COLUMN_TO_SCHEMA_FIELD = {
    "invoice_no.": "invoice_number",
}

SCALAR_FIELDS = [
    "vendor_name", "invoice_number", "invoice_date", "due_date",
    "customer_name", "subtotal", "tax_amount", "total_amount", "currency"
]

NUMERIC_FIELDS = {"subtotal", "tax_amount", "total_amount"}
DATE_FIELDS = {"invoice_date", "due_date"}


def load_ground_truth(gt_path: str) -> dict:
    """
    Load the ground truth file into {document_id: {field: value}}.
    Detects the REAL format by content, not just the file extension --
    this handles cases where a CSV was saved/renamed with an .xlsx name.
    """
    path = Path(gt_path)

    # A real xlsx file is a zip archive and starts with the bytes 'PK'.
    # If it doesn't, treat it as CSV regardless of what the extension says.
    with open(path, "rb") as f:
        header_bytes = f.read(2)
    is_real_xlsx = header_bytes == b"PK"

    if is_real_xlsx:
        return _load_ground_truth_xlsx(gt_path)
    else:
        return _load_ground_truth_csv(gt_path)


def _load_ground_truth_xlsx(xlsx_path: str) -> dict:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]

    gt = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        row_dict = dict(zip(headers, row))
        doc_id = row_dict.get("document_id")
        if not doc_id:
            continue

        fields = {}
        for col_name, value in row_dict.items():
            if col_name == "document_id":
                continue
            field_name = GT_COLUMN_TO_SCHEMA_FIELD.get(col_name, col_name)
            fields[field_name] = value
        gt[doc_id] = fields

    return gt


def _load_ground_truth_csv(csv_path: str) -> dict:
    gt = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_dict in reader:
            doc_id = row_dict.get("document_id")
            if not doc_id or not doc_id.strip():
                continue

            fields = {}
            for col_name, value in row_dict.items():
                if col_name == "document_id":
                    continue
                # CSV gives everything as strings, including blanks -- normalize blanks to None
                if value is not None and value.strip() == "":
                    value = None
                field_name = GT_COLUMN_TO_SCHEMA_FIELD.get(col_name, col_name)
                fields[field_name] = value
            gt[doc_id.strip()] = fields

    return gt


CURRENCY_ALIASES = {
    "₹": "inr", "rs": "inr", "rs.": "inr", "inr": "inr", "rupees": "inr",
    "$": "usd", "usd": "usd", "us$": "usd",
    "€": "eur", "eur": "eur",
    "£": "gbp", "gbp": "gbp",
}


def normalize_currency(val) -> str:
    """Normalize a currency symbol/code/word to a canonical lowercase code."""
    if val is None:
        return ""
    s = str(val).strip().lower()
    s = re.sub(r"[^\w₹$€£]", "", s)  # keep currency symbols, strip other punctuation
    return CURRENCY_ALIASES.get(s, s)


def normalize_string(val) -> str:
    """Normalize a string value for comparison: lowercase, strip whitespace/punctuation."""
    if val is None:
        return ""
    s = str(val).strip().lower()
    s = re.sub(r"[^\w\s]", "", s)  # strip punctuation
    s = re.sub(r"\s+", " ", s)     # collapse whitespace
    return s


def normalize_date(val) -> str:
    """Normalize a date to YYYY-MM-DD string for comparison."""
    if val is None:
        return ""
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d")
    s = str(val).strip()
    # Try common formats if it's a string
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s  # fall back to raw string if unparseable


def normalize_number(val) -> float:
    """Normalize a numeric value for comparison (strip currency symbols/commas)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return round(float(val), 2)
    s = re.sub(r"[^\d.\-]", "", str(val))
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def values_match(field: str, gt_value, pred_value) -> bool:
    """Compare a ground-truth value to a predicted value using the right normalization for its type."""
    if field in DATE_FIELDS:
        return normalize_date(gt_value) == normalize_date(pred_value) and normalize_date(gt_value) != ""
    elif field in NUMERIC_FIELDS or field == "total_amount":
        gt_num = normalize_number(gt_value)
        pred_num = normalize_number(pred_value)
        if gt_num is None or pred_num is None:
            return False
        return abs(gt_num - pred_num) < 0.01
    elif field == "currency":
        gt_norm = normalize_currency(gt_value)
        pred_norm = normalize_currency(pred_value)
        return gt_norm != "" and gt_norm == pred_norm
    else:
        gt_norm = normalize_string(gt_value)
        pred_norm = normalize_string(pred_value)
        return gt_norm != "" and gt_norm == pred_norm


def evaluate_document(doc_id: str, gt_fields: dict, extraction_result: dict) -> list:
    """
    Compare one document's ground truth against its extraction.
    Returns a list of per-field result dicts.
    """
    extracted = extraction_result.get("extracted_fields", {})
    results = []

    for field in SCALAR_FIELDS:
        gt_value = gt_fields.get(field)
        gt_is_empty = gt_value is None or (isinstance(gt_value, str) and gt_value.strip() == "")

        pred_obj = extracted.get(field, {})
        pred_value = pred_obj.get("value") if isinstance(pred_obj, dict) else None
        pred_is_null = pred_value is None

        if gt_is_empty and pred_is_null:
            outcome = "true_negative"       # correctly absent
        elif gt_is_empty and not pred_is_null:
            outcome = "hallucination"       # model invented a value that shouldn't exist
        elif not gt_is_empty and pred_is_null:
            outcome = "missed"              # model failed to extract an existing value
        elif values_match(field, gt_value, pred_value):
            outcome = "correct"
        else:
            outcome = "incorrect"           # extracted something, but it's wrong

        results.append({
            "document_id": doc_id,
            "field": field,
            "gt_value": gt_value,
            "pred_value": pred_value,
            "outcome": outcome,
        })

    return results


def compute_metrics(all_results: list, group_key=None) -> dict:
    """
    Compute precision, recall, F1, hallucination rate from a list of per-field results.
    group_key: optional function(result) -> str to compute metrics per group (e.g. by field, by doc type).
    """
    def _metrics_for(subset):
        correct = sum(1 for r in subset if r["outcome"] == "correct")
        incorrect = sum(1 for r in subset if r["outcome"] == "incorrect")
        missed = sum(1 for r in subset if r["outcome"] == "missed")
        hallucinated = sum(1 for r in subset if r["outcome"] == "hallucination")
        true_negative = sum(1 for r in subset if r["outcome"] == "true_negative")

        predicted_non_null = correct + incorrect + hallucinated
        should_exist = correct + incorrect + missed

        precision = correct / predicted_non_null if predicted_non_null > 0 else None
        recall = correct / should_exist if should_exist > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall) > 0) else None

        evaluated_opportunities = correct + incorrect + missed + hallucinated + true_negative
        hallucination_rate = hallucinated / evaluated_opportunities if evaluated_opportunities > 0 else None

        return {
            "n": len(subset),
            "correct": correct,
            "incorrect": incorrect,
            "missed": missed,
            "hallucinated": hallucinated,
            "true_negative": true_negative,
            "precision": round(precision, 3) if precision is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
            "hallucination_rate": round(hallucination_rate, 3) if hallucination_rate is not None else None,
        }

    if group_key is None:
        return _metrics_for(all_results)

    groups = {}
    for r in all_results:
        key = group_key(r)
        groups.setdefault(key, []).append(r)
    return {k: _metrics_for(v) for k, v in groups.items()}


def run_evaluation(
    ground_truth_path: str = None,
    extractions_folder: str = "extractions",
    output_path: str = "evaluation_report.json",
):
    if ground_truth_path is None:
        # Auto-detect: check common filenames regardless of what format they actually contain
        candidates = ["ground_truth.csv", "ground_truth.xlsx", "ground_truth.xls", "ground_truth"]
        found = [c for c in candidates if Path(c).exists()]
        if found:
            ground_truth_path = found[0]
        else:
            raise FileNotFoundError(
                "No ground_truth file found in the current folder. "
                "Place your labeled ground truth file here, named 'ground_truth.csv' or 'ground_truth.xlsx'."
            )

    gt = load_ground_truth(ground_truth_path)
    extractions_dir = Path(extractions_folder)

    all_results = []
    skipped_docs = []

    for doc_id, gt_fields in gt.items():
        # Skip documents with no ground truth filled in at all (e.g. shipment_1)
        if all(v is None or (isinstance(v, str) and v.strip() == "") for v in gt_fields.values()):
            skipped_docs.append(doc_id)
            continue

        extraction_path = extractions_dir / f"{doc_id}.json"
        if not extraction_path.exists():
            print(f"WARNING: no extraction file found for {doc_id}, skipping")
            skipped_docs.append(doc_id)
            continue

        with open(extraction_path, "r", encoding="utf-8") as f:
            extraction_result = json.load(f)

        doc_results = evaluate_document(doc_id, gt_fields, extraction_result)
        # attach document_type for slicing
        doc_type = extraction_result.get("document_type", "unknown")
        for r in doc_results:
            r["document_type"] = doc_type
        all_results.extend(doc_results)

    overall = compute_metrics(all_results)
    by_field = compute_metrics(all_results, group_key=lambda r: r["field"])
    by_doc_type = compute_metrics(all_results, group_key=lambda r: r["document_type"])

    report = {
        "documents_evaluated": len(gt) - len(skipped_docs),
        "documents_skipped": skipped_docs,
        "overall": overall,
        "by_field": by_field,
        "by_document_type": by_doc_type,
        "raw_results": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # Print a readable summary to console
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Documents evaluated: {report['documents_evaluated']}")
    if skipped_docs:
        print(f"Documents skipped (no ground truth): {skipped_docs}")

    print(f"\nOVERALL:")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    print(f"\nBY FIELD:")
    for field, m in by_field.items():
        print(f"  {field:20s} precision={m['precision']}, recall={m['recall']}, "
              f"hallucination_rate={m['hallucination_rate']}, n={m['n']}")

    print(f"\nBY DOCUMENT TYPE:")
    for dtype, m in by_doc_type.items():
        print(f"  {dtype:20s} precision={m['precision']}, recall={m['recall']}, "
              f"hallucination_rate={m['hallucination_rate']}, n={m['n']}")

    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    run_evaluation()
