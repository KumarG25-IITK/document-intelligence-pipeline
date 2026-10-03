"""
Failure analysis for the Document Intelligence pipeline (Day 5).

Reads evaluation_report.json (from evaluate.py) and surfaces every real failure
case -- incorrect, missed, or hallucinated fields -- grouped by document type
and field, so patterns can be identified and classified into a failure taxonomy.

This does NOT auto-diagnose root causes (that needs human judgment looking at the
actual document), but it does the tedious part: finding every failure and grouping
them so the worst patterns are obvious at a glance.

Run:
    python failure_analysis.py
"""

import json
from collections import defaultdict
from pathlib import Path


def load_failures(evaluation_report_path: str = "evaluation_report.json") -> list:
    with open(evaluation_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    failures = [
        r for r in report["raw_results"]
        if r["outcome"] in ("incorrect", "missed", "hallucination")
    ]
    return failures


def group_and_report(failures: list, output_path: str = "failure_analysis_report.txt"):
    import io
    buf = io.StringIO()

    def p(*args):
        text = " ".join(str(a) for a in args)
        print(text)
        buf.write(text + "\n")

    by_doc_type = defaultdict(list)
    by_field = defaultdict(list)
    by_outcome = defaultdict(list)

    for f in failures:
        by_doc_type[f["document_type"]].append(f)
        by_field[f["field"]].append(f)
        by_outcome[f["outcome"]].append(f)

    p(f"\n{'='*70}")
    p(f"FAILURE ANALYSIS — {len(failures)} total failures found")
    p(f"{'='*70}")

    p(f"\nFAILURES BY OUTCOME TYPE:")
    for outcome, items in sorted(by_outcome.items(), key=lambda x: -len(x[1])):
        p(f"  {outcome:15s}: {len(items)}")

    p(f"\nFAILURES BY DOCUMENT TYPE (worst first):")
    for doc_type, items in sorted(by_doc_type.items(), key=lambda x: -len(x[1])):
        p(f"  {doc_type:20s}: {len(items)} failures")

    p(f"\nFAILURES BY FIELD (worst first):")
    for field, items in sorted(by_field.items(), key=lambda x: -len(x[1])):
        p(f"  {field:20s}: {len(items)} failures")

    p(f"\n{'='*70}")
    p("DETAILED FAILURE LIST (for manual classification)")
    p(f"{'='*70}")
    p("For each, note WHERE it likely broke: ingestion / OCR / extraction /")
    p("normalization / validation / confidence -- and why.\n")

    doc_ids_by_failure_count = defaultdict(int)
    for f in failures:
        doc_ids_by_failure_count[f["document_id"]] += 1

    for doc_id in sorted(doc_ids_by_failure_count, key=lambda d: -doc_ids_by_failure_count[d]):
        doc_failures = [f for f in failures if f["document_id"] == doc_id]
        p(f"\n--- {doc_id} ({doc_failures[0]['document_type']}) — {len(doc_failures)} failure(s) ---")
        for f in doc_failures:
            p(f"  [{f['outcome']}] {f['field']}:")
            p(f"      ground truth: {f['gt_value']!r}")
            p(f"      predicted:    {f['pred_value']!r}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"\n\nFull report saved to: {output_path}")


if __name__ == "__main__":
    failures = load_failures()
    group_and_report(failures)