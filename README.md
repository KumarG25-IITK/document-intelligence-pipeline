# Document Intelligence Pipeline

An OCR + LLM pipeline that turns real-world invoices and bills into structured JSON, then **measures how often it is wrong** and **decides which fields a human should review**.

Most extraction demos stop at "the LLM returned JSON." This project goes further: field-level evaluation against hand-labeled ground truth, a hallucination metric, a failure taxonomy, and a calibrated confidence layer for human-in-the-loop review.

## Results

Evaluated on 30 real documents (Amazon, Flipkart, Myntra, electricity bills, hospital bills, and others), with hand-labeled ground truth on 9 scalar fields per document.

| Metric | Result |
|---|---|
| Precision | 91.6% |
| Recall | 92.0% |
| F1 | 91.8% |
| Hallucination rate | 3.3% |

**By document type:** Amazon and Myntra invoices reach 100%. The weakest types are electricity bills (66.7%, scanned, Odia-language) and hospital bills (74.1%, inconsistent layouts).

**Human-review layer (threshold 0.85):** 74.1% of fields are auto-accepted, the error rate among accepted fields is 5.5%, and 59.3% of all wrong fields are caught and routed to review.

*Definitions:* precision = correct / non-null predictions; recall = correct / fields that exist in ground truth; hallucination rate = fields where the model invented a value that is absent from ground truth, divided by all field evaluations.

## How it works

```
invoices/ (PDFs)
   │
   ▼
ocr_pipeline.py        digital PDF → pdftotext   |   scanned PDF → Tesseract (eng+ori)
   │                   → outputs/*.json (raw_text + metadata) + inventory.csv
   ▼
extraction.py          Gemini + extraction_prompt.py → structured JSON with an evidence span per field
   │                   → schema validation, currency normalization, model fallback on quota limits
   ▼                   → extractions/*.json
evaluate.py            compare against ground_truth → precision / recall / F1 / hallucination rate
   │
   ├─▶ failure_analysis.py   groups every failure by document type and field
   └─▶ confidence.py         per-field confidence score → review threshold calibration
```

### Design decisions

- **Evidence-backed extraction.** Every non-null field must include the exact source text it came from. Fields with a value but no evidence get flagged by validation.
- **Null over guessing.** The prompt tells the model to return `null` when a value is missing or the OCR is corrupted, and the evaluator counts invented values as hallucinations.
- **Hybrid ingestion.** The pipeline checks for a text layer first, so clean digital PDFs skip OCR (faster, more accurate) and only scans go through Tesseract.
- **Observable confidence, not self-reported.** The LLM's own confidence isn't a calibrated probability. The score combines evidence presence, validation warnings, a `subtotal + tax ≈ total` cross-check, and source quality (digital vs OCR).
- **Resumable batch runs with model fallback.** Extraction skips documents that already have results and switches to the next Gemini model when a free-tier quota runs out.

### Schema

`vendor_name`, `invoice_number`, `invoice_date`, `due_date`, `customer_name`, `subtotal`, `tax_amount`, `total_amount`, `currency`, plus `line_items`. Each scalar field is returned as `{"value": ..., "evidence": ...}`. See `schema.json`.

## Failure analysis highlights

- **Marketplace vs seller:** models tended to extract the platform (Flipkart, Amazon) as the vendor instead of the actual seller. This was fixed with an explicit prompt rule.
- **Subtotal confusion:** shipping charges and multiple tables caused wrong subtotals. This was addressed with a prompt instruction to check that tax + subtotal = total.
- **Currency encoding:** a mojibake bug (`â‚¹` instead of `₹`) caused false mismatches and was fixed in normalization.
- **Scanned regional-language bills:** these were the weakest category because of OCR noise on Odia text. Extraction quality depends heavily on OCR quality.

## Limitations

- The evaluation set is small (30 documents), and the confidence threshold was chosen on the same set it is evaluated on, so the numbers are likely optimistic.
- `line_items` are extracted but not evaluated, only the scalar fields are.
- Matching is exact after normalization, so a vendor written two different ways counts as a miss.
- At threshold 0.85 the score acts partly as a rule: fields from scanned documents score about 0.845 even with evidence and no warnings, so they are routed to review. Coverage therefore reflects the digital/scanned split as much as a graded score.
- A failed total cross-check lowers a field's score but does not by itself force review.

## Setup

**Requirements:** Python 3.10+, [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (install the Odia `ori` language data for scanned regional documents), and [Poppler](https://github.com/oschwartz10612/poppler-windows).

```bash
pip install -r requirements.txt
```

`ocr_pipeline.py` has Windows paths for Tesseract and Poppler at the top. Edit `POPPLER_BIN` and the Tesseract path to match your machine.

Set your Gemini API key as an environment variable (never commit it):

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-key"
# macOS / Linux
export GEMINI_API_KEY="your-key"
```

## Run

```bash
python ocr_pipeline.py invoices     # 1. OCR / text extraction  → outputs/
python extraction.py                # 2. LLM extraction          → extractions/
python evaluate.py                  # 3. metrics (needs ground_truth.xlsx or .csv)
python failure_analysis.py          # 4. failure report
python confidence.py                # 5. confidence + review calibration
```

## Data and privacy

The original invoices, extracted outputs, and ground-truth labels contain real personal information and are **not included** in this repository. The generated reports (`evaluation_report.json`, `confidence_report.json`, `failure_analysis_report.txt`) are also excluded for the same reason. To reproduce the results, add your own PDFs to `invoices/` and label them using the column layout described in `evaluate.py`.

## Project structure

```
ocr_pipeline.py        OCR + digital text extraction
extraction.py          LLM extraction, validation, normalization
extraction_prompt.py   extraction prompt and schema instructions
evaluate.py            evaluation harness
failure_analysis.py    failure grouping
confidence.py          confidence scoring and calibration
schema.json            field schema
```

## Tech

Python, Tesseract OCR, Poppler, Google Gemini API, openpyxl.
