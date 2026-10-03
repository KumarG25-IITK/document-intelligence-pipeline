"""
Batch processor for the Document Intelligence pipeline.

Processes every PDF in a folder, auto-detecting:
- document type (from filename prefix, e.g. "amazon_1.pdf" -> "amazon")
- digital vs scanned (via ocr_pipeline.has_text_layer)

For scanned documents, uses eng+ori by default (safe since eng+ori also handles
pure-English scans fine -- it just checks against both models).

Outputs:
- One JSON file per document in outputs/
- A single inventory.csv summarizing all documents (for your Day 1 dataset register)

IMPORTANT: Windows paths -- edit POPPLER_BIN and TESSERACT_EXE below to match your setup.
"""

import subprocess
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
import json
import csv
import sys
import io

# --- Windows setup: EDIT THESE TO MATCH YOUR MACHINE ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_BIN = r"C:\poppler\poppler-26.02.0\Library\bin"
# ---------------------------------------------------------

def has_text_layer(pdf_path: str) -> bool:
    result = subprocess.run(
        [f"{POPPLER_BIN}\\pdftotext.exe", "-enc", "UTF-8", pdf_path, "-"],
        capture_output=True, text=True, encoding="utf-8"
    )
    return len(result.stdout.strip()) > 20


def extract_digital_text(pdf_path: str) -> str:
    result = subprocess.run(
        [f"{POPPLER_BIN}\\pdftotext.exe", "-layout", "-enc", "UTF-8", pdf_path, "-"],
        capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout


def ocr_scanned_pdf(pdf_path: str, lang: str = "eng") -> str:
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_BIN)
    full_text = []
    for i, page_img in enumerate(pages):
        text = pytesseract.image_to_string(page_img, lang=lang)
        full_text.append(f"--- Page {i+1} ---\n{text}")
    return "\n".join(full_text)


def guess_doc_type(filename: str) -> str:
    """Infer document type from filename prefix. Adjust keywords as needed."""
    name = filename.lower()
    if "amazon" in name:
        return "amazon"
    elif "flipkart" in name:
        return "flipkart"
    elif "myntra" in name:
        return "myntra"
    elif "electricity" in name:
        return "electricity_bill"
    elif "hospital" in name:
        return "hospital_bill"
    elif "bill" in name:
        return "bill"
    elif "scanned" in name:
        return "scanned_document"
    else:
        return "unknown"


def process_document(pdf_path: Path, lang: str = "eng+ori") -> dict:
    is_digital = has_text_layer(str(pdf_path))

    if is_digital:
        text = extract_digital_text(str(pdf_path))
        method = "digital_text_extraction"
    else:
        text = ocr_scanned_pdf(str(pdf_path), lang=lang)
        method = f"ocr_tesseract_{lang}"

    return {
        "document_id": pdf_path.stem,
        "source_path": str(pdf_path),
        "document_type": guess_doc_type(pdf_path.name),
        "is_digital": is_digital,
        "extraction_method": method,
        "raw_text": text,
        "char_count": len(text.strip()),
    }


def batch_process(input_folder: str, output_folder: str = "outputs"):
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_folder}")
        return

    inventory_rows = []

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Processing {pdf_path.name} ...")
        try:
            result = process_document(pdf_path)

            # Save individual JSON
            out_json = output_dir / f"{pdf_path.stem}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Add to inventory
            inventory_rows.append({
                "document_id": result["document_id"],
                "filename": pdf_path.name,
                "document_type": result["document_type"],
                "is_digital": result["is_digital"],
                "extraction_method": result["extraction_method"],
                "char_count": result["char_count"],
                "status": "ok",
            })
            print(f"    -> {result['extraction_method']}, {result['char_count']} chars extracted")

        except Exception as e:
            # Never let one bad file kill the whole batch -- log it and move on
            print(f"    !! FAILED: {e}")
            inventory_rows.append({
                "document_id": pdf_path.stem,
                "filename": pdf_path.name,
                "document_type": guess_doc_type(pdf_path.name),
                "is_digital": "",
                "extraction_method": "",
                "char_count": 0,
                "status": f"error: {e}",
            })

    # Write inventory CSV -- this is your Day 1 dataset register
    inventory_path = output_dir / "inventory.csv"
    with open(inventory_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "document_id", "filename", "document_type", "is_digital",
            "extraction_method", "char_count", "status"
        ])
        writer.writeheader()
        writer.writerows(inventory_rows)

    print(f"\nDone. {len(pdf_files)} documents processed.")
    print(f"Individual results: {output_dir}/*.json")
    print(f"Inventory summary: {inventory_path}")

if __name__ == "__main__":
    # Fix Unicode printing in the Windows terminal (only when run as a script)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Default: looks for a folder called "invoices" next to this script
    folder = sys.argv[1] if len(sys.argv) > 1 else "invoices"
    batch_process(folder)