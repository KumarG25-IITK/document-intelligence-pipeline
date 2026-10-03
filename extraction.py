import json
import os
import time
from pathlib import Path
import google.generativeai as genai
from extraction_prompt import EXTRACTION_PROMPT_TEMPLATE

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Try models in this order. Each model has its OWN separate free-tier daily quota,
# so when one runs out, falling through to the next gives more total capacity per day
# instead of waiting for one model's quota to reset.
MODEL_FALLBACK_ORDER = [
    "gemini-3.8-flash",
    "gemini-3.1pro",
]

REQUIRED_FIELDS = ["vendor_name", "invoice_number", "invoice_date", "total_amount", "currency"]
ALL_SCALAR_FIELDS = [
    "vendor_name", "invoice_number", "invoice_date", "due_date",
    "customer_name", "subtotal", "tax_amount", "total_amount", "currency"
]


def validate_extraction(data: dict) -> list:
    """
    Check the LLM's output against the schema contract.
    Returns a list of warning strings (empty list = clean).
    """
    warnings = []

    for field in REQUIRED_FIELDS:
        if field not in data:
            warnings.append(f"MISSING_FIELD: '{field}' not in output at all")
        elif data[field].get("value") is None:
            warnings.append(f"REQUIRED_NULL: '{field}' is required but null")

    allowed_keys = set(ALL_SCALAR_FIELDS + ["line_items"])
    extra_keys = set(data.keys()) - allowed_keys
    if extra_keys:
        warnings.append(f"UNEXPECTED_FIELDS: {extra_keys} not in approved schema")

    for field in ALL_SCALAR_FIELDS:
        if field in data and data[field].get("value") is not None:
            if not data[field].get("evidence"):
                warnings.append(f"MISSING_EVIDENCE: '{field}' has a value but no evidence span")

    for field in ["subtotal", "tax_amount", "total_amount"]:
        if field in data and data[field].get("value") is not None:
            if not isinstance(data[field]["value"], (int, float)):
                warnings.append(f"TYPE_ERROR: '{field}' should be numeric, got {type(data[field]['value']).__name__}")

    return warnings

def normalize_currency(raw_currency_string: str) -> str:
    """Normalize various Indian Rupee symbols/strings to 'INR'."""
    if not raw_currency_string:
        return None

    cleaned = raw_currency_string.strip().upper()
    # Handle the weird encoding you saw in your terminal logs ('â‚¹')
    inr_variants = ["₹", "RS", "RUPEES", "INR", "Â‚¹"]

    if any(var in cleaned for var in inr_variants):
        return "INR"

    usd_variants = ["$", "USD"]
    if any(var in cleaned for var in usd_variants):
        return "USD"

    return raw_currency_string


def extract_from_document(ocr_json_path: Path, model_name: str) -> dict:
    """Run LLM extraction on one OCR-output JSON and validate the result."""
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        ocr_result = json.load(f)

    document_text = ocr_result["raw_text"]
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(document_text=document_text)

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    raw_response = response.text.strip()

    # Strip markdown code fences if the model added them despite instructions
    if raw_response.startswith("```"):
        raw_response = raw_response.split("```")[1]
        if raw_response.startswith("json"):
            raw_response = raw_response[4:]
        raw_response = raw_response.strip()

    try:
        extracted = json.loads(raw_response)
    except json.JSONDecodeError as e:
        return {
            "document_id": ocr_result["document_id"],
            "extraction_status": "PARSE_FAILED",
            "error": str(e),
            "raw_response": raw_response,
        }
    if "currency" in extracted and extracted["currency"].get("value"):
        extracted["currency"]["value"] = normalize_currency(str(extracted["currency"]["value"]))

    warnings = validate_extraction(extracted)

    return {
        "document_id": ocr_result["document_id"],
        "document_type": ocr_result.get("document_type", "unknown"),
        "extraction_status": "ok" if not warnings else "ok_with_warnings",
        "warnings": warnings,
        "extracted_fields": extracted,
    }


def batch_extract(ocr_output_folder: str = "outputs", extraction_output_folder: str = "extractions",
                   delay_seconds: float = 5.0, max_retries: int = 3):
    """
    delay_seconds: pause between requests to stay under free-tier rate limits.
    max_retries: how many times to retry a single document if rate-limited (429),
                 waiting longer each time (exponential backoff).
    Resumable: skips any document that already has a result file in extraction_output_folder,
               so re-running after a rate-limit stop picks up where it left off.
    """
    input_dir = Path(ocr_output_folder)
    output_dir = Path(extraction_output_folder)
    output_dir.mkdir(exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != "inventory.csv"]

    # Resume: skip documents already successfully processed
    remaining = []
    skipped = 0
    for f in json_files:
        out_path = output_dir / f.name
        if out_path.exists():
            skipped += 1
        else:
            remaining.append(f)

    if skipped:
        print(f"Skipping {skipped} already-processed documents (resuming).")
    print(f"Extracting fields from {len(remaining)} remaining documents...")

    # Track which model we're currently using -- once one is exhausted for the day,
    # stick with the next one for the rest of the run instead of re-testing the dead one.
    current_model_idx = 0

    for i, ocr_path in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] {ocr_path.name} ...", end=" ")

        attempt = 0
        while True:
            attempt += 1
            model_name = MODEL_FALLBACK_ORDER[current_model_idx]
            try:
                result = extract_from_document(ocr_path, model_name)
                out_path = output_dir / ocr_path.name
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                status = result["extraction_status"]
                n_warnings = len(result.get("warnings", []))
                print(f"{status} [{model_name}]" + (f" ({n_warnings} warnings)" if n_warnings else ""))
                break  # success, move to next document

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "quota" in error_str.lower()

                if is_rate_limit and current_model_idx < len(MODEL_FALLBACK_ORDER) - 1:
                    current_model_idx += 1
                    next_model = MODEL_FALLBACK_ORDER[current_model_idx]
                    print(f"\n    {model_name} quota exhausted. Switching to {next_model}...")
                    continue  # retry immediately with the next model, no waiting needed

                elif is_rate_limit and attempt <= max_retries:
                    wait_time = 60 * attempt
                    print(f"\n    All models rate limited. Waiting {wait_time}s before retry {attempt}/{max_retries}...")
                    time.sleep(wait_time)
                    continue

                else:
                    print(f"FAILED: {e}")
                    break  # give up on this document, move to next

        time.sleep(delay_seconds)  # pause between every request regardless

    print(f"\nDone. Results saved to {extraction_output_folder}/")


if __name__ == "__main__":
    batch_extract()