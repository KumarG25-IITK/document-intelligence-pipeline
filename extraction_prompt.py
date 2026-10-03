EXTRACTION_PROMPT_TEMPLATE = """You are a document data extraction system. Extract structured data from the invoice/bill text below, following the schema exactly.

SCHEMA (extract these fields):
- vendor_name (string, required): Name of the company issuing the document. CRITICAL: Do NOT extract the marketplace platform (e.g., Flipkart, Amazon, Myntra, Tech-Connect) as the vendor. Look for the specific 'Sold By' or 'Billed By' entity.
- invoice_number (string, required): Unique invoice/bill identifier
- invoice_date (date, required, format YYYY-MM-DD): Issue date. Convert from whatever format appears in the source.
- due_date (date, optional, format YYYY-MM-DD): Payment due date, if present
- customer_name (string, optional): Name of the buyer/billed customer
- line_items (array, optional): List of {{description, quantity, unit_price, amount}}. Use null for quantity/unit_price if not itemized that way (e.g. utility bills).
- subtotal (number, optional): Amount before tax. CRITICAL: Beware of multiple tables or shipping charges. The subtotal should represent the sum of the items before tax. Ensure the extracted Tax + Subtotal = Total Amount.
- tax_amount (number, optional): Total tax charged
- total_amount (number, required): Final amount payable
- currency (string, required): Currency code/symbol as it appears (e.g. INR, Rs, EUR)
CRITICAL RULES:
1. If a field's value does not appear in the text, output null for it. NEVER guess, infer, round, or fabricate a plausible-looking value.
2. For every non-null field, include the exact snippet of source text you extracted it from, in an "evidence" object alongside the value.
3. Numbers must be extracted as plain numbers (no currency symbols, no commas) in the value field — but preserve the original formatting in the evidence snippet.
4. Output ONLY valid JSON. No explanation, no markdown code fences, no preamble.
5. If the document text is garbled, unreadable, or clearly OCR-corrupted for a given field, set that field to null rather than guessing from partial/corrupted text.

OUTPUT FORMAT (follow this exact structure):
{{
  "vendor_name": {{"value": "...", "evidence": "..."}} or {{"value": null, "evidence": null}},
  "invoice_number": {{"value": "...", "evidence": "..."}},
  "invoice_date": {{"value": "YYYY-MM-DD", "evidence": "..."}},
  "due_date": {{"value": "YYYY-MM-DD", "evidence": "..."}} or {{"value": null, "evidence": null}},
  "customer_name": {{"value": "...", "evidence": "..."}} or {{"value": null, "evidence": null}},
  "line_items": [
    {{"description": "...", "quantity": 1, "unit_price": 10.5, "amount": 10.5}}
  ] or [],
  "subtotal": {{"value": 100.0, "evidence": "..."}} or {{"value": null, "evidence": null}},
  "tax_amount": {{"value": 10.0, "evidence": "..."}} or {{"value": null, "evidence": null}},
  "total_amount": {{"value": 110.0, "evidence": "..."}},
  "currency": {{"value": "INR", "evidence": "..."}}
}}

DOCUMENT TEXT:
---
{document_text}
---

Extract the JSON now."""
