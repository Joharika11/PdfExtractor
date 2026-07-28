"""
AI-Powered Document/Invoice Data Extractor
--------------------------------------------
Upload an invoice, contract, or form (PDF) and this app will:
  1. Extract the raw text from the PDF
  2. Send it to Gemini (Google AI API) with a structured extraction prompt
  3. Return clean, structured fields (JSON) + flag any missing/inconsistent data
  4. Let you download the result as CSV or JSON

Run with:
    streamlit run app.py

Requires a Google AI API key. Set it as an environment variable:
    export GOOGLE_API_KEY="your-key-here"
or paste it into the sidebar box when the app opens.
"""

import os
import json
import io

import streamlit as st
import pandas as pd
import pdfplumber
import google.generativeai as genai

st.set_page_config(page_title="AI Document Extractor", page_icon="📄", layout="wide")

# ---------- Helper functions ----------

def extract_text_from_pdf(uploaded_file) -> str:
    """Pull raw text out of an uploaded PDF file."""
    text_parts = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def build_extraction_prompt(document_text: str) -> str:
    return f"""You are a data extraction engine for business operations. Extract structured
information from the document text below. Return ONLY valid JSON, no other commentary,
no markdown fences.

Extract these fields if present (use null if a field is missing or unclear):
- document_type (e.g. invoice, contract, purchase order, receipt)
- document_number
- date
- due_date
- vendor_name
- customer_name
- total_amount
- currency
- line_items (a list of objects with description, quantity, unit_price, amount — if applicable)
- tax_amount
- payment_terms

Also include a field called "flags": a list of short strings describing any issues you notice,
such as missing critical fields, inconsistent totals (line items don't sum to total),
unclear dates, or ambiguous currency. If everything looks consistent, return an empty list.

Document text:
---
{document_text}
---

Return ONLY the JSON object."""


def call_gemini(api_key: str, prompt: str) -> dict:
    genai.configure(api_key=api_key)
    # If this model name is unavailable in your account/region, swap it for
    # another current Gemini model (check https://ai.google.dev/gemini-api/docs/models).
   model = genai.GenerativeModel("gemini-flash-latest")
    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    # Strip markdown fences if the model added them anyway
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    return json.loads(raw_text)


def flatten_for_table(data: dict) -> pd.DataFrame:
    """Turn the top-level (non line-item) fields into a simple 2-column table."""
    rows = []
    for key, value in data.items():
        if key in ("line_items", "flags"):
            continue
        rows.append({"Field": key, "Value": value})
    return pd.DataFrame(rows)


# ---------- UI ----------

st.title("📄 AI-Powered Document Data Extractor")
st.caption(
    "Upload an invoice, contract, or form. Gemini reads it, pulls out structured "
    "fields, and flags anything that looks missing or inconsistent — cutting down "
    "manual data entry and speeding up review."
)

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input(
        "Google AI API Key",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        type="password",
        help="Get one at aistudio.google.com/apikey. Or set GOOGLE_API_KEY as an env var.",
    )
    st.markdown("---")
    st.markdown(
        "**How it works**\n"
        "1. Extracts raw text from the PDF\n"
        "2. Sends it to Gemini with a structured extraction prompt\n"
        "3. Parses the JSON response into a table\n"
        "4. Flags missing/inconsistent data for manual review"
    )

uploaded_file = st.file_uploader("Upload a PDF (invoice, contract, receipt, form)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        document_text = extract_text_from_pdf(uploaded_file)

    if not document_text.strip():
        st.error("Couldn't extract any text from this PDF. It may be a scanned image without OCR.")
    else:
        with st.expander("Show raw extracted text"):
            st.text(document_text[:5000])

        if st.button("🔍 Extract Structured Data", type="primary"):
            if not api_key_input:
                st.error("Please enter your Google AI API key in the sidebar first.")
            else:
                with st.spinner("Calling Gemini to extract structured fields..."):
                    try:
                        prompt = build_extraction_prompt(document_text)
                        result = call_gemini(api_key_input, prompt)
                        st.session_state["result"] = result
                    except json.JSONDecodeError:
                        st.error("Model returned non-JSON output. Try again or check the document quality.")
                    except Exception as e:
                        st.error(f"Error calling Gemini API: {e}")

if "result" in st.session_state:
    result = st.session_state["result"]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Extracted Fields")
        st.dataframe(flatten_for_table(result), use_container_width=True, hide_index=True)

        if result.get("line_items"):
            st.subheader("Line Items")
            st.dataframe(pd.DataFrame(result["line_items"]), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("⚠️ Flags")
        flags = result.get("flags", [])
        if flags:
            for f in flags:
                st.warning(f)
        else:
            st.success("No issues detected. Data looks consistent.")

    st.markdown("---")
    st.subheader("Download")

    json_bytes = json.dumps(result, indent=2).encode("utf-8")
    st.download_button("Download JSON", data=json_bytes, file_name="extracted_data.json", mime="application/json")

    csv_buffer = io.StringIO()
    flatten_for_table(result).to_csv(csv_buffer, index=False)
    st.download_button(
        "Download CSV",
        data=csv_buffer.getvalue(),
        file_name="extracted_data.csv",
        mime="text/csv",
    )