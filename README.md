# Geotechnical PDF Data Extraction Pipeline

Automated pipeline that parses heterogeneous geotechnical laboratory report PDFs and extracts soil test data into a standardized table format.

## Features

- **Keyword-based extraction** — No hard-coded page positions. Uses regex and keyword matching to identify test types and extract data from any geotechnical PDF.
- **7-category data normalization** — Cleans numeric values, standardizes depths, normalizes borehole IDs, canonicalizes null values, and validates percentages.
- **Confidence scoring** — Each row scored 0.0–1.0 based on field completeness, numeric validity, PSD consistency, and metadata quality.
- **Groq LLM fallback** — Low-confidence rows are automatically re-extracted using a Groq LLM (configurable via environment variables). Entirely optional — the pipeline works without it.
- **Multiple output formats** — CSV, JSON, and console table.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment (optional — for LLM fallback)

Copy `.env` and set your Groq API key:

```env
GROQ_API_KEY=gsk_your_api_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile
CONFIDENCE_THRESHOLD=0.6
```

### 3. Run Pipeline (CLI)

```bash
python extract_geotech.py "Lab Results-Geotechnical-Factual-Report (1).pdf"
```

Output files are saved to `./output/`:
- `extracted_results.csv`
- `extracted_results.json`

### 4. Run Dashboard (Web UI)

```bash
streamlit run app.py
```
This will open a web browser where you can upload PDF reports and download extracted data interactively.

### CLI Options

```
python extract_geotech.py <pdf_path> [options]

Options:
  --output-dir DIR         Output directory (default: ./output)
  --format {csv,json,console}  Output formats (default: all three)
  --confidence-threshold N Override threshold (default: 0.6)
  -v, --verbose            Debug logging
```

## Output Table Schema

| Column | Description |
|---|---|
| Location | Borehole ID (e.g., BH01) or lab sample number |
| Sample depth (m) | Depth range (e.g., 2.5-2.95) |
| Soil description (group symbol) | Soil type with USCS symbol if present |
| Geological unit | Formation name (e.g., Alluvium) or `-` if not in PDF |
| % Fines | Passing 0.075mm sieve |
| % Sand | Fraction between 0.075mm and 4.75mm |
| % Gravel | Fraction above 4.75mm |
| LL (%) | Liquid Limit |
| PL (%) | Plastic Limit |
| PI (%) | Plasticity Index (LL - PL) |
| LS (%) | Linear Shrinkage |
| Confidence | Extraction quality score (0.0–1.0) |
| Extraction Method | `rule-based`, `llm-enhanced`, or `llm-fallback` |

## Pipeline Architecture

```
PDF → Page Classifier → Extractors (PSD, Atterberg, Metadata)
    → Data Normalizer → Aggregator → Confidence Scorer
    → [Groq LLM Fallback if score < threshold] → Output
```

## How It Handles PDF Variability

1. **Page classification**: Each page is scanned for keywords (e.g., "Particle Size Distribution", "Liquid Limit") to determine the test type — no page numbers are hard-coded.
2. **Regex metadata**: Sample IDs, depths, bore IDs are extracted via flexible regex patterns that handle multiple formats.
3. **Table parsing**: Uses `pdfplumber` to extract tables, then matches column headers to identify data fields.
4. **Missing data**: Any field not found in the PDF outputs `-`. The code never crashes on missing data.

## Confidence Scoring

Each row is scored using four weighted sub-scores:

| Sub-Score | Weight | Measures |
|---|---|---|
| Field Completeness | 40% | How many of 11 target columns are filled |
| Numeric Validity | 20% | Whether numeric fields are parseable numbers |
| PSD Consistency | 20% | Whether Fines + Sand + Gravel ≈ 100% |
| Metadata Quality | 20% | Location + Depth + Description present |

Rows below the threshold trigger the Groq LLM fallback (if configured), which re-extracts from raw page text and fills gaps using a "fill gaps, don't overwrite" strategy.

## Dependencies

- `pdfplumber` — PDF text and table extraction
- `pandas` — DataFrame operations and CSV output
- `groq` — Groq LLM API client (optional)
- `python-dotenv` — Environment variable loading
