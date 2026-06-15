#!/usr/bin/env python3
"""
Geotechnical PDF Data Extraction Pipeline
==========================================
Parses heterogeneous geotechnical laboratory report PDFs and extracts
soil test data into a standardized table format.

Pipeline: PDF → Page Classify → Extract → Normalize → Aggregate
         → Confidence Score → (optional) Groq LLM Fallback → Output

Usage:
    python extract_geotech.py "Lab Results-Geotechnical-Factual-Report (1).pdf"
    python extract_geotech.py report.pdf --output-dir ./results --format csv json

Author: Auto-generated extraction pipeline
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pdfplumber
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment configuration
# ---------------------------------------------------------------------------
load_dotenv()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# COMPONENT 1 — Data Normalizer
# ===========================================================================

# Canonical set of strings that mean "missing"
NULL_PATTERNS = {
    "", "none", "n/a", "na", "nt", "not obtainable", "not tested",
    "not reported", "nr", "ins", "-", "\u2013", "\u2014", "null",
}


def is_null(value) -> bool:
    """Check if a value should be treated as missing."""
    if value is None:
        return True
    return str(value).strip().lower() in NULL_PATTERNS


def normalize_numeric(value) -> str:
    """Clean a raw numeric string from PDF extraction."""
    if value is None:
        return "-"
    value = str(value).strip()
    value = re.sub(r"[\n\r\t]", "", value)          # control chars
    value = re.sub(r"[*#\u2020\u2021]+$", "", value).strip()  # footnote markers
    if is_null(value):
        return "-"
    # Try to coerce to a clean number
    try:
        num = float(value)
        return str(int(num)) if num == int(num) else str(round(num, 1))
    except ValueError:
        return value  # Preserve qualifiers like "<10"


def normalize_text(value) -> str:
    """Clean free-text fields (descriptions, soil types)."""
    if value is None:
        return "-"
    value = str(value)
    value = re.sub(r"[\n\r]+", " ", value)      # newlines → space
    value = re.sub(r"\s{2,}", " ", value)        # collapse spaces
    value = value.replace("\ufffd", "\u00b0")    # fix encoding for degree symbol
    value = value.strip()
    return value if value and not is_null(value) else "-"


def normalize_depth(value) -> str:
    """Standardize depth range to 'X.X-X.X' format."""
    if not value or is_null(value):
        return "-"
    value = str(value)
    value = re.sub(r"[mM]\b", "", value)                 # strip unit
    value = re.sub(r"\s*[\u2013\u2014]\s*", "-", value)  # en/em dash → hyphen
    value = re.sub(r"\s+to\s+", "-", value, flags=re.I)  # "to" → hyphen
    value = re.sub(r"\s*-\s*", "-", value)                # spaces around dash
    return value.strip() if value.strip() else "-"


def normalize_location(value) -> str:
    """Standardize borehole/location identifiers."""
    if not value or is_null(value):
        return "-"
    value = str(value).strip()
    # "Bore 29" / "Borehole 12" → "BH29" / "BH12"
    m = re.match(r"(?:Bore(?:hole)?)\s*[-]?\s*(\d+)", value, re.I)
    if m:
        return f"BH{m.group(1)}"
    # "BH-01" or "BH 01" → "BH01"
    m = re.match(r"(BH)\s*[-]?\s*(\d+)", value, re.I)
    if m:
        return f"BH{m.group(2)}"
    return value.upper().strip()


def normalize_soil_description(value) -> str:
    """Clean soil description text to title case, preserving USCS symbols."""
    value = normalize_text(value)
    if value == "-":
        return "-"
    # FIX: Preserve group symbols in uppercase (e.g., "(CL)", "(ML)", "(CH)")
    # Only apply title case to non-symbol parts
    if "(" in value and ")" in value:
        # Has a group symbol - preserve it
        match = re.search(r"^(.*?)\s*(\([A-Z]{2}(?:-[A-Z]{2})?\))$", value)
        if match:
            desc_part = match.group(1).title()
            symbol_part = match.group(2)
            return f"{desc_part} {symbol_part}"
    return value.title()


def extract_group_symbol(description: str) -> str:
    """Extract USCS group symbol if present in parentheses, e.g. (ML)."""
    if not description or description == "-":
        return ""
    m = re.search(r"\(([A-Z]{2}(?:-[A-Z]{2})?)\)", description, re.I)
    return m.group(1).upper() if m else ""


def validate_percentage(value: str, field_name: str) -> str:
    """Validate that a percentage is 0-100."""
    if value == "-":
        return "-"
    try:
        num = float(value)
        if not (0 <= num <= 100):
            logger.warning(f"{field_name}={num} outside 0-100% range")
            return "-"
        return value
    except ValueError:
        return value


# ===========================================================================
# COMPONENT 2 — Page Classifier
# ===========================================================================

# Keyword sets for classifying pages by test type (synonym-expanded)
TEST_KEYWORDS = {
    "psd": [
        "particle size distribution", "sieve", "passed %",
        "grain size", "gradation", "grading", "sieve analysis",
        "grain size distribution", "psd", "mechanical analysis",
        "wet sieving", "dry sieving", "hydrometer",
    ],
    "atterberg": [
        "atterberg limit", "liquid limit", "plastic limit",
        "atterberg", "consistency limit", "plasticity",
        "ll (%", "pl (%", "ll(%", "pl(%",
    ],
    "linear_shrinkage": [
        "linear shrinkage", "lin. shrinkage", "lin shrinkage",
        "shrinkage limit", "ls (%",
    ],
    "cbr": ["california bearing ratio", "cbr"],
    "moisture": [
        "moisture content", "water content", "natural moisture",
        "w (%", "w(%", "mc (%",
    ],
    "ucs": [
        "uniaxial compressive strength", "unconfined compressive",
        "ucs", "qu (",
    ],
    "emerson": ["emerson class", "emerson"],
    "pinhole": ["pinhole", "pin hole"],
    "aggressivity": [
        "soil aggressivity", "chloride", "sulphate", "sulfate",
        "ph value", "resistivity",
    ],
}


def classify_page(text: str) -> list:
    """Return list of test types found on a page by keyword matching."""
    if not text:
        return []
    text_lower = text.lower()
    found = []
    for test_type, keywords in TEST_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(test_type)
    return found


# ===========================================================================
# COMPONENT 3 — Metadata Extractor
# ===========================================================================

def extract_metadata(text: str) -> dict:
    """Extract sample-level metadata from a page's header text."""
    meta = {
        "sample_number": "-",
        "location": "-",
        "depth": "-",
        "soil_description": "-",
        "geological_unit": "-",
    }
    if not text:
        return meta

    # --- Sample Number ---
    # Matches patterns like: GU-3498A, NC-4562C, BH01, S-1, etc.
    # Synonym headers: "Sample Number", "Sample No", "Sample ID", "Lab No", "Specimen No"
    for sn_pattern in [
        r"(?:Sample\s*(?:Number|No\.?|ID|Ref)|Lab(?:oratory)?\s*(?:No\.?|Number|Ref)|Specimen\s*(?:No\.?|Number|ID))\s*[:\s]*([A-Z]{2,4}[-]?\d+[A-Z]*)",
    ]:
        m = re.search(sn_pattern, text, re.I)
        if m:
            meta["sample_number"] = m.group(1).strip()
            break

    # --- Location / Bore ID ---
    # Synonym headers: "Sample Location", "Test Location", "Bore", "Borehole",
    #                  "Location", "Hole No", "Drill Hole", "Test Pit", "TP"
    location_patterns = [
        r"(?:Sample|Test)\s*Location\s*[:\s]*(Bore\s*\d+|BH\s*[-]?\d+)",
        r"(?:Bore(?:hole)?|Hole|Drill\s*Hole)\s*(?:No\.?|Number|ID)?\s*[:\s]*(\d+)",
        r"(?:Test\s*Pit|TP)\s*(?:No\.?)?\s*[:\s]*(\d+)",
        r"Location\s*(?:ID|No\.?)?\s*[:\s]*(BH\s*[-]?\d+|TP\s*[-]?\d+)",
    ]
    for loc_pat in location_patterns:
        m = re.search(loc_pat, text, re.I)
        if m:
            loc_val = m.group(1).strip()
            # If only a number was captured, prefix with BH
            if loc_val.isdigit():
                loc_val = f"BH{loc_val}"
            meta["location"] = normalize_location(loc_val)
            break

    # --- Depth ---
    # Synonym headers: "Depth (m)", "Depth", "Sample Depth", "Depth Range",
    #                  "Depth Below Surface", "Depth (mbgl)"
    depth_patterns = [
        r"(?:Sample\s*)?Depth\s*(?:Range)?\s*(?:\(m(?:bgl)?\))?\s*[:\s]*([\d.]+\s*[-\u2013]\s*[\d.]+)",
        r"Depth\s*(?:Below\s*(?:Surface|Ground))?\s*(?:\(m\))?\s*[:\s]*([\d.]+\s*[-\u2013]\s*[\d.]+)",
        r"Depth\s*(?:\(m\))?\s*[:\s]*([\d.]+)\s*m?\b",
    ]
    for d_pat in depth_patterns:
        m = re.search(d_pat, text, re.I)
        if m:
            meta["depth"] = normalize_depth(m.group(1))
            break

    # --- Soil Description ---
    # Synonym headers: "Soil Description", "Material Description", "Soil Type",
    #                  "Description of Sample", "Visual Classification", "Sample Description"
    # FIX: Capture multi-line descriptions where text spans lines (e.g., "Red Brown Sandy" / "Clay")
    # Direct approach: match "Soil Description" followed by text, possibly with continuation on next line
    m = re.search(r"Soil\s+Description\s+([A-Za-z\s()-]+?)(?:\n\s*([A-Za-z\s()-]+?))?(?:\n|$)", text, re.I)
    if m:
        desc = m.group(1).strip()
        # If there's text on the next line and current desc is incomplete, append it
        if m.group(2):
            continuation = m.group(2).strip()
            # Only append if it looks like a word, not a header/section
            if re.match(r'^[A-Z][a-z]', continuation) and len(continuation) > 2 and ':' not in continuation:
                desc = f"{desc} {continuation}".strip()
        meta["soil_description"] = normalize_soil_description(desc)
    else:
        # Fallback patterns: "Material:" (with colon) or other description fields
        # FIX: Require colon after "Material" to avoid matching "Material Test Report" header
        for pattern in [
            r"Material\s*:\s*([A-Za-z\s()-]+?)(?:\n|$)",
            r"(?:Soil|Sample)\s*(?:Description|Type|Classification)\s*[:\s]*([A-Za-z\s()-]+?)(?:\n|$)",
            r"Description\s*(?:of\s*(?:Sample|Soil|Material))\s*[:\s]*([A-Za-z\s()-]+?)(?:\n|$)",
            r"Visual\s*(?:Classification|Description)\s*[:\s]*([A-Za-z\s()-]+?)(?:\n|$)",
        ]:
            m = re.search(pattern, text, re.I | re.MULTILINE)
            if m:
                desc = m.group(1).strip()
                if len(desc) > 2:
                    meta["soil_description"] = normalize_soil_description(desc)
                    break

    # --- Geological Unit ---
    # Synonym headers: "Geological Unit", "Geological Formation", "Stratigraphy",
    #                  "Geology", "Formation", "Stratum"
    for pattern in [
        r"(?:Geological|Geo\.?)\s*(?:Unit|Formation|Stratum)\s*[:\s]*([A-Za-z\s]+?)(?:\n|$)",
        r"(?:Stratigraphy|Formation|Stratum|Geology)\s*[:\s]*([A-Za-z\s]+?)(?:\n|$)",
        r"(?:Alluvium|Tamala\s*Sand|Residual\s*Soil|Fill|Colluvium|Basalt|Laterite|Saprolite|Shale|Sandstone|Limestone|Claystone|Siltstone|Mudstone)",
    ]:
        m = re.search(pattern, text, re.I)
        if m:
            meta["geological_unit"] = normalize_text(
                m.group(0) if not m.lastindex else m.group(1)
            )
            break

    return meta


# ===========================================================================
# COMPONENT 4 — PSD Extractor
# ===========================================================================

def extract_psd(tables: list) -> dict:
    """
    Extract Particle Size Distribution data from page tables.
    Returns: {pct_fines, pct_sand, pct_gravel}
    
    Logic:
      % Fines = passing 0.075 mm
      % Sand  = passing 4.75 mm - passing 0.075 mm
      % Gravel = 100 - passing 4.75 mm
    """
    result = {"pct_fines": "-", "pct_sand": "-", "pct_gravel": "-"}
    
    for table in tables:
        if not table:
            continue
        # Identify PSD table by looking for sieve/grading synonyms in first rows
        # Synonyms: "Sieve", "Passed %", "Passing %", "Percent Passing",
        #           "% Finer", "Grain Size", "Aperture", "Cumulative"
        header_text = " ".join(str(cell) for row in table[:2] for cell in row if cell)
        header_lower = header_text.lower()
        psd_header_synonyms = [
            "sieve", "passed", "passing", "percent passing",
            "% finer", "finer", "grain size", "aperture",
            "cumulative", "retained", "gradation",
        ]
        if not any(syn in header_lower for syn in psd_header_synonyms):
            continue

        # Parse sieve data into {sieve_mm: passing_%}
        sieve_data = {}
        for row in table:
            if not row or len(row) < 2:
                continue
            sieve_str = str(row[0]).strip() if row[0] else ""
            passed_str = str(row[1]).strip() if row[1] else ""

            # Extract sieve size in mm from strings like "0.075 mm", "4.75mm", "19 mm"
            # FIX: Enhanced regex to handle optional spaces and more sieve formats
            sieve_match = re.match(r"([\d.]+)\s*mm", sieve_str, re.I)
            if not sieve_match:
                continue
            try:
                sieve_mm = float(sieve_match.group(1))
                passed_pct = float(passed_str)
                sieve_data[sieve_mm] = passed_pct
            except (ValueError, TypeError):
                continue

        if not sieve_data:
            continue

        # Extract key sieve values
        fines_075 = sieve_data.get(0.075)
        sand_475 = sieve_data.get(4.75)

        # If 4.75mm sieve not present, look for nearby sieves
        if sand_475 is None:
            for candidate in [4.75, 4.7, 5.0]:
                if candidate in sieve_data:
                    sand_475 = sieve_data[candidate]
                    break

        # If 0.075mm not present, check for it
        if fines_075 is None:
            for candidate in [0.075, 0.08, 0.063]:
                if candidate in sieve_data:
                    fines_075 = sieve_data[candidate]
                    break

        # Compute percentages
        if fines_075 is not None:
            result["pct_fines"] = normalize_numeric(str(fines_075))

        if sand_475 is not None and fines_075 is not None:
            sand_pct = sand_475 - fines_075
            result["pct_sand"] = normalize_numeric(str(round(sand_pct, 1)))
        elif sand_475 is not None:
            result["pct_sand"] = normalize_numeric(str(sand_475))
        elif fines_075 is not None:
            # INFERENCE: If we only have fines but no sand/gravel boundary, infer sand = 100-fines, gravel = 0
            # This handles cases where PSD table only reports 0.075mm sieve without coarser boundaries
            sand_pct = 100.0 - fines_075
            result["pct_sand"] = normalize_numeric(str(round(sand_pct, 1)))
            result["pct_gravel"] = "0"

        if sand_475 is not None:
            gravel_pct = 100.0 - sand_475
            result["pct_gravel"] = normalize_numeric(str(round(gravel_pct, 1)))

        # Validate totals
        try:
            f = float(result["pct_fines"]) if result["pct_fines"] != "-" else 0
            s = float(result["pct_sand"]) if result["pct_sand"] != "-" else 0
            g = float(result["pct_gravel"]) if result["pct_gravel"] != "-" else 0
            total = f + s + g
            if total > 0 and abs(total - 100) > 5:
                logger.warning(f"PSD total={total}% (expected ~100%)")
        except ValueError:
            pass

        break  # Use first valid PSD table found

    return result


# ===========================================================================
# COMPONENT 5 — Atterberg & Linear Shrinkage Extractor
# ===========================================================================

# Synonym sets for Atterberg / Linear Shrinkage labels
# Each tuple: (target_key, list_of_long_synonyms, list_of_short_synonyms)
# Long synonyms use substring matching; short synonyms use word-boundary regex
# to avoid false positives (e.g. "pl" matching inside "sample").
ATTERBERG_SYNONYMS = [
    ("ll", {
        "long": ["liquid limit", "liq. limit", "liq limit", "liq lim", "liquidlimit"],
        "short": [r"\bll\b", r"\bwl\b", r"\bw_l\b"],
    }),
    ("pl", {
        "long": ["plastic limit", "plas. limit", "plas limit", "plas lim", "plasticlimit"],
        "short": [r"\bpl\b", r"\bwp\b", r"\bw_p\b"],
    }),
    ("ls", {
        "long": ["linear shrinkage", "lin. shrinkage", "lin shrinkage", "lin shrink", "linearshrinkage"],
        "short": [r"\bls\b"],
    }),
    ("pi", {
        "long": ["plasticity index", "plas. index", "plas index", "plasticityindex"],
        "short": [r"\bpi\b", r"\bip\b"],
    }),
]


def _match_atterberg_label(label: str) -> str:
    """Match a row label against Atterberg synonym sets. Returns field key or ''.

    Uses substring matching for long descriptive synonyms and word-boundary
    regex for short abbreviations (<=3 chars) to avoid false positives.
    """
    label_clean = label.lower().strip()
    for field_key, syn_dict in ATTERBERG_SYNONYMS:
        # Check long synonyms via substring
        for syn in syn_dict["long"]:
            if syn in label_clean:
                return field_key
        # Check short synonyms via word-boundary regex
        for pattern in syn_dict["short"]:
            if re.search(pattern, label_clean, re.I):
                return field_key
    return ""


def _is_numeric_value(value: str) -> bool:
    """Check if a value string looks like a number (allows 'Non Plastic' etc.)."""
    if not value or value == "-":
        return False
    # Accept "Non Plastic", "NP" as valid Atterberg results
    if value.lower() in ("non plastic", "np", "non-plastic"):
        return True
    try:
        float(value.replace(",", ""))
        return True
    except ValueError:
        return False


def extract_atterberg(tables: list) -> dict:
    """
    Extract Atterberg Limits and Linear Shrinkage from page tables.
    Uses synonym-expanded label matching to handle different lab formats.
    Only accepts values that look numeric (rejects 'Oven Dried', etc.).
    Returns: {ll, pl, pi, ls}
    """
    result = {"ll": "-", "pl": "-", "pi": "-", "ls": "-"}

    for table in tables:
        if not table:
            continue
        for row in table:
            if not row or len(row) < 2:
                continue
            label = str(row[0]).strip().lower() if row[0] else ""
            value = str(row[1]).strip() if row[1] else ""

            matched_key = _match_atterberg_label(label)
            if matched_key and result[matched_key] == "-" and _is_numeric_value(value):
                result[matched_key] = normalize_numeric(value)

    # Compute PI = LL - PL if both are numeric
    if result["ll"] != "-" and result["pl"] != "-":
        try:
            ll_val = float(result["ll"])
            pl_val = float(result["pl"])
            result["pi"] = normalize_numeric(str(round(ll_val - pl_val, 1)))
        except ValueError:
            pass

    return result


# ===========================================================================
# COMPONENT 6 — Confidence Scorer
# ===========================================================================

ALL_FIELDS = [
    "location", "sample_depth", "soil_description", "geological_unit",
    "pct_fines", "pct_sand", "pct_gravel", "ll", "pl", "pi", "ls",
]
NUMERIC_FIELDS = ["pct_fines", "pct_sand", "pct_gravel", "ll", "pl", "pi", "ls"]
METADATA_FIELDS = ["location", "sample_depth", "soil_description"]


def compute_confidence(row: dict) -> float:
    """Score a single aggregated row from 0.0 to 1.0."""

    # 1. Field completeness (40%)
    filled = sum(1 for f in ALL_FIELDS if row.get(f, "-") != "-")
    completeness = filled / len(ALL_FIELDS)

    # 2. Numeric validity (20%)
    numeric_count = 0
    valid_count = 0
    for f in NUMERIC_FIELDS:
        val = row.get(f, "-")
        if val != "-":
            numeric_count += 1
            try:
                float(val)
                valid_count += 1
            except ValueError:
                pass
    numeric_score = valid_count / numeric_count if numeric_count > 0 else 0.0

    # 3. PSD consistency (20%) - FIX: Handle partial PSD data correctly
    # Count how many PSD fields are present (not "-")
    psd_fields_present = sum(1 for k in ["pct_fines", "pct_sand", "pct_gravel"] if row.get(k, "-") != "-")
    psd_score = 0.0
    
    if psd_fields_present > 0:
        try:
            psd_values = []
            for k in ["pct_fines", "pct_sand", "pct_gravel"]:
                val = row.get(k, "-")
                if val != "-":
                    psd_values.append(float(val))
            
            if psd_values:
                total = sum(psd_values)
                # Score based on how close to 100% the present fields sum to
                if abs(total - 100) <= 2:
                    psd_score = 1.0
                elif abs(total - 100) <= 10:
                    psd_score = 0.7  # Partial but reasonable
                elif psd_fields_present == 3 and abs(total - 100) <= 15:
                    psd_score = 0.4  # All fields present but off
                else:
                    psd_score = 0.3
        except (ValueError, TypeError):
            psd_score = 0.0
    else:
        # No PSD data present; score neutral (not penalized)
        psd_score = 0.5

    # 4. Metadata quality (20%)
    meta_filled = sum(1 for f in METADATA_FIELDS if row.get(f, "-") != "-")
    meta_score = meta_filled / len(METADATA_FIELDS)

    # Weighted average
    score = (
        0.40 * completeness
        + 0.20 * numeric_score
        + 0.20 * psd_score
        + 0.20 * meta_score
    )
    return round(score, 2)


# ===========================================================================
# COMPONENT 7 — Groq LLM Fallback
# ===========================================================================

class GroqFallback:
    """
    For low-confidence rows, re-extract data from raw page text
    using the Groq LLM API.
    
    Env vars:
        GROQ_API_KEY      — Your Groq API key
        GROQ_MODEL_NAME   — Model to use (default: llama-3.3-70b-versatile)
    """

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
        self.client = None
        if self.api_key and self.api_key != "gsk_your_api_key_here":
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info(f"Groq LLM fallback enabled (model: {self.model})")
            except ImportError:
                logger.warning("groq package not installed — LLM fallback disabled")
            except Exception as e:
                logger.warning(f"Groq client init failed: {e}")
        else:
            logger.info("GROQ_API_KEY not set — LLM fallback disabled (rule-based only)")

    def is_available(self) -> bool:
        return self.client is not None

    def extract_from_text(self, raw_page_text: str, sample_id: str) -> dict:
        """Send raw page text to Groq LLM for structured extraction."""
        if not self.is_available():
            return {}

        prompt = f"""You are a geotechnical data extraction expert.

Extract the following fields from this laboratory test report page text.
Return ONLY a valid JSON object with these exact keys.
Use "-" for any field not found in the text.

Required fields:
- "location": Borehole or test location ID (e.g., BH01, Bore 12)
- "sample_depth": Depth range in meters (e.g., "1.5-2.0")
- "soil_description": Soil type description with group symbol if present
- "geological_unit": Geological formation name if mentioned
- "pct_fines": Percentage passing 0.075mm sieve (number only)
- "pct_sand": Percentage of sand fraction 0.075-4.75mm (number only)
- "pct_gravel": Percentage of gravel fraction >4.75mm (number only)
- "ll": Liquid Limit percentage (number only)
- "pl": Plastic Limit percentage (number only)
- "pi": Plasticity Index = LL - PL (number only)
- "ls": Linear Shrinkage percentage (number only)

Sample ID for context: {sample_id}

--- PAGE TEXT ---
{raw_page_text[:4000]}
--- END ---

JSON output:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content
            result = json.loads(raw_content)
            # Normalize LLM output values
            for key in result:
                if isinstance(result[key], (int, float)):
                    result[key] = str(result[key])
                elif result[key] is None:
                    result[key] = "-"
            filled = sum(1 for v in result.values() if v != "-")
            logger.info(f"  LLM extracted {filled} fields for {sample_id}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"  LLM returned invalid JSON for {sample_id}: {e}")
            return {}
        except Exception as e:
            logger.error(f"  Groq API error for {sample_id}: {e}")
            return {}


def merge_with_llm_result(rule_based_row: dict, llm_row: dict) -> dict:
    """Fill missing fields from LLM result. Rule-based values take priority."""
    merged = rule_based_row.copy()
    for key, llm_value in llm_row.items():
        if key in merged and merged[key] == "-" and llm_value and llm_value != "-":
            merged[key] = normalize_numeric(llm_value) if key in NUMERIC_FIELDS else normalize_text(llm_value)
    return merged


# ===========================================================================
# COMPONENT 8 — Main Pipeline
# ===========================================================================

def process_pdf(pdf_path: str) -> list:
    """
    Main pipeline: parse PDF → classify → extract → normalize → aggregate
    → score → (optional LLM fallback) → return rows.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        sys.exit(1)

    logger.info(f"Processing: {pdf_path.name}")

    # -----------------------------------------------------------------------
    # Phase 1: Parse all pages, classify, and extract raw data
    # -----------------------------------------------------------------------
    # sample_data[sample_number] = { merged dict of all extracted fields }
    sample_data = defaultdict(lambda: {
        "sample_number": "-",
        "location": "-",
        "sample_depth": "-",
        "soil_description": "-",
        "geological_unit": "-",
        "pct_fines": "-",
        "pct_sand": "-",
        "pct_gravel": "-",
        "ll": "-",
        "pl": "-",
        "pi": "-",
        "ls": "-",
        "raw_page_texts": [],   # kept for LLM fallback
        "page_numbers": [],
    })

    with pdfplumber.open(str(pdf_path)) as pdf:
        logger.info(f"Total pages: {len(pdf.pages)}")

        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1
            text = page.extract_text(layout=True) or ""
            text_plain = page.extract_text() or ""

            # Classify page
            test_types = classify_page(text)
            if not test_types:
                continue

            # Skip non-soil test pages
            if any(t in test_types for t in ["ucs", "aggressivity"]):
                logger.debug(f"  Page {page_num}: skipping non-soil test ({test_types})")
                continue

            # Extract metadata from header
            meta = extract_metadata(text)
            sample_id = meta["sample_number"]
            if sample_id == "-":
                # Try to get from plain text (sometimes layout breaks it)
                meta2 = extract_metadata(text_plain)
                if meta2["sample_number"] != "-":
                    sample_id = meta2["sample_number"]
                    meta = meta2

            if sample_id == "-":
                logger.debug(f"  Page {page_num}: no sample number found, skipping")
                continue

            logger.info(
                f"  Page {page_num}: Sample={sample_id}, "
                f"Tests={test_types}"
            )

            # Update sample record with metadata (don't overwrite existing non-null)
            rec = sample_data[sample_id]
            rec["sample_number"] = sample_id
            # Map metadata keys to sample record keys
            meta_key_map = {
                "location": "location",
                "depth": "sample_depth",
                "soil_description": "soil_description",
                "geological_unit": "geological_unit",
            }
            for meta_key, rec_key in meta_key_map.items():
                if rec[rec_key] == "-" and meta.get(meta_key, "-") != "-":
                    rec[rec_key] = meta[meta_key]

            # Store raw text for LLM fallback
            rec["raw_page_texts"].append(text_plain)
            rec["page_numbers"].append(page_num)

            # Extract tables
            tables = page.extract_tables() or []

            # --- PSD extraction ---
            if "psd" in test_types:
                psd = extract_psd(tables)
                for key in ["pct_fines", "pct_sand", "pct_gravel"]:
                    if rec[key] == "-" and psd[key] != "-":
                        rec[key] = validate_percentage(psd[key], key)

            # --- Atterberg / Linear Shrinkage extraction ---
            if "atterberg" in test_types or "linear_shrinkage" in test_types:
                att = extract_atterberg(tables)
                for key in ["ll", "pl", "pi", "ls"]:
                    if rec[key] == "-" and att[key] != "-":
                        rec[key] = validate_percentage(att[key], key)

            # --- Try extracting soil description from any test table (not just Emerson) ---
            # FIX: Expanded search from "emerson" test pages to ALL test pages with tables
            # Soil descriptions often appear in PSD, Atterberg, CBR, and Emerson test tables
            desc_synonyms = [
                "soil description", "material description", "material",
                "sample description", "description of", "soil type", 
                "material type", "visual classification", "description",
                "soil name", "sample name", "classification",
            ]
            # First try to find description in table rows if not already captured from header
            if rec["soil_description"] == "-":
                for table in tables:
                    for row in (table or []):
                        if row and len(row) >= 2:
                            label = str(row[0]).strip().lower() if row[0] else ""
                            value = str(row[1]).strip() if row[1] else ""
                            # Check if this row is a soil description row
                            if any(syn in label for syn in desc_synonyms) and value and value != "-":
                                desc = normalize_soil_description(value)
                                if desc != "-":
                                    rec["soil_description"] = desc
                                    logger.debug(f"  Page {page_num}: Extracted description from table: {desc}")
                                    break
                    if rec["soil_description"] != "-":
                        break

            # --- Try to extract location from pinhole test (multi-bore tables) ---
            if "pinhole" in test_types:
                _extract_pinhole_locations(tables, sample_data, text_plain)

    # -----------------------------------------------------------------------
    # Phase 2: Confidence scoring & LLM fallback
    # -----------------------------------------------------------------------
    groq = GroqFallback()

    rows = []
    for sample_id, rec in sample_data.items():
        # Build output row
        row = {k: rec[k] for k in ALL_FIELDS}
        row["location"] = rec["location"] if rec["location"] != "-" else sample_id

        # Append group symbol to soil description if found
        symbol = extract_group_symbol(rec.get("soil_description", ""))
        if symbol:
            # If description already contains the symbol, don't duplicate
            if f"({symbol})" not in row.get("soil_description", ""):
                row["soil_description"] = f"{row['soil_description']} ({symbol})"

        # Score confidence
        confidence = compute_confidence(row)
        extraction_method = "rule-based"

        if confidence >= 0.8:
            extraction_method = "rule-based"
        elif confidence >= CONFIDENCE_THRESHOLD:
            extraction_method = "rule-based (low)"
            logger.warning(
                f"  {sample_id}: confidence={confidence} (below 0.8, above threshold)"
            )
        else:
            # FIX: Check if we have valid PSD data before triggering LLM fallback
            # If PSD sums to ~100, don't trigger LLM even if other fields are missing
            psd_valid = False
            try:
                psd_fields = [float(row.get(k, "-")) for k in ["pct_fines", "pct_sand", "pct_gravel"] if row.get(k, "-") != "-"]
                if psd_fields and len(psd_fields) >= 2:  # At least 2 of 3 PSD fields
                    psd_total = sum(psd_fields)
                    if abs(psd_total - 100) <= 5:  # Within 5% of 100
                        psd_valid = True
            except (ValueError, TypeError):
                pass
            
            # Trigger LLM fallback
            logger.warning(
                f"  {sample_id}: confidence={confidence} (below threshold={CONFIDENCE_THRESHOLD})"
            )
            if groq.is_available() and rec.get("raw_page_texts") and not psd_valid:
                # Only use LLM if PSD is invalid/missing
                combined_text = "\n\n---PAGE BREAK---\n\n".join(rec["raw_page_texts"])
                llm_result = groq.extract_from_text(combined_text, sample_id)
                if llm_result:
                    row = merge_with_llm_result(row, llm_result)
                    new_confidence = compute_confidence(row)
                    if new_confidence > confidence:
                        confidence = new_confidence
                        extraction_method = "llm-enhanced"
                        logger.info(
                            f"  {sample_id}: LLM improved confidence "
                            f"{confidence - new_confidence:.2f} → {new_confidence}"
                        )
                    else:
                        extraction_method = "llm-fallback"
                else:
                    extraction_method = "rule-based (low)"
            else:
                extraction_method = "rule-based (low)"

        row["confidence"] = confidence
        row["extraction_method"] = extraction_method
        row["pages"] = ", ".join(str(p) for p in rec.get("page_numbers", []))
        rows.append(row)

    # Sort by location/sample_id
    rows.sort(key=lambda r: r.get("location", ""))
    logger.info(f"Extracted {len(rows)} sample rows")
    return rows


def _extract_pinhole_locations(tables, sample_data, text):
    """
    Special handler for pinhole dispersion test pages that contain
    multi-bore summary tables (e.g., page 30 of sample PDF).
    These tables have multiple bores/depths in a single table row.
    """
    for table in tables:
        if not table:
            continue
        for row in table:
            if not row or len(row) < 4:
                continue
            label = str(row[0]).strip().lower() if row[0] else ""
            if "test" in label and "location" in label:
                # Multi-line cell: "Bore 29\nBore 19\nBore 32..."
                locations_str = str(row[0]) if row[0] else ""
                depths_str = str(row[1]) if row[1] else ""
                descs_str = str(row[3]) if len(row) > 3 and row[3] else ""

                locations = re.findall(r"Bore\s*(\d+)", locations_str, re.I)
                depths = re.findall(r"[\d.]+-[\d.]+|[\d.]+", depths_str)
                descs = [d.strip() for d in descs_str.split("\n") if d.strip()]

                for i, loc_num in enumerate(locations):
                    loc_id = f"BH{loc_num}"
                    # Find matching sample in existing data
                    for sid, rec in sample_data.items():
                        if rec["location"] == loc_id:
                            if i < len(depths) and rec["sample_depth"] == "-":
                                rec["sample_depth"] = normalize_depth(depths[i])
                            if i < len(descs) and rec["soil_description"] == "-":
                                rec["soil_description"] = normalize_soil_description(descs[i])


# ===========================================================================
# COMPONENT 9 — Output Generator
# ===========================================================================

# Column display names matching the expected table format
OUTPUT_COLUMNS = {
    "location": "Location",
    "sample_depth": "Sample depth (m)",
    "soil_description": "Soil description (group symbol)",
    "geological_unit": "Geological unit",
    "pct_fines": "% Fines",
    "pct_sand": "% Sand",
    "pct_gravel": "% Gravel",
    "ll": "LL (%)",
    "pl": "PL (%)",
    "pi": "PI (%)",
    "ls": "LS (%)",
    "confidence": "Confidence",
    "extraction_method": "Extraction Method",
}


def output_table(rows: list, output_dir: str = "output", formats: list = None):
    """Write extraction results to CSV, JSON, and console."""
    if formats is None:
        formats = ["csv", "json", "console"]

    os.makedirs(output_dir, exist_ok=True)

    # Build display DataFrame
    display_rows = []
    for row in rows:
        display_row = {}
        for key, col_name in OUTPUT_COLUMNS.items():
            display_row[col_name] = row.get(key, "-")
        display_rows.append(display_row)

    df = pd.DataFrame(display_rows)

    # --- CSV output (write first - most reliable) ---
    if "csv" in formats:
        csv_path = os.path.join(output_dir, "extracted_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"CSV saved: {csv_path}")

    # --- JSON output ---
    if "json" in formats:
        json_path = os.path.join(output_dir, "extracted_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON saved: {json_path}")

    # --- Console output (last - may fail on Windows encoding) ---
    if "console" in formats:
        try:
            print("\n" + "=" * 120)
            print("EXTRACTION RESULTS - Summary of Geotechnical Laboratory Test Results")
            print("=" * 120)
            print(df.to_string(index=False))
            print(f"\nTotal rows: {len(rows)}")

            # Summary statistics
            confidences = [r["confidence"] for r in rows if isinstance(r.get("confidence"), (int, float))]
            if confidences:
                print(f"Confidence: min={min(confidences)}, max={max(confidences)}, avg={sum(confidences)/len(confidences):.2f}")
                low_conf = sum(1 for c in confidences if c < CONFIDENCE_THRESHOLD)
                if low_conf:
                    print(f"[!] {low_conf} row(s) below confidence threshold ({CONFIDENCE_THRESHOLD})")
            print("=" * 120 + "\n")
        except UnicodeEncodeError:
            logger.warning("Console output failed due to encoding. Check CSV/JSON output files.")

    return df


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def main():
    global CONFIDENCE_THRESHOLD

    parser = argparse.ArgumentParser(
        description="Extract geotechnical lab data from PDF reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_geotech.py report.pdf
  python extract_geotech.py report.pdf --output-dir ./results
  python extract_geotech.py report.pdf --format csv json
        """,
    )
    parser.add_argument("pdf", help="Path to the PDF report file")
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory for output files (default: ./output)"
    )
    parser.add_argument(
        "--format", nargs="+", default=["csv", "json", "console"],
        choices=["csv", "json", "console"],
        help="Output formats (default: csv json console)"
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=None,
        help="Override confidence threshold (env default: 0.6)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.confidence_threshold is not None:
        CONFIDENCE_THRESHOLD = args.confidence_threshold

    # Run the pipeline
    rows = process_pdf(args.pdf)

    if not rows:
        logger.warning("No data extracted from the PDF.")
        sys.exit(0)

    # Output results
    output_table(rows, output_dir=args.output_dir, formats=args.format)

    logger.info("Done.")


if __name__ == "__main__":
    main()
