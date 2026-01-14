from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List
import math
import re

import numpy as np



def cosine_similarity_score(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Cosine similarity between the two 1D vector

    """
    v1 = np.asarray(vec1, dtype=float).ravel()
    v2 = np.asarray(vec2, dtype=float).ravel()

    if v1.size == 0 or v2.size == 0:
        return 0.0

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    sim = float(np.dot(v1, v2) / (norm1 * norm2))
    return sim


def extract_keywords(tokens: Iterable[str], top_n: int = 20) -> List[str]:
    """
    keyword extraction based on appearences
    """
    filtered = [t for t in tokens if not t.isdigit()]
    counter = Counter(filtered)
    if not counter:
        return []

    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [token for token, _count in items[:top_n]]


# Numerical extraction

_AMOUNT_PATTERN = re.compile(
    r"(\d[\d\s\u00A0]*)(?=\s*(huf|ft|forint|eur|usd))",
    flags=re.IGNORECASE
)

_PERCENT_PATTERN = re.compile(
    r"(\d[\d\s\u00A0]*([.,]\d+)?)(?=\s*%)",
    flags=re.IGNORECASE
)

_YEAR_PATTERN = re.compile(
    r"\b(19[0-9]{2}|20[0-9]{2}|2100)\b"
)


_GENERIC_NUMBER_PATTERN = re.compile(
    r"\d[\d\s\u00A0]*([.,]\d+)?"
)


def _normalize_number_string(raw: str) -> float | None:
    """
    NRaw num string to float

    - removing whitespaces
    - ',' -> '.' switcvh
    """
    cleaned = raw.replace("\u00A0", " ")
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _span_overlaps(span: tuple[int, int], used_spans: List[tuple[int, int]]) -> bool:
    """helper to check the span overlay"""
    s1, e1 = span
    for s2, e2 in used_spans:
        if not (e1 <= s2 or e2 <= s1):
            return True
    return False


def extract_numeric_tokens_by_type(text: str) -> Dict[str, List[float]]:
    """
    Extracting numerical values and categorize

    Categories:
        - amounts: (HUF, FT, FORINT, EUR, USD)
        - percent: (num + '%')
        - year: 4 character numers between 1900-2100
        - other: Every other number

    Return:
        {
          "amount": [...],
          "percent": [...],
          "year": [...],
          "other": [...]
        }
    """
    result: Dict[str, List[float]] = {
        "amount": [],
        "percent": [],
        "year": [],
        "other": [],
    }

    used_spans: List[tuple[int, int]] = []

    # 1) Amounts
    for match in _AMOUNT_PATTERN.finditer(text):
        span = match.span(1)
        if _span_overlaps(span, used_spans):
            continue
        num_str = match.group(1)
        value = _normalize_number_string(num_str)
        if value is not None:
            result["amount"].append(value)
            used_spans.append(span)

    # 2) Percentages
    for match in _PERCENT_PATTERN.finditer(text):
        span = match.span(1)
        if _span_overlaps(span, used_spans):
            continue
        num_str = match.group(1)
        value = _normalize_number_string(num_str)
        if value is not None:
            result["percent"].append(value)
            used_spans.append(span)

    # 3) Dates
    for match in _YEAR_PATTERN.finditer(text):
        span = match.span(1)
        if _span_overlaps(span, used_spans):
            continue
        num_str = match.group(1)
        value = _normalize_number_string(num_str)
        if value is not None:
            result["year"].append(value)
            used_spans.append(span)

    # 4) Other numbers
    for match in _GENERIC_NUMBER_PATTERN.finditer(text):
        span = match.span(0)
        if _span_overlaps(span, used_spans):
            continue
        num_str = match.group(0)
        value = _normalize_number_string(num_str)
        if value is not None:
            result["other"].append(value)
            used_spans.append(span)

    return result


# Numerical similarity

def _compute_numeric_stats_for_values(ref_vals: List[float], gen_vals: List[float]) -> Dict[str, Any]:
    """
    Jaccard-similarity

    """
    def _to_rounded_set(values: List[float]) -> set[float]:
        return {round(v, 6) for v in values}

    ref_set = _to_rounded_set(ref_vals)
    gen_set = _to_rounded_set(gen_vals)

    union = ref_set | gen_set
    inter = ref_set & gen_set

    if not union:
        similarity = 1.0
    else:
        similarity = len(inter) / len(union)

    numeric_common = sorted(inter)
    numeric_missing = sorted(ref_set - gen_set)
    numeric_added = sorted(gen_set - ref_set)

    return {
        "numeric_similarity": float(similarity),
        "ref_numeric_values": sorted(ref_set),
        "gen_numeric_values": sorted(gen_set),
        "numeric_common": numeric_common,
        "numeric_missing": numeric_missing,
        "numeric_added": numeric_added,
    }


def compute_numeric_similarity(
    ref_nums: Dict[str, List[float]],
    gen_nums: Dict[str, List[float]],
) -> Dict[str, Any]:
    """
    Summarizing the numerical outputs

    Input:
        ref_nums: {"amount": [...], "percent": [...], "year": [...], "other": [...]}
        gen_nums: same

    Output:
        {
          "numeric_similarity": float,
          "ref_numeric_values": [...],
          "gen_numeric_values": [...],
          "numeric_common": [...],
          "numeric_missing": [...],
          "numeric_added": [...],
          "by_type": {
              "amount": { ... },
              "percent": { ... },
              "year": { ... },
              "other": { ... },
          }
        }
    """
    all_types = sorted(set(ref_nums.keys()) | set(gen_nums.keys()))

    by_type: Dict[str, Any] = {}
    all_ref: List[float] = []
    all_gen: List[float] = []

    for kind in all_types:
        ref_vals = ref_nums.get(kind, []) or []
        gen_vals = gen_nums.get(kind, []) or []

        all_ref.extend(ref_vals)
        all_gen.extend(gen_vals)

        stats = _compute_numeric_stats_for_values(ref_vals, gen_vals)
        by_type[kind] = stats

    overall_stats = _compute_numeric_stats_for_values(all_ref, all_gen)
    overall_stats["by_type"] = by_type
    return overall_stats
