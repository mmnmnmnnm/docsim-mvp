from __future__ import annotations

from typing import Any, Dict, List


def _format_number(value: float) -> str:
    """
    Number formater

    """
    abs_val = abs(value)
    if abs_val >= 1_000_000:
        millions = value / 1_000_000
        return f"{millions:.1f} M"
    if abs_val.is_integer():
        return f"{int(value)}"
    return f"{value:.2f}"


def _format_list(values: List[Any], max_items: int = 10) -> str:
    """
    Short list
    """
    if not values:
        return "-"
    if len(values) > max_items:
        shown = values[:max_items]
        return f"{', '.join(map(str, shown))} ... (+{len(values) - max_items} Others)"
    return ", ".join(map(str, values))


def _format_similarity_block(result: Dict[str, Any]) -> str:
    sim = result.get("similarity", None)
    numeric = result.get("numeric", {})
    num_sim = numeric.get("numeric_similarity", None)

    lines: List[str] = []
    lines.append("Textual and numeric values")

    if sim is not None:
        lines.append(f"- Textual similarity (cosine): {sim:.4f}")

    if num_sim is not None:
        lines.append(f"- Numerical similarity (Jaccard): {num_sim:.4f}")

    return "\n".join(lines)


def _format_keyword_block(result: Dict[str, Any], max_items: int = 15) -> str:
    keywords = result.get("keywords", {})

    common = keywords.get("common", [])
    missing = keywords.get("missing", [])
    added = keywords.get("added", [])

    lines = ["Keywords comparison"]

    def _fmt(title: str, items: list[str]) -> None:
        if items:
            lines.append(f"- {title}: {', '.join(items[:max_items])}")
        else:
            lines.append(f"- {title}: none")

    _fmt("Common", common)
    _fmt("Missing", missing)
    _fmt("Added", added)

    return "\n".join(lines)

    lines: List[str] = []
    lines.append("Keywords")
    lines.append(f"- Common keywords: { _format_list(common, max_items=max_items) }")
    lines.append(f"- Missing (Only from the reference): { _format_list(missing, max_items=max_items) }")
    lines.append(f"- Added (Only in the generated): { _format_list(added, max_items=max_items) }")

    return "\n".join(lines)


def _format_numeric_type_block(kind: str, stats: Dict[str, Any]) -> str:
    """
    Overview of num types.
    """
    ref_vals = stats.get("ref_numeric_values", [])
    gen_vals = stats.get("gen_numeric_values", [])
    common = stats.get("numeric_common", [])
    missing = stats.get("numeric_missing", [])
    added = stats.get("numeric_added", [])
    sim = stats.get("numeric_similarity", None)

    # Formatting the numbers
    ref_fmt = [_format_number(v) for v in ref_vals]
    gen_fmt = [_format_number(v) for v in gen_vals]
    common_fmt = [_format_number(v) for v in common]
    missing_fmt = [_format_number(v) for v in missing]
    added_fmt = [_format_number(v) for v in added]

    kind_label_map = {
        "amount": "Ammounts",
        "percent": "Percentages",
        "year": "Dates",
        "other": "Others",
    }
    label = kind_label_map.get(kind, kind)

    lines: List[str] = []
    lines.append(f"- {label}:")
    if sim is not None:
        lines.append(f"    - Similarity: {sim:.4f}")
    if ref_vals or gen_vals:
        lines.append(f"    - Reference values: { _format_list(ref_fmt) }")
        lines.append(f"    - Generated values:   { _format_list(gen_fmt) }")
        lines.append(f"    - Common:              { _format_list(common_fmt) }")
        lines.append(f"    - Only reference:    { _format_list(missing_fmt) }")
        lines.append(f"    - Only generated:      { _format_list(added_fmt) }")
    else:
        lines.append("    * No numerical values in the documents.")

    return "\n".join(lines)


def _format_numeric_block(result: Dict[str, Any]) -> str:
    numeric = result.get("numeric", {})
    if not numeric:
        return "Numerical similarity\nN/A."

    lines: List[str] = []
    lines.append("Numerical similarity")

    overall_sim = numeric.get("numeric_similarity", None)
    if overall_sim is not None:
        lines.append(f"- Overall numerical similaritiy: {overall_sim:.4f}")

    by_type = numeric.get("by_type", {})
    # Order
    for kind in ["amount", "percent", "year", "other"]:
        stats = by_type.get(kind)
        if not stats:
            continue

        # Shown only if there atr 1 value in the ref/gen side
        if stats.get("ref_numeric_values") or stats.get("gen_numeric_values"):
            lines.append("")
            lines.append(_format_numeric_type_block(kind, stats))

    return "\n".join(lines)


def _format_short_conclusion(result: Dict[str, Any]) -> str:
    """
    overview
    """
    sim = result.get("similarity", None)
    numeric = result.get("numeric", {})
    num_sim = numeric.get("numeric_similarity", None)

    lines: List[str] = []
    lines.append("Summary")

    text_part = ""
    if sim is not None:
        if sim > 0.9:
            text_part = "The two documents are very similar in terms of wording."
        elif sim > 0.7:
            text_part = "The two documents are similar in wording, but there are differences."
        else:
            text_part = "The two documents differ significantly in terms of wording.."
        lines.append(f"- Textual: {text_part} (cosine={sim:.4f})")

    if num_sim is not None:
        if num_sim > 0.8:
            num_part = "The numerical content is almost identical."
        elif num_sim > 0.5:
            num_part = "The numerical content is partly the same, but there are significant differences."
        else:
            num_part = "The numerical content differs significantly."
        lines.append(f"- Based on the numers: {num_part} (numeric_similarity={num_sim:.4f})")

    if not lines:
        return "Short summary\nN/A."

    return "\n".join(lines)


def format_report(result: Dict[str, Any]) -> str:
    """
    things from raw output dict

    """
    blocks: List[str] = []

    blocks.append(_format_similarity_block(result))
    blocks.append("")
    blocks.append(_format_keyword_block(result))
    blocks.append("")
    blocks.append(_format_numeric_block(result))
    blocks.append("")
    blocks.append(_format_short_conclusion(result))

    return "\n".join(blocks)
