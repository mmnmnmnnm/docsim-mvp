from __future__ import annotations

from typing import Dict

from bert_score import score as bert_score
from rouge_score import rouge_scorer


# Rogue-L

def compute_rouge_scores(ref_text: str, gen_text: str) -> Dict[str, float]:
    """
    ROUGE-L F1: Longest Common Subsequence
    Return:
        {"rougeL_f1": float}
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(ref_text, gen_text)
    return {"rougeL_f1": float(scores["rougeL"].fmeasure)}


# BERTScore

def compute_bert_scores(ref_text: str, gen_text: str) -> Dict[str, float]:
    """
    BERTScore (P/R/F1) token-level

    Return:
        {"bertscore_precision": float, "bertscore_recall": float, "bertscore_f1": float}
    """
    P, R, F1 = bert_score(
        [gen_text],
        [ref_text],
        lang=None,
        model_type="distilbert-base-multilingual-cased",
        verbose=False,
        rescale_with_baseline=False,
    )
    return {
        "bertscore_precision": float(P[0]),
        "bertscore_recall": float(R[0]),
        "bertscore_f1": float(F1[0]),
    }


# Composite

def compute_composite_index(
    cosine: float,
    rougeL_f1: float,
    bert_f1: float,
    numeric_jaccard: float,
    *,
    bert_cap: float = 0.85,
    numeric_veto_threshold: float = 0.20,
    numeric_veto_overall_cap: float = 0.40,
) -> Dict[str, float]:
    """
    Calculating composite indexes

    - BERTScore cap: bert_f1 = min(bert_f1, bert_cap)

    - Textual index weighting (ROUGE more imoprtant):
        textual_index = 0.30 * cosine + 0.25 * bert_f1 + 0.45 * rougeL_f1

    - Overall index:
        overall_index = 0.70 * textual_index + 0.30 * numeric_jaccard

    - Numeric veto (Business rule):
        if numeric_jaccard < 0.20 -> overall_index max 0.40
    """

    bert_f1_capped = min(float(bert_f1), float(bert_cap))

    textual_index = (
        0.30 * float(cosine) +
        0.25 * bert_f1_capped +
        0.45 * float(rougeL_f1)
    )

    overall_index = (
        0.70 * float(textual_index) +
        0.30 * float(numeric_jaccard)
    )

    # Business veto
    if float(numeric_jaccard) < float(numeric_veto_threshold):
        overall_index = min(float(overall_index), float(numeric_veto_overall_cap))

    return {
        "textual_index": float(textual_index),
        "overall_index": float(overall_index),
        "bertscore_f1_capped": float(bert_f1_capped),
    }
