from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .preprocessing import preprocess_text, tokenize_nlp
from .embedding import embed_text
from .metrics import (
    cosine_similarity_score,
    extract_keywords,
    extract_numeric_tokens_by_type,
    compute_numeric_similarity,
)
from .metrics_advanced import (
    compute_rouge_scores,
    compute_bert_scores,
    compute_composite_index,
)


# Chunking starts here

def _chunk_text_by_words(text: str, max_words: int = 220) -> List[str]:
    """
    Simple deterministic chunking
    max_words=220
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _length_penalty(n_ref: int, n_gen: int) -> float:
    """
    Lenght-penalty: If a page is longer, reduce the chunks to the average
    """
    if n_ref == 0 and n_gen == 0:
        return 1.0
    if n_ref == 0 or n_gen == 0:
        return 0.0
    return min(n_ref, n_gen) / max(n_ref, n_gen)


def _chunkwise_rougeL(ref_chunks: List[str], gen_chunks: List[str]) -> float:
    """
    ROUGE-L by chunks
    """
    if not ref_chunks and not gen_chunks:
        return 1.0
    if not ref_chunks or not gen_chunks:
        return 0.0

    m = min(len(ref_chunks), len(gen_chunks))
    scores = [compute_rouge_scores(ref_chunks[i], gen_chunks[i])["rougeL_f1"] for i in range(m)]
    avg = float(np.mean(scores)) if scores else 0.0
    return avg * _length_penalty(len(ref_chunks), len(gen_chunks))


def _chunkwise_bertscore(ref_chunks: List[str], gen_chunks: List[str]) -> Dict[str, float]:
    """
    BERTScore by chunks
    """
    if not ref_chunks and not gen_chunks:
        return {"bertscore_precision": 1.0, "bertscore_recall": 1.0, "bertscore_f1": 1.0}
    if not ref_chunks or not gen_chunks:
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}

    m = min(len(ref_chunks), len(gen_chunks))
    ps: List[float] = []
    rs: List[float] = []
    fs: List[float] = []

    for i in range(m):
        s = compute_bert_scores(ref_chunks[i], gen_chunks[i])
        ps.append(s["bertscore_precision"])
        rs.append(s["bertscore_recall"])
        fs.append(s["bertscore_f1"])

    penalty = _length_penalty(len(ref_chunks), len(gen_chunks))

    return {
        "bertscore_precision": float(np.mean(ps)) * penalty if ps else 0.0,
        "bertscore_recall": float(np.mean(rs)) * penalty if rs else 0.0,
        "bertscore_f1": float(np.mean(fs)) * penalty if fs else 0.0,
    }


def _embed_document_chunked(text: str, max_words: int = 220) -> np.ndarray:
    """
    Document embedding by chunks
    - embedding chunks
    - averaging
    """
    chunks = _chunk_text_by_words(text, max_words=max_words)
    if not chunks:
        return np.zeros((384,), dtype=float)

    vecs = [embed_text(ch) for ch in chunks]
    mat = np.vstack([v.reshape(1, -1) for v in vecs])
    return mat.mean(axis=0)


# This is the api

def compare_documents(ref_text: str, gen_text: str) -> Dict[str, Any]:
    """
    - preprocess
    - tokenization
    - chunking
    - chunked embedding + cosine
    - chunked ROUGE-L + chunked BERTScore
    - numerical extraction + Jaccard
    - composite textual + overall + numeric veto + BERTScore cap
    """

    # 1) Preprocess
    ref_clean = preprocess_text(ref_text)
    gen_clean = preprocess_text(gen_text)

    # 2) Tokenization
    ref_tokens: List[str] = tokenize_nlp(ref_clean)
    gen_tokens: List[str] = tokenize_nlp(gen_clean)

    # 3) Chunking
    ref_chunks = _chunk_text_by_words(ref_clean, max_words=220)
    gen_chunks = _chunk_text_by_words(gen_clean, max_words=220)

    # 4) Embedding + cosine (chunked doc embedding)
    ref_vec = _embed_document_chunked(ref_clean, max_words=220)
    gen_vec = _embed_document_chunked(gen_clean, max_words=220)
    cosine = float(cosine_similarity_score(ref_vec, gen_vec))

    # 5) ROUGE-L (chunked)
    rougeL_f1 = float(_chunkwise_rougeL(ref_chunks, gen_chunks))

    # 6) BERTScore (chunked)
    bert = _chunkwise_bertscore(ref_chunks, gen_chunks)
    bert_f1 = float(bert["bertscore_f1"])

    # 7) Keyword (Not neccessary, but I leaved in here)
    ref_kw = extract_keywords(ref_tokens, top_n=50)
    gen_kw = extract_keywords(gen_tokens, top_n=50)

    common_kw = sorted(set(ref_kw).intersection(gen_kw))
    missing_kw = sorted(set(ref_kw) - set(gen_kw))
    added_kw = sorted(set(gen_kw) - set(ref_kw))

    # 8) Numerical data
    ref_nums = extract_numeric_tokens_by_type(ref_clean)
    gen_nums = extract_numeric_tokens_by_type(gen_clean)

    numeric_summary = compute_numeric_similarity(ref_nums, gen_nums)
    numeric_jaccard = float(numeric_summary["numeric_similarity"])

    # 9) composite index + veto + BERT cap
    composite = compute_composite_index(
        cosine=cosine,
        rougeL_f1=rougeL_f1,
        bert_f1=bert_f1,
        numeric_jaccard=numeric_jaccard,
        bert_cap=0.85,
        numeric_veto_threshold=0.20,
        numeric_veto_overall_cap=0.40,
    )

    return {
        "similarity": cosine,
        "rouge": {"rougeL_f1": rougeL_f1},
        "bertscore": bert,
        "numeric": numeric_summary,
        "keywords": {
            "common": common_kw,
            "missing": missing_kw,
            "added": added_kw,
        },
        "scores": {
            "cosine": cosine,
            "rougeL_f1": rougeL_f1,
            "bertscore_f1": bert_f1,
            "numeric_jaccard": numeric_jaccard,
            "textual_index": composite["textual_index"],
            "overall_index": composite["overall_index"],
            "bertscore_f1_capped": composite["bertscore_f1_capped"],
        },
        "debug": {
            "ref_chunks": len(ref_chunks),
            "gen_chunks": len(gen_chunks),
            "chunk_words": 220,
            "length_penalty": _length_penalty(len(ref_chunks), len(gen_chunks)),
        },
    }


def compare_document_files(ref_path: str | Path, gen_path: str | Path) -> Dict[str, Any]:
    """
    Wrapper: comparing the 2 .txt file
    """
    ref_path = Path(ref_path)
    gen_path = Path(gen_path)

    ref_text = ref_path.read_text(encoding="utf-8")
    gen_text = gen_path.read_text(encoding="utf-8")

    return compare_documents(ref_text, gen_text)
