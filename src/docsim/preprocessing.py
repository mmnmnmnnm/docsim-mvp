# This is the preprocessing + tokenization module v1.0
# Imports:
from __future__ import annotations

from typing import List
import re

from transformers import AutoTokenizer

# Multilingual NLP tokenization for HU/SI/EN inputs.
_TOKENIZER_NAME = "xlm-roberta-base"
_tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)


def preprocess_text(text: str) -> str:
    """
    Simple preprocessing:
    - whitespace normalization
    - lower()
    - Remove the control characters
    """
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    return text


def tokenize_nlp(text: str) -> List[str]:
    """
    NLP tokenization with XLM-R
    """
    if not text:
        return []

    encoded = _tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )

    token_ids = encoded["input_ids"]
    tokens = _tokenizer.convert_ids_to_tokens(token_ids)

    # remove prefix
    cleaned = [tok.lstrip("▁") for tok in tokens if tok.strip("▁").strip()]

    return cleaned
