"""Text Preprocessing Module for Sentiment Analysis.

This module provides separate text preprocessing pipelines for:
1. Traditional Machine Learning / TF-IDF baselines.
2. Modern Transformer models (DistilBERT).

Key Conceptual Difference:
- Traditional NLP (TF-IDF): Relies on bag-of-words / n-grams. Lowercasing,
  removing special characters, and stripping HTML reduce vocabulary sparsity.
- Transformer NLP (DistilBERT): Relies on bidirectional self-attention and
  subword WordPiece tokenization. Aggressive cleaning (e.g. removing punctuation,
  stripping negation words like 'not', 'never', 'barely', or aggressive stemming)
  destroys semantic context and syntax. DistilBERT only needs HTML removal and
  whitespace normalization.
"""

import re
import html
import logging
from typing import List, Union, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Regular expression to identify HTML tags (e.g. <br />, <p>, etc.)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
# Multiple whitespace pattern
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_html(text: str) -> str:
    """Removes HTML tags and unescapes HTML entities from text.

    Args:
        text: Raw input string possibly containing HTML markup.

    Returns:
        Cleaned text string without HTML tags.
    """
    if not isinstance(text, str):
        return ""
    # Unescape HTML entities like &quot;, &amp;, &lt;
    unescaped = html.unescape(text)
    # Remove HTML tags like <br />
    no_html = HTML_TAG_PATTERN.sub(" ", unescaped)
    # Remove artificial space preceding punctuation introduced by stripped inline tags
    no_html = re.sub(r"\s+([.,!?:;])", r"\1", no_html)
    return no_html


def normalize_whitespace(text: str) -> str:
    """Normalizes excessive spaces, tabs, and newline characters into single spaces.

    Args:
        text: Input string.

    Returns:
        Stripped string with single whitespace separators.
    """
    if not isinstance(text, str):
        return ""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def validate_text(text: Any) -> str:
    """Validates and coerces input text into a sanitized string.

    Args:
        text: Any input to be verified as valid text.

    Returns:
        Sanitized non-null string.

    Raises:
        ValueError: If input is None or evaluates to empty after stripping.
    """
    if text is None:
        raise ValueError("Input text cannot be None.")
    if not isinstance(text, str):
        text = str(text)
    sanitized = text.strip()
    if not sanitized:
        raise ValueError("Input text cannot be empty or solely whitespace.")
    return sanitized


def preprocess_for_transformer(text: str) -> str:
    """Minimal, context-preserving preprocessing suitable for Transformer models.

    Preserves:
    - Casing and capitalization (which carry emphasis/sentiment intensity).
    - Negation words ('not', 'hardly', 'scarcely', 'no').
    - Punctuation (e.g., '!', '?', which signify strong sentiment).

    Cleans:
    - HTML markup (e.g., <br />, <div>).
    - Erratic whitespace and irregular newlines.

    Args:
        text: Raw review text.

    Returns:
        Cleaned text retaining syntax and semantic indicators.
    """
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    cleaned = clean_html(text)
    # Normalize whitespaces
    cleaned = normalize_whitespace(cleaned)
    return cleaned


def preprocess_for_tfidf(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = False
) -> str:
    """Standard text normalization pipeline for TF-IDF / Logistic Regression.

    Performs HTML tag stripping, whitespace normalization, and optional lowercasing.
    Crucially avoids removing negation words ('not', 'never', 'no') which are vital
    for sentiment detection even in bag-of-words models.

    Args:
        text: Raw review text.
        lowercase: Whether to convert text to lowercase (default: True).
        remove_punctuation: Whether to strip punctuation characters (default: False).

    Returns:
        Preprocessed text string ready for vectorization.
    """
    if not isinstance(text, str):
        return ""
    cleaned = clean_html(text)
    cleaned = normalize_whitespace(cleaned)
    if lowercase:
        cleaned = cleaned.lower()
    if remove_punctuation:
        cleaned = re.sub(r"[^\w\s]", "", cleaned)
        cleaned = normalize_whitespace(cleaned)
    return cleaned


def tokenize_for_distilbert(
    texts: Union[str, List[str]],
    tokenizer: Any,
    max_length: int = 256,
    padding: Union[bool, str] = "max_length",
    truncation: bool = True,
    return_tensors: Optional[str] = "pt"
) -> Dict[str, Any]:
    """Tokenizes text using the pretrained DistilBERT tokenizer.

    Produces input_ids and attention_mask tensors required by DistilBERT.

    Args:
        texts: A single text string or list of text strings.
        tokenizer: Pretrained Hugging Face DistilBertTokenizer or DistilBertTokenizerFast.
        max_length: Maximum sequence token length.
        padding: Padding strategy ('max_length', 'longest', or True/False).
        truncation: Whether to truncate sequences longer than max_length.
        return_tensors: Tensor format to return ('pt' for PyTorch, 'np' for NumPy, None for lists).

    Returns:
        Dictionary containing 'input_ids' and 'attention_mask'.
    """
    if isinstance(texts, str):
        texts = [texts]

    # Preprocess all inputs minimally to preserve context
    cleaned_texts = [preprocess_for_transformer(t) for t in texts]

    encoded = tokenizer(
        cleaned_texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
        return_attention_mask=True
    )
    return encoded
