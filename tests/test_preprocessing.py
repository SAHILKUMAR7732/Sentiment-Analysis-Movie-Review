"""Unit tests for text preprocessing functions.

Tests text sanitization, HTML stripping, whitespace normalization,
negation preservation, and validation logic.
"""

import pytest
from src.preprocessing import (
    clean_html,
    normalize_whitespace,
    validate_text,
    preprocess_for_transformer,
    preprocess_for_tfidf
)


def test_clean_html_removes_tags_and_unescapes_entities():
    sample = "This movie was <b>fantastic</b>! &quot;Must watch&quot;<br /><br />"
    cleaned = clean_html(sample)
    assert "<b>" not in cleaned
    assert "</b>" not in cleaned
    assert "<br />" not in cleaned
    assert '"Must watch"' in cleaned
    assert "fantastic" in cleaned


def test_clean_html_non_string_handling():
    assert clean_html(None) == ""
    assert clean_html(123) == ""


def test_normalize_whitespace():
    sample = "  This   is  a\n\n\ttest    review.   "
    normalized = normalize_whitespace(sample)
    assert normalized == "This is a test review."


def test_validate_text_valid_input():
    valid = "   Great movie experience!   "
    assert validate_text(valid) == "Great movie experience!"


def test_validate_text_raises_on_empty_or_none():
    with pytest.raises(ValueError, match="cannot be None"):
        validate_text(None)

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_text("    ")

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_text("")


def test_preprocess_for_transformer_preserves_negation_and_punctuation():
    # Critical requirement: negation words and exclamation marks must be preserved
    text = "The movie was <br />NOT good at all!!!!"
    result = preprocess_for_transformer(text)

    assert "NOT" in result
    assert "!" in result
    assert "<br />" not in result
    assert "The movie was NOT good at all!!!!" == result


def test_preprocess_for_tfidf_lowercasing_and_options():
    text = "The Movie Was <b>SUPERB</b>!"
    result = preprocess_for_tfidf(text, lowercase=True, remove_punctuation=False)
    assert result == "the movie was superb!"

    result_no_punct = preprocess_for_tfidf(text, lowercase=True, remove_punctuation=True)
    assert result_no_punct == "the movie was superb"
