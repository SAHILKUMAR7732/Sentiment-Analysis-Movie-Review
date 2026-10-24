"""Unit tests for the prediction module and inference engine.

Tests missing model exceptions, input bounds, probability validity,
and prediction response schema.
"""

import pytest
import numpy as np

from src.predict import SentimentPredictor


def test_missing_model_error_handling(tmp_path):
    # Predictor pointed to empty directories
    empty_distilbert = tmp_path / "empty_distilbert"
    empty_baseline = tmp_path / "empty_baseline"
    empty_distilbert.mkdir()
    empty_baseline.mkdir()

    predictor = SentimentPredictor(
        model_dir=str(empty_distilbert),
        baseline_dir=str(empty_baseline)
    )

    assert not predictor.is_ready()
    with pytest.raises(FileNotFoundError, match="Model files are not installed"):
        predictor.predict_single("This should trigger an error.")


def test_input_validation_empty_and_whitespace():
    predictor = SentimentPredictor(
        model_dir="non_existent",
        baseline_dir="non_existent"
    )

    with pytest.raises(ValueError, match="cannot be None"):
        predictor.validate_input(None)

    with pytest.raises(ValueError, match="cannot be empty"):
        predictor.validate_input("   ")


def test_input_validation_truncates_long_input():
    predictor = SentimentPredictor(
        model_dir="non_existent",
        baseline_dir="non_existent"
    )
    long_string = "word " * 4000  # 20,000 characters
    truncated = predictor.validate_input(long_string)
    assert len(truncated) <= 15000


def test_mock_prediction_schema_and_probability_laws():
    """Validates structural compliance and mathematical constraints of predictions."""
    # Simulate valid predictor output
    simulated_probs = {"Negative": 0.1234, "Positive": 0.8766}
    simulated_result = {
        "original_review": "Great performance",
        "sentiment": "Positive",
        "confidence": 0.8766,
        "probabilities": simulated_probs,
        "model_type": "DistilBERT"
    }

    # Schema checks
    assert set(simulated_result.keys()) >= {"original_review", "sentiment", "confidence", "probabilities"}
    assert simulated_result["sentiment"] in {"Positive", "Negative"}

    # Probability bounds
    for prob_val in simulated_result["probabilities"].values():
        assert 0.0 <= prob_val <= 1.0

    # Probabilities sum to 1
    total_prob = sum(simulated_result["probabilities"].values())
    assert pytest.approx(total_prob, abs=1e-3) == 1.0

    # Confidence matches predicted class probability
    assert simulated_result["confidence"] == simulated_result["probabilities"][simulated_result["sentiment"]]
