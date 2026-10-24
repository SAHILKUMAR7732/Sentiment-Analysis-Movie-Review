"""Unit tests for dataset loading, validation, and stratified splitting.

Validates schema constraints, error handling, label sanity, and split integrity.
"""

import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    load_config,
    validate_dataframe,
    create_train_val_test_splits
)


def test_load_config_valid_and_invalid():
    # Valid default config
    cfg = load_config("configs/config.yaml")
    assert "dataset" in cfg
    assert "transformer" in cfg
    assert "baseline" in cfg
    assert "imdb" in cfg["dataset"]["name"]

    # Non-existent config
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent.yaml")


def test_validate_dataframe_success():
    data = {
        "review": ["Great film!", "Terrible plot.", "Loved the cinematography."],
        "label": [1, 0, 1]
    }
    df = pd.DataFrame(data)
    validated = validate_dataframe(df, "test")
    assert len(validated) == 3
    assert set(validated.columns) >= {"review", "label"}


def test_validate_dataframe_missing_column():
    data = {"text": ["Sample text"], "label": [1]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(df, "test")


def test_validate_dataframe_invalid_labels():
    data = {
        "review": ["Okay film", "Bad film"],
        "label": [1, 5]  # 5 is an invalid sentiment label
    }
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Invalid labels detected"):
        validate_dataframe(df, "test")


def test_validate_dataframe_handles_nulls_and_empty():
    data = {
        "review": ["Good film", None, "   ", "Bad film"],
        "label": [1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    validated = validate_dataframe(df, "test")
    # Rows with None and whitespace-only should be pruned
    assert len(validated) == 2
    assert list(validated["label"]) == [1, 0]


def test_create_train_val_test_splits_stratification_and_leakage():
    # Construct mock dataset with 100 samples (50 pos, 50 neg)
    reviews = [f"Unique review text sample #{i}" for i in range(100)]
    labels = [1] * 50 + [0] * 50
    mock_train = pd.DataFrame({"review": reviews, "label": labels})
    mock_test = pd.DataFrame({
        "review": [f"Test review text sample #{i}" for i in range(20)],
        "label": [1] * 10 + [0] * 10
    })

    train_split, val_split, test_split = create_train_val_test_splits(
        train_df=mock_train,
        test_df=mock_test,
        val_size=0.20,
        random_seed=42
    )

    # Validate split lengths
    assert len(train_split) == 80
    assert len(val_split) == 20
    assert len(test_split) == 20

    # Validate stratification preservation
    assert train_split["label"].value_counts()[1] == 40
    assert train_split["label"].value_counts()[0] == 40
    assert val_split["label"].value_counts()[1] == 10
    assert val_split["label"].value_counts()[0] == 10

    # Validate zero data overlap / leakage
    train_texts = set(train_split["review"])
    val_texts = set(val_split["review"])
    test_texts = set(test_split["review"])

    assert len(train_texts.intersection(val_texts)) == 0
    assert len(train_texts.intersection(test_texts)) == 0
