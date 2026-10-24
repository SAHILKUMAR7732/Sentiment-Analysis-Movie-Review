"""Dataset Management and Ingestion Module for IMDb Movie Reviews.

Handles loading, validating, splitting, and caching the IMDb sentiment dataset.
Ensures stratified train/validation/test partitions with zero data leakage.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("data_loader")

LABEL_MAPPING = {0: "Negative", 1: "Positive"}
REVERSE_LABEL_MAPPING = {"Negative": 0, "Positive": 1}


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads configuration settings from YAML file.

    Args:
        config_path: Relative or absolute path to config.yaml.

    Returns:
        Dictionary containing project configurations.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def validate_dataframe(df: pd.DataFrame, split_name: str = "dataset") -> pd.DataFrame:
    """Validates schema, checks missing values, and ensures valid binary labels.

    Args:
        df: Pandas DataFrame containing dataset records.
        split_name: Name of the partition (for logging clarity).

    Returns:
        Sanitized and validated DataFrame.

    Raises:
        ValueError: If required columns are missing or labels are invalid.
    """
    required_cols = {"review", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"[{split_name}] Missing required columns. Expected at least {required_cols}, "
            f"got: {set(df.columns)}"
        )

    initial_len = len(df)
    # Check and report missing reviews or labels
    null_reviews = df["review"].isnull().sum()
    null_labels = df["label"].isnull().sum()
    if null_reviews > 0 or null_labels > 0:
        logger.warning(
            f"[{split_name}] Found {null_reviews} null reviews and {null_labels} null labels. "
            f"Dropping rows with missing values."
        )
        df = df.dropna(subset=["review", "label"]).copy()

    # Verify review content is not empty string
    empty_strings = (df["review"].astype(str).str.strip() == "").sum()
    if empty_strings > 0:
        logger.warning(f"[{split_name}] Found {empty_strings} empty review strings. Filtering them out.")
        df = df[df["review"].astype(str).str.strip() != ""].copy()

    # Ensure labels are binary integers {0, 1}
    df["label"] = df["label"].astype(int)
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"[{split_name}] Invalid labels detected: {unique_labels}. Expected only {{0, 1}}.")

    # Duplicate check for informational transparency
    dup_count = df.duplicated(subset=["review"]).sum()
    if dup_count > 0:
        logger.info(
            f"[{split_name}] Found {dup_count} duplicate review texts out of {len(df)} samples."
        )

    logger.info(f"[{split_name}] Validation passed: {len(df)} valid records (retained from {initial_len}).")
    return df


def download_or_load_raw_dataset(
    cache_dir: str = "data/raw",
    sample_size: Optional[int] = None,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads or loads the official IMDb dataset via Hugging Face datasets.

    Falls back to local offline caching if already downloaded or presents clear
    instructions if network connectivity fails.

    Args:
        cache_dir: Directory where raw dataset cache is stored.
        sample_size: Optional integer to downsample for quick CPU/dev testing.
        random_seed: Random seed for sampling reproducibility.

    Returns:
        Tuple of (train_df, test_df) containing the official splits.
    """
    raw_path = Path(cache_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    local_train_csv = raw_path / "raw_train.csv"
    local_test_csv = raw_path / "raw_test.csv"

    # Check local cache first
    if local_train_csv.is_file() and local_test_csv.is_file():
        logger.info(f"Loading cached raw dataset from {raw_path}...")
        train_df = pd.read_csv(local_train_csv)
        test_df = pd.read_csv(local_test_csv)
    else:
        logger.info("Downloading official IMDb dataset via Hugging Face `datasets`...")
        try:
            from datasets import load_dataset
            # Use canonical HF namespace stanfordnlp/imdb with fallback
            try:
                hf_dataset = load_dataset("stanfordnlp/imdb", cache_dir=str(raw_path))
            except Exception:
                hf_dataset = load_dataset("imdb", cache_dir=str(raw_path))

            train_df = pd.DataFrame(hf_dataset["train"])
            test_df = pd.DataFrame(hf_dataset["test"])

            # Normalize column name if HF returns 'text' instead of 'review'
            if "text" in train_df.columns and "review" not in train_df.columns:
                train_df = train_df.rename(columns={"text": "review"})
            if "text" in test_df.columns and "review" not in test_df.columns:
                test_df = test_df.rename(columns={"text": "review"})

            # Cache locally for offline resilience
            train_df.to_csv(local_train_csv, index=False)
            test_df.to_csv(local_test_csv, index=False)
            logger.info(f"Saved raw partitions to {local_train_csv} and {local_test_csv}.")
        except Exception as e:
            logger.error(
                "\n==================== DATASET RECOVERY INSTRUCTIONS ====================\n"
                "Failed to download IMDb dataset from Hugging Face.\n"
                f"Error details: {e}\n\n"
                "Troubleshooting Steps:\n"
                "1. Verify your internet connection.\n"
                "2. If working behind a corporate proxy, set HTTP_PROXY and HTTPS_PROXY.\n"
                "3. Alternatively, manually place `raw_train.csv` and `raw_test.csv` in:\n"
                f"   {raw_path.resolve()}\n"
                "   with columns ['review', 'label'] (0 = Negative, 1 = Positive).\n"
                "========================================================================\n"
            )
            raise RuntimeError(
                "IMDb dataset could not be downloaded and was not found in local cache. "
                "See recovery instructions above."
            ) from e

    train_df = validate_dataframe(train_df, split_name="Raw Train")
    test_df = validate_dataframe(test_df, split_name="Raw Test")

    if sample_size is not None and sample_size > 0:
        logger.info(f"Dev mode active: Subsampling dataset to {sample_size} total records.")
        half_sample = sample_size // 2
        train_df = train_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), half_sample // 2), random_state=random_seed)
        ).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

        test_df = test_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), half_sample // 2), random_state=random_seed)
        ).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    return train_df, test_df


def create_train_val_test_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_size: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partitions the training set into train and validation using stratified sampling.

    The official test set is kept completely untouched to prevent data leakage.

    Args:
        train_df: Full raw training DataFrame.
        test_df: Full raw test DataFrame.
        val_size: Fraction of training records to allocate for validation.
        random_seed: Seed for reproducible partitioning.

    Returns:
        Tuple of (train_split, val_split, test_split).
    """
    logger.info(f"Creating stratified validation split ({val_size * 100:.1f}%) from training data...")

    train_split, val_split = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["label"],
        random_state=random_seed
    )

    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)
    test_split = test_df.reset_index(drop=True)

    # Verification: check for overlap between splits
    train_reviews = set(train_split["review"])
    val_reviews = set(val_split["review"])
    test_reviews = set(test_split["review"])

    val_leakage = len(train_reviews.intersection(val_reviews))
    test_leakage = len(train_reviews.intersection(test_reviews))
    if val_leakage > 0:
        logger.warning(
            f"Identified {val_leakage} reviews shared between train and val splits due to exact duplicates in source data."
        )
    if test_leakage > 0:
        logger.warning(
            f"Identified {test_leakage} reviews shared between train and test splits due to exact duplicates in source data."
        )

    logger.info(
        f"Splits successfully prepared:\n"
        f"  - Training samples:   {len(train_split):,} ({train_split['label'].value_counts().to_dict()})\n"
        f"  - Validation samples: {len(val_split):,} ({val_split['label'].value_counts().to_dict()})\n"
        f"  - Test samples:       {len(test_split):,} ({test_split['label'].value_counts().to_dict()})"
    )

    return train_split, val_split, test_split


def save_processed_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_dir: str = "data/processed"
) -> None:
    """Saves processed partitions into parquet or csv format.

    Args:
        train_df: Processed train DataFrame.
        val_df: Processed validation DataFrame.
        test_df: Processed test DataFrame.
        processed_dir: Directory where splits will be written.
    """
    out_path = Path(processed_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = out_path / "train.csv"
    val_file = out_path / "val.csv"
    test_file = out_path / "test.csv"

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    logger.info(f"Processed splits saved to {out_path.resolve()}.")


def load_processed_splits(
    processed_dir: str = "data/processed"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads pre-saved processed partitions.

    Args:
        processed_dir: Directory containing train.csv, val.csv, test.csv.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    path = Path(processed_dir)
    train_file = path / "train.csv"
    val_file = path / "val.csv"
    test_file = path / "test.csv"

    if not (train_file.is_file() and val_file.is_file() and test_file.is_file()):
        raise FileNotFoundError(
            f"Processed files not found in {path}. Run data_loader pipeline first."
        )

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    return train_df, val_df, test_df


def run_data_pipeline(config_path: str = "configs/config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end data pipeline: load config -> download/load -> split -> save.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    config = load_config(config_path)
    ds_cfg = config.get("dataset", {})

    raw_dir = ds_cfg.get("raw_dir", "data/raw")
    processed_dir = ds_cfg.get("processed_dir", "data/processed")
    val_size = ds_cfg.get("val_size", 0.15)
    random_seed = ds_cfg.get("random_seed", 42)
    sample_size = ds_cfg.get("sample_size", None)

    train_raw, test_raw = download_or_load_raw_dataset(
        cache_dir=raw_dir,
        sample_size=sample_size,
        random_seed=random_seed
    )

    train_split, val_split, test_split = create_train_val_test_splits(
        train_df=train_raw,
        test_df=test_raw,
        val_size=val_size,
        random_seed=random_seed
    )

    save_processed_splits(
        train_df=train_split,
        val_df=val_split,
        test_df=test_split,
        processed_dir=processed_dir
    )

    return train_split, val_split, test_split


if __name__ == "__main__":
    logger.info("Executing dataset pipeline standalone...")
    run_data_pipeline()
