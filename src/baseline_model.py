"""Baseline Model: TF-IDF Vectorizer + Logistic Regression.

Provides a fast, interpretable, bag-of-words benchmark against which the
DistilBERT Transformer model can be rigorously evaluated.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.preprocessing import preprocess_for_tfidf
from src.data_loader import load_config, load_processed_splits, LABEL_MAPPING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("baseline_model")


def train_baseline_pipeline(
    config_path: str = "configs/config.yaml"
) -> Dict[str, Any]:
    """Trains TF-IDF + Logistic Regression baseline, evaluates on test set, and saves artifacts.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Dictionary of baseline test evaluation metrics.
    """
    config = load_config(config_path)
    b_cfg = config.get("baseline", {})
    ds_cfg = config.get("dataset", {})
    out_cfg = config.get("output", {})

    processed_dir = ds_cfg.get("processed_dir", "data/processed")
    save_dir = Path(b_cfg.get("save_dir", "models/baseline"))
    figures_dir = Path(out_cfg.get("figures_dir", "reports/figures"))
    metrics_dir = Path(out_cfg.get("metrics_dir", "reports/metrics"))

    save_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed splits...")
    train_df, val_df, test_df = load_processed_splits(processed_dir)

    logger.info("Preprocessing text for TF-IDF...")
    X_train_raw = [preprocess_for_tfidf(t) for t in train_df["review"]]
    y_train = train_df["label"].values

    X_val_raw = [preprocess_for_tfidf(t) for t in val_df["review"]]
    y_val = val_df["label"].values

    X_test_raw = [preprocess_for_tfidf(t) for t in test_df["review"]]
    y_test = test_df["label"].values

    max_features = b_cfg.get("max_features", 10000)
    ngram_range = tuple(b_cfg.get("ngram_range", [1, 2]))
    c_param = b_cfg.get("c_param", 1.0)
    class_weight = b_cfg.get("class_weight", "balanced")
    seed = ds_cfg.get("random_seed", 42)

    logger.info(f"Fitting TfidfVectorizer (max_features={max_features}, ngram_range={ngram_range})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_val = vectorizer.transform(X_val_raw)
    X_test = vectorizer.transform(X_test_raw)

    logger.info(f"Training Logistic Regression (C={c_param}, class_weight={class_weight})...")
    classifier = LogisticRegression(
        C=c_param,
        class_weight=class_weight,
        max_iter=1000,
        random_state=seed,
        solver="lbfgs"
    )
    classifier.fit(X_train, y_train)

    # Validation evaluation for model validation
    val_preds = classifier.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    logger.info(f"Validation Accuracy: {val_acc:.4f}")

    # Held-out Test evaluation
    logger.info("Evaluating baseline model on untouched held-out test split...")
    test_preds = classifier.predict(X_test)
    test_probs = classifier.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": float(accuracy_score(y_test, test_preds)),
        "precision": float(precision_score(y_test, test_preds, zero_division=0)),
        "recall": float(recall_score(y_test, test_preds, zero_division=0)),
        "f1": float(f1_score(y_test, test_preds, zero_division=0)),
        "macro_f1": float(f1_score(y_test, test_preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, test_preds, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
        "num_test_samples": int(len(y_test))
    }

    report_str = classification_report(
        y_test,
        test_preds,
        target_names=["Negative", "Positive"],
        digits=4
    )
    logger.info(f"Test Classification Report:\n{report_str}")

    # Save metrics JSON
    metrics_path = metrics_dir / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Baseline metrics written to {metrics_path}.")

    # Generate and save Confusion Matrix plot
    cm = confusion_matrix(y_test, test_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax
    )
    ax.set_title("Confusion Matrix: TF-IDF + Logistic Regression", pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    cm_plot_path = figures_dir / "baseline_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_plot_path}.")

    # Serialize vectorizer and classifier
    vec_path = save_dir / "tfidf_vectorizer.joblib"
    clf_path = save_dir / "logistic_regression.joblib"
    joblib.dump(vectorizer, vec_path)
    joblib.dump(classifier, clf_path)
    logger.info(f"Saved serialized baseline models to {vec_path} and {clf_path}.")

    return metrics


if __name__ == "__main__":
    train_baseline_pipeline()
