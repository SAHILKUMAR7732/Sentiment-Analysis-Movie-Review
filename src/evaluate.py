"""Model Evaluation and Comparative Analysis Module.

Evaluates both TF-IDF baseline and fine-tuned DistilBERT models on the held-out
test set. Generates comparative bar charts, confusion matrices, ROC curves,
and persists verified test metrics.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.preprocessing import preprocess_for_tfidf, preprocess_for_transformer
from src.data_loader import load_config, load_processed_splits, LABEL_MAPPING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("evaluate")


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """Computes all key evaluation metrics for binary classification."""
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if y_prob is not None else None,
        "num_test_samples": int(len(y_true))
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path
) -> None:
    """Plots and saves normalized and raw confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
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
    ax.set_title(title, pad=12)
    ax.set_xlabel("Predicted Sentiment")
    ax.set_ylabel("True Sentiment")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix figure to {output_path}.")


def plot_roc_curves(
    curves_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path
) -> None:
    """Plots ROC curves comparing multiple models."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, (y_true, y_prob) in curves_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})", lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", pad=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curves figure to {output_path}.")


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Generates side-by-side grouped bar chart comparing performance metrics."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

    df_melted = comparison_df.melt(
        id_vars=["model"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=df_melted,
        x="Metric",
        y="Score",
        hue="model",
        palette=["#4a90e2", "#50e3c2"],
        ax=ax
    )

    ax.set_title("Model Performance Comparison on Held-Out Test Set", pad=12)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("Score")
    ax.legend(title="Model", loc="lower right")

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2., height),
                ha="center", va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points"
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved comparison bar chart to {output_path}.")


def plot_training_history(
    history_path: Path,
    output_path: Path
) -> None:
    """Plots training and validation loss curves from history JSON if available."""
    if not history_path.is_file():
        return
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    logs = history.get("log_history", [])
    if not logs:
        return

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in logs:
        step = entry.get("step")
        if "loss" in entry and step is not None:
            train_steps.append(step)
            train_losses.append(entry["loss"])
        if "eval_loss" in entry and step is not None:
            eval_steps.append(step)
            eval_losses.append(entry["eval_loss"])

    if not train_losses and not eval_losses:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if train_losses:
        ax.plot(train_steps, train_losses, label="Training Loss", color="#4a90e2", lw=2)
    if eval_losses:
        ax.plot(eval_steps, eval_losses, label="Validation Loss", color="#e94e77", lw=2, marker="o")

    ax.set_title("DistilBERT Training & Validation Loss Curve", pad=12)
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved training history curves to {output_path}.")


def evaluate_baseline(
    config: Dict[str, Any],
    test_df: pd.DataFrame
) -> Optional[Tuple[Dict[str, Any], np.ndarray, np.ndarray]]:
    """Loads and evaluates the saved baseline model."""
    b_cfg = config.get("baseline", {})
    save_dir = Path(b_cfg.get("save_dir", "models/baseline"))
    vec_path = save_dir / "tfidf_vectorizer.joblib"
    clf_path = save_dir / "logistic_regression.joblib"

    if not (vec_path.is_file() and clf_path.is_file()):
        logger.warning(f"Baseline model artifacts not found at {save_dir}. Skipping baseline evaluation.")
        return None

    logger.info("Evaluating TF-IDF + Logistic Regression on test split...")
    vectorizer = joblib.load(vec_path)
    classifier = joblib.load(clf_path)

    X_test_clean = [preprocess_for_tfidf(t) for t in test_df["review"]]
    y_test = test_df["label"].values

    X_test = vectorizer.transform(X_test_clean)
    preds = classifier.predict(X_test)
    probs = classifier.predict_proba(X_test)[:, 1]

    metrics = compute_comprehensive_metrics(y_test, preds, probs, "TF-IDF + Logistic Regression")
    return metrics, preds, probs


def evaluate_distilbert(
    config: Dict[str, Any],
    test_df: pd.DataFrame
) -> Optional[Tuple[Dict[str, Any], np.ndarray, np.ndarray]]:
    """Loads and evaluates the fine-tuned DistilBERT model."""
    t_cfg = config.get("transformer", {})
    save_dir = Path(t_cfg.get("save_dir", "models/distilbert-sentiment"))
    config_file = save_dir / "config.json"

    if not config_file.is_file():
        logger.warning(f"DistilBERT model artifacts not found at {save_dir}. Skipping DistilBERT evaluation.")
        return None

    logger.info("Evaluating fine-tuned DistilBERT on test split...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(save_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(save_dir))
    model.to(device)
    model.eval()

    X_test_clean = [preprocess_for_transformer(t) for t in test_df["review"]]
    y_test = test_df["label"].values
    max_length = t_cfg.get("max_length", 256)
    batch_size = t_cfg.get("eval_batch_size", 16)

    encodings = tokenizer(
        X_test_clean,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(y_test))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_probs = [], []

    with torch.no_grad():
        for batch_input_ids, batch_mask, _ in dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_mask = batch_mask.to(device)

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            pos_probs = probs[:, 1].cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(pos_probs)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = compute_comprehensive_metrics(y_test, all_preds, all_probs, "DistilBERT Fine-Tuned")
    return metrics, all_preds, all_probs


def run_evaluation(
    config_path: str = "configs/config.yaml",
    max_eval_samples: Optional[int] = None
) -> pd.DataFrame:
    """Executes evaluation pipeline on held-out test data and saves comparisons."""
    config = load_config(config_path)
    ds_cfg = config.get("dataset", {})
    out_cfg = config.get("output", {})

    processed_dir = ds_cfg.get("processed_dir", "data/processed")
    figures_dir = Path(out_cfg.get("figures_dir", "reports/figures"))
    metrics_dir = Path(out_cfg.get("metrics_dir", "reports/metrics"))
    artifacts_dir = Path(out_cfg.get("artifacts_dir", "artifacts"))

    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_df = load_processed_splits(processed_dir)

    if max_eval_samples is not None and max_eval_samples > 0:
        logger.info(f"Subsampling test split (stratified) to {max_eval_samples} samples.")
        half = max_eval_samples // 2
        neg_df = test_df[test_df["label"] == 0].sample(min((test_df["label"] == 0).sum(), half), random_state=42)
        pos_df = test_df[test_df["label"] == 1].sample(min((test_df["label"] == 1).sum(), half), random_state=42)
        test_df = pd.concat([neg_df, pos_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    y_test = test_df["label"].values

    results = []
    roc_curves_dict = {}

    # 1. Baseline Evaluation
    baseline_res = evaluate_baseline(config, test_df)
    if baseline_res is not None:
        b_metrics, b_preds, b_probs = baseline_res
        results.append(b_metrics)
        roc_curves_dict["TF-IDF + Logistic Regression"] = (y_test, b_probs)
        plot_confusion_matrix(
            y_test,
            b_preds,
            "Confusion Matrix: TF-IDF Baseline",
            figures_dir / "baseline_confusion_matrix.png"
        )

    # 2. DistilBERT Evaluation
    distilbert_res = evaluate_distilbert(config, test_df)
    if distilbert_res is not None:
        d_metrics, d_preds, d_probs = distilbert_res
        results.append(d_metrics)
        roc_curves_dict["DistilBERT Fine-Tuned"] = (y_test, d_probs)
        plot_confusion_matrix(
            y_test,
            d_preds,
            "Confusion Matrix: DistilBERT Fine-Tuned",
            figures_dir / "distilbert_confusion_matrix.png"
        )

    # Plot training loss curve if history is present
    plot_training_history(
        artifacts_dir / "distilbert_training_history.json",
        figures_dir / "distilbert_loss_curve.png"
    )

    if not results:
        logger.warning("No trained models were found for evaluation.")
        return pd.DataFrame()

    comparison_df = pd.DataFrame(results)

    # Save metrics
    metrics_json_path = metrics_dir / "evaluation_summary.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    comparison_csv_path = metrics_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)

    # Plot ROC comparison if both are present
    if len(roc_curves_dict) > 1:
        plot_roc_curves(roc_curves_dict, figures_dir / "roc_curve_comparison.png")
        plot_model_comparison_bar(comparison_df, figures_dir / "model_comparison_bar.png")

    logger.info(f"\n==================== TEST SET EVALUATION SUMMARY ====================\n")
    logger.info(f"\n{comparison_df.to_string(index=False)}\n")
    logger.info(f"======================================================================\n")

    return comparison_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate sentiment models")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap test samples for fast evaluation")
    parser.add_argument("--quick-test", action="store_true", help="Run fast evaluation on 500 test samples")
    args = parser.parse_args()

    samples = 500 if args.quick_test else args.max_samples
    run_evaluation(config_path=args.config, max_eval_samples=samples)
