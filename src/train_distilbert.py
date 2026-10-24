"""DistilBERT Fine-Tuning Pipeline for Binary Sentiment Analysis.

Leverages Hugging Face Transformers, PyTorch, and optional weighted cross-entropy
loss for class imbalance handling. Supports GPU (CUDA) and CPU execution, early stopping,
and best checkpoint persistence.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    EvalPrediction,
    set_seed
)

from src.preprocessing import preprocess_for_transformer
from src.data_loader import load_config, load_processed_splits, LABEL_MAPPING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_distilbert")


class IMDbDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for tokenized text encodings and sentiment labels."""

    def __init__(self, encodings: Dict[str, Any], labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)


class WeightedTrainer(Trainer):
    """Custom Hugging Face Trainer implementing weighted cross-entropy loss."""

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Computes loss with optional class weights."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            # Move class weights to the active device of the model
            weights = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Computes accuracy, precision, recall, and F1 scores for evaluation."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def train_distilbert_pipeline(
    config_path: str = "configs/config.yaml",
    resume_from_checkpoint: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Executes the DistilBERT fine-tuning pipeline.

    Args:
        config_path: Path to YAML configuration.
        resume_from_checkpoint: Optional checkpoint folder to resume training from.
        max_samples: Optional integer to cap training samples for fast validation.

    Returns:
        Dictionary of training results and metrics.
    """
    config = load_config(config_path)
    t_cfg = config.get("transformer", {})
    ds_cfg = config.get("dataset", {})
    imb_cfg = config.get("class_imbalance", {})
    es_cfg = config.get("early_stopping", {})
    out_cfg = config.get("output", {})

    seed = ds_cfg.get("random_seed", 42)
    set_seed(seed)

    processed_dir = ds_cfg.get("processed_dir", "data/processed")
    save_dir = Path(t_cfg.get("save_dir", "models/distilbert-sentiment"))
    save_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(out_cfg.get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Hardware detection
    use_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    logger.info(f"Hardware inspection: Target device is {device_name} (CUDA available: {use_cuda})")

    # Load splits
    logger.info("Loading processed splits...")
    train_df, val_df, _ = load_processed_splits(processed_dir)

    if max_samples is not None and max_samples > 0:
        logger.info(f"Dev/Validation run: capping train split to {max_samples} samples.")
        train_df = train_df.iloc[:max_samples].reset_index(drop=True)
        val_df = val_df.iloc[:max(20, max_samples // 4)].reset_index(drop=True)

    # Class imbalance analysis
    y_train = train_df["label"].values
    classes, counts = np.unique(y_train, return_counts=True)
    majority_count = max(counts)
    minority_count = min(counts)
    imbalance_ratio = majority_count / max(minority_count, 1)

    strategy = imb_cfg.get("strategy", "auto")
    threshold_ratio = imb_cfg.get("threshold_ratio", 1.2)
    class_weights_tensor = None

    if strategy == "weighted_loss" or (strategy == "auto" and imbalance_ratio > threshold_ratio):
        logger.info(
            f"Class imbalance detected (ratio: {imbalance_ratio:.2f} > {threshold_ratio}). "
            f"Computing balanced class weights from training distribution..."
        )
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float)
        logger.info(f"Class weights: Negative={weights[0]:.4f}, Positive={weights[1]:.4f}")
    else:
        logger.info(
            f"Dataset is reasonably balanced (Negative: {counts[0]:,}, Positive: {counts[1]:,}, "
            f"ratio: {imbalance_ratio:.2f} <= {threshold_ratio}). Standard cross-entropy loss applied."
        )

    # Tokenizer & Model setup
    model_name = t_cfg.get("model_name_or_path", "distilbert-base-uncased")
    logger.info(f"Loading pretrained tokenizer & model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=t_cfg.get("num_labels", 2),
        id2label={0: "Negative", 1: "Positive"},
        label2id={"Negative": 0, "Positive": 1},
        attn_implementation="eager"
    )

    # Tokenization
    max_length = t_cfg.get("max_length", 256)
    logger.info(f"Tokenizing train and validation sets (max_length={max_length})...")
    X_train_clean = [preprocess_for_transformer(t) for t in train_df["review"]]
    X_val_clean = [preprocess_for_transformer(t) for t in val_df["review"]]

    train_encodings = tokenizer(
        X_train_clean,
        truncation=True,
        padding=True,
        max_length=max_length
    )
    val_encodings = tokenizer(
        X_val_clean,
        truncation=True,
        padding=True,
        max_length=max_length
    )

    train_dataset = IMDbDataset(train_encodings, y_train)
    val_dataset = IMDbDataset(val_encodings, val_df["label"].values)

    # Training arguments
    batch_size = t_cfg.get("train_batch_size", 8)
    eval_batch_size = t_cfg.get("eval_batch_size", 8)
    epochs = t_cfg.get("num_train_epochs", 2)
    lr = float(t_cfg.get("learning_rate", 2e-5))
    weight_decay = float(t_cfg.get("weight_decay", 0.01))
    grad_accum = t_cfg.get("gradient_accumulation_steps", 2)

    training_args = TrainingArguments(
        output_dir=str(save_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_steps=t_cfg.get("warmup_steps", 10),
        logging_steps=t_cfg.get("logging_steps", 50),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=t_cfg.get("save_total_limit", 1),
        fp16=use_cuda,  # Enable FP16 on GPU for memory efficiency and throughput
        report_to="none",
        seed=seed
    )

    callbacks = []
    if es_cfg.get("enabled", True):
        patience = es_cfg.get("patience", 2)
        threshold = es_cfg.get("threshold", 0.001)
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=patience,
            early_stopping_threshold=threshold
        ))

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights_tensor
    )

    logger.info("Commencing fine-tuning...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save best model and tokenizer
    logger.info(f"Saving fine-tuned model and tokenizer to {save_dir}...")
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    # Save training history and configuration
    history = {
        "train_runtime_seconds": train_result.metrics.get("train_runtime", 0.0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0.0),
        "total_flos": train_result.metrics.get("total_flos", 0.0),
        "train_loss": train_result.metrics.get("train_loss", 0.0),
        "log_history": trainer.state.log_history,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint
    }
    history_file = artifacts_dir / "distilbert_training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_file}.")

    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DistilBERT fine-tuning pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--quick-test", action="store_true", help="Run quick validation on 100 samples")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap training samples for dev mode")
    args = parser.parse_args()

    cap = 100 if args.quick_test else args.max_samples
    train_distilbert_pipeline(
        config_path=args.config,
        resume_from_checkpoint=args.resume,
        max_samples=cap
    )
