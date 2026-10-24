"""Sentiment Inference and Prediction Module.

Provides robust prediction interfaces for single reviews and batched inference
with softmax probability computation and input validation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

import torch
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.preprocessing import (
    clean_html,
    normalize_whitespace,
    preprocess_for_transformer,
    preprocess_for_tfidf,
    validate_text
)
from src.data_loader import LABEL_MAPPING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("predict")

MAX_CHAR_LIMIT = 15000  # Guardrail against excessive memory consumption


class SentimentPredictor:
    """Inference engine supporting DistilBERT and TF-IDF baseline models."""

    def __init__(
        self,
        model_dir: str = "models/distilbert-sentiment",
        baseline_dir: str = "models/baseline",
        device: Optional[str] = None
    ):
        self.model_dir = Path(model_dir)
        self.baseline_dir = Path(baseline_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_type: Optional[str] = None
        self.distilbert_model = None
        self.distilbert_tokenizer = None
        self.baseline_vectorizer = None
        self.baseline_classifier = None

        self._load_available_model()

    def _load_available_model(self) -> None:
        """Attempts to load fine-tuned DistilBERT first, falling back to TF-IDF baseline."""
        # 1. Try DistilBERT
        if (self.model_dir / "config.json").is_file():
            logger.info(f"Loading fine-tuned DistilBERT from {self.model_dir}...")
            self.distilbert_tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.distilbert_model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir),
                attn_implementation="eager"
            )
            self.distilbert_model.to(self.device)
            self.distilbert_model.eval()
            self.model_type = "DistilBERT"
            return

        # 2. Try Baseline
        vec_file = self.baseline_dir / "tfidf_vectorizer.joblib"
        clf_file = self.baseline_dir / "logistic_regression.joblib"
        if vec_file.is_file() and clf_file.is_file():
            logger.info(f"DistilBERT checkpoint not found. Loading baseline from {self.baseline_dir}...")
            self.baseline_vectorizer = joblib.load(vec_file)
            self.baseline_classifier = joblib.load(clf_file)
            self.model_type = "TF-IDF + Logistic Regression"
            return

        logger.warning(
            "No trained model artifacts found on disk.\n"
            "To train the baseline:    python -m src.baseline_model\n"
            "To train DistilBERT:      python -m src.train_distilbert"
        )
        self.model_type = None

    def is_ready(self) -> bool:
        """Returns True if at least one model is initialized."""
        return self.model_type is not None

    def validate_input(self, raw_text: Any) -> str:
        """Validates input type, nullity, and enforces sane length bounds."""
        sanitized = validate_text(raw_text)
        if len(sanitized) > MAX_CHAR_LIMIT:
            logger.warning(
                f"Input review length ({len(sanitized)} chars) exceeds maximum threshold ({MAX_CHAR_LIMIT}). "
                f"Truncating input."
            )
            sanitized = sanitized[:MAX_CHAR_LIMIT]
        return sanitized

    def predict_single(
        self,
        review_text: str,
        include_attention: bool = False
    ) -> Dict[str, Any]:
        """Classifies sentiment for an individual review string.

        Args:
            review_text: Raw movie review string.
            include_attention: Whether to extract token attention summary.

        Returns:
            Dictionary matching specifications:
            {
                "original_review": str,
                "sentiment": "Positive" | "Negative",
                "confidence": float,
                "probabilities": {"Negative": float, "Positive": float},
                "model_type": str
            }
        """
        if not self.is_ready():
            raise FileNotFoundError(
                "Model files are not installed. Please train a model first using:\n"
                "  python -m src.baseline_model  OR\n"
                "  python -m src.train_distilbert"
            )

        validated_text = self.validate_input(review_text)

        if self.model_type == "DistilBERT":
            cleaned = preprocess_for_transformer(validated_text)
            inputs = self.distilbert_tokenizer(
                cleaned,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.distilbert_model(**inputs, output_attentions=include_attention)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            neg_prob = float(probs[0])
            pos_prob = float(probs[1])

        else:
            # Baseline model
            cleaned = preprocess_for_tfidf(validated_text)
            vec = self.baseline_vectorizer.transform([cleaned])
            probs = self.baseline_classifier.predict_proba(vec)[0]
            neg_prob = float(probs[0])
            pos_prob = float(probs[1])

        pred_label_id = int(np.argmax([neg_prob, pos_prob]))
        predicted_sentiment = LABEL_MAPPING[pred_label_id]
        confidence = max(neg_prob, pos_prob)

        result: Dict[str, Any] = {
            "original_review": review_text,
            "sentiment": predicted_sentiment,
            "confidence": round(float(confidence), 4),
            "probabilities": {
                "Negative": round(neg_prob, 4),
                "Positive": round(pos_prob, 4)
            },
            "model_type": self.model_type
        }

        return result

    def predict(
        self,
        reviews: Union[str, List[str]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Predicts sentiment for a single review or list of reviews."""
        if isinstance(reviews, str):
            return self.predict_single(reviews)
        elif isinstance(reviews, (list, tuple)):
            return [self.predict_single(r) for r in reviews]
        else:
            raise TypeError(f"Expected str or list of str, got {type(reviews).__name__}")


# Module-level convenience function
_DEFAULT_PREDICTOR: Optional[SentimentPredictor] = None


def predict_sentiment(
    reviews: Union[str, List[str]],
    model_dir: str = "models/distilbert-sentiment",
    baseline_dir: str = "models/baseline"
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Top-level convenience prediction function."""
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None or not _DEFAULT_PREDICTOR.is_ready():
        _DEFAULT_PREDICTOR = SentimentPredictor(model_dir=model_dir, baseline_dir=baseline_dir)
    return _DEFAULT_PREDICTOR.predict(reviews)


if __name__ == "__main__":
    sample = "The cinematography was breathtaking and the performances were remarkably moving."
    predictor = SentimentPredictor()
    if predictor.is_ready():
        res = predictor.predict(sample)
        print("Prediction Result:")
        print(res)
    else:
        print("No models currently installed. Run training pipeline first.")
