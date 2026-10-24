# Models Directory

This directory stores trained model checkpoints, tokenizers, and serialized artifacts.

## Structure
- `baseline/`: Serialized TF-IDF vectorizer and Logistic Regression classifier (`.joblib`).
- `distilbert-sentiment/`: Fine-tuned DistilBERT weights, configuration, and tokenizer files.

## Note
Trained binary weights and checkpoints are git-ignored to prevent bloating Git history. They can be reproduced using the project training scripts.
