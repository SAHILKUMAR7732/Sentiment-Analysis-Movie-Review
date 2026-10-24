# Model Results Template & Experimental Log

This document records the empirical evaluation metrics obtained by running models on the identical held-out test split of the IMDb dataset.

> **Integrity Policy:** All metrics recorded here must be generated through real script execution (`src/evaluate.py`). Do not record fabricated, hypothetical, or estimated numbers. If a model has not been trained, record status as **Pending Training**.

---

## 1. Test Set Performance Comparison

| Model Architecture | Test Samples | Accuracy | Precision | Recall | F1 Score | Macro F1 | Weighted F1 | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression (Full Test)** | 25,000 | 0.8948 | 0.8908 | 0.9000 | 0.8954 | 0.8948 | 0.8948 | 0.9603 | **Verified Complete** |
| **TF-IDF + Logistic Regression (Benchmark Subset)** | 500 | 0.8960 | 0.9125 | 0.8760 | 0.8939 | 0.8960 | 0.8960 | 0.9599 | **Verified Complete** |
| **DistilBERT (Dev Fine-Tuned, 100 samples)** | 500 | 0.4700 | 0.4749 | 0.5680 | 0.5173 | 0.4649 | 0.4649 | 0.4896 | **Verified Pipeline** |
| **DistilBERT (Full 21,250 samples)** | 25,000 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | **Ready for GPU run** |

*Verified from `reports/metrics/baseline_metrics.json` and `reports/metrics/model_comparison.csv`.*

---

## 2. Experimental Run Metadata

| Parameter | Baseline Run | DistilBERT Transformer Run |
| :--- | :--- | :--- |
| **Execution Date** | YYYY-MM-DD | YYYY-MM-DD |
| **Hardware Device** | CPU / CUDA GPU | CPU / CUDA GPU |
| **Dataset Version** | Stanford IMDb (Hugging Face) | Stanford IMDb (Hugging Face) |
| **Vocabulary / Tokenizer** | 10,000 $N$-grams (1, 2) | WordPiece (30,522 vocab) |
| **Sequence Length** | Full review text | 256 tokens (truncated/padded) |
| **Effective Batch Size** | Full-batch / L-BFGS | 16 (8 per dev $\times$ 2 grad accum) |
| **Training Duration** | ~X seconds | ~X minutes |
| **Checkpoint Path** | `models/baseline/` | `models/distilbert-sentiment/` |

---

## 3. Key Observations & Trade-Offs
- **Inference Latency:** [Record latency per review in milliseconds]
- **Resource Footprint:** [Record disk space and RAM/VRAM consumption]
- **Domain Strengths:** [Record analysis of where DistilBERT outperforms TF-IDF, e.g., complex negations]
