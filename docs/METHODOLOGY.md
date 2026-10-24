# Methodology: From Raw Text to Production Inference

This document details the step-by-step engineering and scientific methodology governing the sentiment analysis pipeline.

---

## 1. Pipeline Overview
The end-to-end pipeline consists of six distinct phases:
1. **Data Ingestion & Integrity Auditing**
2. **Context-Preserving Text Sanitization**
3. **Exploratory Data Analysis & Length Modeling**
4. **Baseline Formulation (TF-IDF + Logistic Regression)**
5. **Transformer Fine-Tuning (DistilBERT)**
6. **Interpretability, Evaluation, & Serving**

---

## 2. Phase 1: Data Ingestion & Splitting
- **Source:** Stanford Large Movie Review Dataset (IMDb) accessed via Hugging Face `datasets` or local offline cache.
- **Partitioning Strategy:**
  - Standard training partition (25,000 reviews) is split into:
    - **Training Set (85%):** 21,250 reviews for feature extraction and parameter optimization.
    - **Validation Set (15%):** 3,750 reviews for early stopping, learning rate scheduling, and hyperparameter selection.
  - **Held-Out Test Set:** 25,000 reviews kept completely isolated until final evaluation.
- **Leakage Prevention:**
  - Tokenizers, vocabularies, and TF-IDF IDF values are computed **strictly** on the training split.
  - Duplicate reviews in raw data are identified and logged to maintain strict boundary isolation.

---

## 3. Phase 2: Preprocessing Strategy
NLP models differ in their sensitivity to text structure:

### A. TF-IDF Pipeline (`preprocess_for_tfidf`)
1. Remove HTML tags (`<br />`, `<p>`).
2. Unescape special HTML characters (`&quot;`, `&amp;`).
3. Normalize erratic whitespace and linebreaks.
4. Convert to lowercase.
5. Retain negation words (`not`, `never`, `no`, `neither`) so $n$-grams (e.g. `not good`) capture contextual sentiment.

### B. Transformer Pipeline (`preprocess_for_transformer`)
1. Remove HTML tags and unescape entities.
2. Normalize whitespace.
3. **Preserve punctuation:** Question marks (`?`), exclamation points (`!`), and ellipsis carry high emotional signals.
4. **Preserve capitalization:** Uppercase words indicate emphasis and strong sentiment.
5. Apply WordPiece subword tokenization with maximum sequence length of 256.

---

## 4. Phase 3: Baseline Architecture
- **Feature Extraction:** `TfidfVectorizer`
  - $N$-gram range: $(1, 2)$ (unigrams + bigrams).
  - Maximum features: 10,000 top vocabulary terms.
  - Sublinear term frequency scaling: replaces $TF$ with $1 + \log(TF)$.
- **Estimator:** `LogisticRegression`
  - Solver: `lbfgs`
  - Maximum iterations: 1,000
  - Regularization parameter: $C = 1.0$ (L2 penalty)
  - Balanced class weighting: inversely proportional to class frequencies.

---

## 5. Phase 4: DistilBERT Fine-Tuning Architecture
- **Model Checkpoint:** `distilbert-base-uncased`
- **Architecture:**
  - 6 Transformer encoder blocks.
  - Hidden dimension: 768.
  - 12 multi-head self-attention mechanisms per layer.
  - Total parameters: ~66.3 Million.
- **Classification Head:**
  - Dropout layer ($p = 0.2$).
  - Linear projection layer: $\mathbb{R}^{768} \to \mathbb{R}^2$.
- **Training Hyperparameters:**
  - Sequence Length: 256 tokens.
  - Train Batch Size: 8 (effective batch size 16 with gradient accumulation = 2).
  - Learning Rate: $2 \times 10^{-5}$ using AdamW optimizer.
  - Weight Decay: 0.01.
  - Warmup Ratio: 10% of total training steps.
  - Precision: FP16 on CUDA GPUs, FP32 on CPU.
  - Checkpointing: Evaluated at every epoch with `load_best_model_at_end=True` and early stopping patience = 2.

---

## 6. Phase 5: Evaluation & Interpretability
- **Metrics Computed on Held-Out Test Set:**
  - Accuracy, Precision, Recall, F1 Score, Macro F1, Weighted F1, ROC-AUC.
- **Attention Map Extraction:**
  - Model forward pass with `output_attentions=True`.
  - Attention matrix from layer 5 extracted for token query-key interaction visualization.
- **Serving:**
  - Interactive Streamlit dashboard in `app.py` utilizing Streamlit caching (`@st.cache_resource`) to provide instant inference.
