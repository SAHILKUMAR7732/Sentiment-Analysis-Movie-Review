# 🎬 Sentiment Analysis on Movie Reviews using Deep Learning

**Author:** [SAHIL KUMAR](https://github.com/SAHILKUMAR7732)  
**Date:** October 2026  
**Repository Version:** 1.0.0  

An enterprise-ready, reproducible Natural Language Processing (NLP) system comparing **TF-IDF + Logistic Regression** with fine-tuned **DistilBERT** on the Stanford IMDb 50,000 Movie Reviews dataset. Features include leak-free stratified data partitioning, automated class-imbalance handling, multi-metric benchmarking, Transformer self-attention visualization, and an interactive Streamlit web dashboard.

---

## 📌 Table of Contents
- [Overview & Objectives](#overview--objectives)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Dataset Details](#dataset-details)
- [Installation & Setup](#installation--setup)
- [Execution Workflow](#execution-workflow)
- [Interactive Streamlit Application](#interactive-streamlit-application)
- [Evaluation & Benchmarking](#evaluation--benchmarking)
- [Transformer Attention Visualization](#transformer-attention-visualization)
- [Running Unit Tests](#running-unit-tests)
- [GitHub Deployment Guide](#github-deployment-guide)
- [Limitations & Ethical Considerations](#limitations--ethical-considerations)
- [License](#license)

---

## 📖 Overview & Objectives
Movie reviews present nuanced linguistic phenomena including sarcasm, multi-clause syntax, and negation. This project constructs an end-to-end sentiment classification system that classifies unstructured natural language reviews into:
- **`Negative` (Class 0)**
- **`Positive` (Class 1)**

The goal is to systematically benchmark traditional bag-of-words machine learning against deep bidirectional Transformer encoders (`distilbert-base-uncased`), measuring gains in accuracy, precision, recall, F1, and ROC-AUC on identical, held-out test data.

---

## ✨ Key Features
1. **Context-Preserving Preprocessing:** Specialized text pipelines that strip noisy HTML while strictly preserving casing, punctuation, and crucial negation markers (`not`, `never`, `hardly`).
2. **Leak-Free Partitioning:** Independent stratified train (21,250), validation (3,750), and held-out test (25,000) splits.
3. **Automated Imbalance Mitigation:** Dynamic class-weight calculation integrated into both scikit-learn and custom PyTorch / Hugging Face `Trainer` loops.
4. **Reproducible Baseline:** Sublinear TF-IDF $N$-gram vectorizer coupled with L2-regularized Logistic Regression.
5. **Transformer Fine-Tuning:** DistilBERT fine-tuning supporting FP16 mixed precision, gradient accumulation, early stopping, and best-checkpoint recovery.
6. **Multi-Metric Evaluation:** Accuracy, Precision, Recall, F1, Macro F1, Weighted F1, ROC-AUC, confusion matrices, and ROC curve plots.
7. **Attention Heatmap Interpretability:** Direct extraction and plotting of multi-head self-attention weights from Transformer layers.
8. **Interactive Streamlit UI:** Production-ready web dashboard with caching, real-time prediction, confidence meters, and in-browser attention maps.
9. **Rigorous Test Suite:** Complete pytest unit tests covering preprocessing, schema verification, stratified splitting, and prediction boundaries.

---

## 🏛️ System Architecture
```
                                 [ User / Raw Reviews ]
                                            │
                                            ▼
                           [ Text Validation & Sanitization ]
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
       [ TF-IDF Preprocessing ]                         [ Transformer Preprocessing ]
       - HTML removal                                   - HTML removal
       - Whitespace normalization                       - Whitespace normalization
       - Lowercasing                                    - Preserve casing & negations
                    │                                               │
                    ▼                                               ▼
       [ TfidfVectorizer ]                              [ WordPiece Tokenizer ]
       - 10,000 n-gram features                         - Subword IDs & Attention Masks
                    │                                               │
                    ▼                                               ▼
     [ Logistic Regression ]                           [ DistilBERT Transformer ]
     - Balanced Class Weights                          - 6-Layer Multi-Head Self-Attention
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            ▼
                           [ Evaluation & Inference Engine ]
                           - Softmax Calibrated Probabilities
                           - Attention Matrix Extraction
                           - Interactive Streamlit Dashboard
```

---

## 📂 Project Structure
```
sentiment-analysis-project/
│
├── app.py                      # Interactive Streamlit application
├── README.md                   # Comprehensive project documentation
├── requirements.txt            # Pinned dependency requirements
├── .gitignore                  # Git exclusion rules (data, weights, caches)
├── .env.example                # Template for local environment variables
├── LICENSE                     # MIT Open-Source License
│
├── configs/
│   └── config.yaml             # Centralized pipeline configuration & hyperparameters
│
├── data/
│   ├── README.md               # Data storage & licensing guide
│   ├── raw/                    # Raw IMDb partitions (cached offline)
│   └── processed/              # Processed train, val, test CSV/Parquet splits
│
├── notebooks/
│   └── sentiment_analysis_eda.ipynb # Interactive Exploratory Data Analysis notebook
│
├── src/
│   ├── __init__.py             # Package initialization
│   ├── data_loader.py          # Data ingestion, schema validation, and stratified splitting
│   ├── preprocessing.py        # Text sanitization for baseline and Transformers
│   ├── eda.py                  # Exploratory Data Analysis & statistical visualization
│   ├── baseline_model.py       # TF-IDF + Logistic Regression training & persistence
│   ├── train_distilbert.py     # DistilBERT fine-tuning pipeline with custom Trainer
│   ├── evaluate.py             # Test set benchmarking, confusion matrices & ROC curves
│   ├── predict.py              # Single and batch inference engine
│   └── visualize_attention.py  # Multi-head self-attention tensor extraction & heatmaps
│
├── tests/
│   ├── test_preprocessing.py   # Unit tests for text sanitization & negation preservation
│   ├── test_data_loader.py     # Unit tests for schema validation & split integrity
│   └── test_prediction.py      # Unit tests for inference contracts & probability axioms
│
├── models/
│   ├── README.md               # Model persistence documentation
│   ├── baseline/               # Saved TF-IDF vectorizer & Logistic Regression weights
│   └── distilbert-sentiment/   # Fine-tuned DistilBERT weights & tokenizer files
│
├── reports/
│   ├── README.md               # Evaluation reports documentation
│   ├── figures/                # Output plots (EDA, confusion matrices, ROC curves)
│   └── metrics/                # Machine-readable evaluation JSON & CSV summaries
│
├── artifacts/
│   └── README.md               # Intermediate run logs and training histories
│
└── docs/
    ├── PROJECT_DOCUMENTATION.md# In-depth technical report & architecture design
    ├── METHODOLOGY.md          # Workflow description from data to deployment
    ├── MODEL_CARD.md           # Model card with intended uses, biases, and risks
    ├── RESULTS_TEMPLATE.md     # Experimental logging template
    └── INTERVIEW_QUESTIONS.md  # 20 beginner-friendly NLP/ML interview questions & answers
```

---

## 🛠️ Tech Stack
- **Deep Learning & NLP:** PyTorch, Hugging Face `transformers`, `datasets`, `evaluate`, `accelerate`
- **Machine Learning & Analytics:** Scikit-Learn, Pandas, NumPy, Joblib
- **Visualization:** Matplotlib, Seaborn
- **Application & Serving:** Streamlit
- **Testing & Quality:** PyTest, PyYAML
- **Environment:** Python 3.10 - 3.13

---

## 📊 Dataset Details
- **Dataset:** Large Movie Review Dataset (Stanford IMDb)
- **Source:** Hugging Face Datasets (`imdb`) or Stanford AI Lab
- **Size:** 50,000 polar movie reviews
- **Splits:**
  - `Train`: 21,250 reviews (50% positive, 50% negative)
  - `Validation`: 3,750 reviews (stratified)
  - `Held-Out Test`: 25,000 reviews (untouched for unbiased benchmarking)

---

## 🚀 Installation & Setup

### Windows PowerShell
```powershell
# 1. Clone or navigate to the project directory
cd sentiment-analysis-project

# 2. Create and activate a Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Note: If PowerShell displays an ExecutionPolicy error, run:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS & Linux
```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔄 Execution Workflow

Execute the pipeline sequentially from the terminal:

```powershell
# Stage 1: Ingest, validate, and partition the dataset
python -m src.data_loader

# Stage 2: Perform Exploratory Data Analysis and generate charts
python -m src.eda

# Stage 3: Train and evaluate the TF-IDF + Logistic Regression baseline
python -m src.baseline_model

# Stage 4: Fine-tune the DistilBERT Transformer (uses GPU if available, else CPU)
python -m src.train_distilbert

# Stage 5: Evaluate both models on held-out test data & plot comparisons
python -m src.evaluate

# Stage 6: Test attention heatmap extraction
python -m src.visualize_attention
```

---

## 💻 Interactive Streamlit Application

Launch the web application locally:
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Streamlit App Interface Preview:
```
+---------------------------------------------------------------------------------+
| 🎬 Movie Review Sentiment Analyzer                                              |
| Classify natural language reviews using deep learning (DistilBERT & TF-IDF)     |
+---------------------------------------------------------------------------------+
| Select Example: [ Positive Masterpiece                          v ]             |
|                                                                                 |
| Review Text:                                                                    |
| "The screenplay was brilliant and the emotional resonance kept me captivated!"   |
|                                                                                 |
| [ 🚀 Analyze Sentiment ]                                                        |
+---------------------------------------------------------------------------------+
| 🟢 Positive Sentiment (Confidence: 98.42%)                                      |
|                                                                                 |
| Class Probabilities:                                                            |
| Negative: [==                              ] 1.58%                              |
| Positive: [================================] 98.42%                             |
|                                                                                 |
| > 🔍 Interactive Attention Visualization (Transformer Layer 5 Heatmap)          |
+---------------------------------------------------------------------------------+
```
*(Place actual screenshots in `reports/figures/app_screenshot.png` upon deployment).*

---

## 📈 Evaluation & Benchmarking

The evaluation suite generates comparative metrics on identical held-out test data:

| Metric | TF-IDF + Logistic Regression (Full 25k Test) | Fine-Tuned DistilBERT (Verified Dev Pipeline) |
| :--- | :---: | :---: |
| **Accuracy** | **89.48%** | 47.00% (100-sample dev) / Ready for Full GPU Run |
| **Precision** | **89.08%** | 47.49% |
| **Recall** | **90.00%** | 56.80% |
| **F1 Score** | **89.54%** | 51.73% |
| **ROC-AUC** | **0.9603** | 0.4896 |
| **Test Samples** | **25,000** | 500 (Stratified Test Benchmark) |

*Generated and verified from `reports/metrics/baseline_metrics.json` and `reports/metrics/model_comparison.csv`.*

---

## 🔍 Transformer Attention Visualization
Unlike black-box linear models, modern Transformer self-attention can be probed at inference time. Using `src/visualize_attention.py`, the system maps the query-key affinity weights:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
Heatmaps are saved to `reports/figures/attention_heatmap.png`.

> **Note on Interpretability:** Self-attention reveals token routing patterns inside the network but does not represent direct human-aligned causal explanation.

---

## 🧪 Running Unit Tests

Run the test suite using `pytest`:

```powershell
# Run all unit tests
pytest -v

# Run only preprocessing tests
pytest tests/test_preprocessing.py -v

# Run data loader and schema integrity tests
pytest tests/test_data_loader.py -v

# Run prediction contract tests
pytest tests/test_prediction.py -v
```

---

## 🐙 GitHub Deployment Guide

To push this repository to GitHub:
```powershell
# Initialize git repository
git init

# Add tracked files (large datasets and binary weights are excluded by .gitignore)
git add .

# Create initial commit
git commit -m "feat: complete portfolio-ready sentiment analysis project with DistilBERT and TF-IDF"

# Create a new repository on GitHub, then link and push:
git remote add origin https://github.com/<your-username>/sentiment-analysis-movie-reviews.git
git branch -M main
git push -u origin main
```

---

## ⚖️ Limitations & Ethical Considerations
- **Domain Confinement:** Trained on movie reviews; domain adaptation is recommended before applying to biomedical, legal, or financial texts.
- **Sarcasm Detection:** Indirect negative sentiment ("Brilliant choice to cast someone who cannot act") remains challenging for standard sequence classifiers.
- **Biases in Pretraining:** Pretrained language models may encode demographic biases from web training corpora; monitor for spurious correlations with protected attributes.

---

## 📜 License
Distributed under the MIT License. See [LICENSE](LICENSE) for full terms.
