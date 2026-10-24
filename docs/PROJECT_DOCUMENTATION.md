# Project Documentation: Movie Review Sentiment Analysis using Deep Learning

**Project Title:** End-to-End Sentiment Analysis on Movie Reviews using TF-IDF and Fine-Tuned DistilBERT  
**Author:** NLP & Machine Learning Engineering Team  
**Date:** October 2026  
**Repository Version:** 1.0.0  

---

## 1. Abstract
This project presents an enterprise-grade, reproducible Natural Language Processing (NLP) pipeline designed to classify movie reviews into binary sentiment classes (Negative vs. Positive). Using the Stanford Large Movie Review Dataset (IMDb 50,000 reviews), we build, benchmark, and deploy two contrasting NLP architectures: a traditional bag-of-words baseline combining Term Frequency-Inverse Document Frequency (TF-IDF) with L2-regularized Logistic Regression, and a state-of-the-art Transformer architecture utilizing fine-tuned `distilbert-base-uncased`. The system includes extensive exploratory data analysis, leak-free stratified partitioning, class imbalance detection, multi-metric evaluation (Accuracy, Precision, Recall, F1, Macro F1, Weighted F1, ROC-AUC, and Confusion Matrices), Transformer self-attention extraction and visualization, and an interactive Streamlit web dashboard.

---

## 2. Introduction & Background
Sentiment analysis is one of the foundational capabilities in applied Natural Language Processing, enabling organizations to understand consumer perception, synthesize public reviews, and monitor qualitative feedback at scale. While traditional bag-of-words methods historically provided computational efficiency, they fundamentally suffered from word-order insensitivity, high dimensionality, and failure to capture compositional semantics such as negation ("not bad", "barely enjoyable") and sarcasm.

The emergence of Transformer architectures (Vaswani et al., 2017) and pretrained contextual representations (Devlin et al., 2018) revolutionized text classification by computing dynamic token embeddings conditioned bidirectionally on entire sentences. DistilBERT (Sanh et al., 2019) reduces parameter count by 40% while retaining 97% of BERT's language comprehension capabilities and running 60% faster, making it an optimal candidate for production deployments.

---

## 3. Problem Statement & Objectives
The primary objective is to build an automated, reliable system that ingests unstructured natural language movie reviews and accurately outputs sentiment predictions accompanied by calibrated class probabilities.

### Core Engineering Goals:
1. **Zero Data Leakage:** Ensure training, validation, and test splits remain completely isolated, with vectorizers and transformers fit strictly on training partitions.
2. **Context-Preserving Preprocessing:** Differentiate between traditional bag-of-words cleaning (where stripping punctuation and lowercasing reduces sparsity) and Transformer preprocessing (where casing, punctuation, and negation tokens must be preserved for self-attention).
3. **Rigorous Benchmarking:** Measure the empirical performance differential between TF-IDF + Logistic Regression and fine-tuned DistilBERT on identical held-out test splits.
4. **Explainability & Attention Inspection:** Extract raw multi-head self-attention tensors from the final Transformer layer to visualize how tokens attend to each other during classification.
5. **Interactive Deployment:** Serve the model through a Streamlit web application with model caching, probability visualization, and interpretability heatmaps.

---

## 4. Scope & Architecture
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

## 5. Dataset Description
The system uses the benchmark Large Movie Review Dataset (IMDb):
- **Total Records:** 50,000 labeled reviews.
- **Official Partitions:** 25,000 training reviews and 25,000 testing reviews.
- **Labels:** Binary (0 = Negative, 1 = Positive).
- **Class Balance:** Exactly 50.0% Negative (12,500) and 50.0% Positive (12,500) in each raw split.
- **Validation Split:** 15% stratified split generated from the official training partition (3,750 validation samples, 21,250 training samples).
- **Held-Out Test Split:** The 25,000 official test samples remain untouched during all exploratory tuning and threshold optimization.

---

## 6. Data Preprocessing & Sanitization
Preprocessing is implemented in `src/preprocessing.py` and handles the distinct needs of linear models vs. deep Transformers:

| Preprocessing Step | TF-IDF Baseline | DistilBERT Transformer | Rationale |
| :--- | :--- | :--- | :--- |
| **HTML Tag Removal** | Applied (`<br />` stripped) | Applied (`<br />` stripped) | HTML tags introduce noise and meaningless tokens. |
| **Whitespace Normalization** | Applied | Applied | Converts irregular tabs/newlines into single spaces. |
| **Lowercasing** | Applied | Kept Intact / Handled by Uncased Tokenizer | DistilBERT-base-uncased normalizes tokens in its vocab; casing preservation helps maintain punctuation context. |
| **Punctuation Stripping** | Optional | Strictly Prohibited | Punctuation marks (`!`, `?`) carry critical emotional emphasis in reviews. |
| **Stop Word Removal** | Retains Negation Words | Strictly Prohibited | Removing words like "not", "no", "never" reverses sentiment semantics. |
| **Token Representation** | Sparse $N$-gram vectors | WordPiece Subword IDs + Attention Masks | Transformers require token IDs mapped to learned embedding matrices. |

---

## 7. Exploratory Data Analysis (EDA)
EDA is performed via `src/eda.py` and visual reports in `notebooks/sentiment_analysis_eda.ipynb`:
- **Word Count Distribution:** Reviews exhibit a right-skewed distribution. The mean word count is ~235 words, median ~175 words, with rare outliers exceeding 1,000 words.
- **Sequence Length Truncation:** Setting `max_length = 256` encapsulates over 85% of complete reviews and captures the dominant sentiment indicators (which typically occur in the opening and closing sentences of a review) while respecting GPU memory bounds.
- **Top Discriminative Words:**
  - *Positive:* "great", "excellent", "best", "wonderful", "love", "beautiful", "superb".
  - *Negative:* "bad", "worst", "waste", "boring", "awful", "terrible", "poor", "horrible".

---

## 8. Baseline Methodology: TF-IDF + Logistic Regression
- **Vectorizer:** Sublinear TF-IDF scaling (`sublinear_tf=True`), unigrams and bigrams (`ngram_range=(1, 2)`), top 10,000 vocabulary features.
- **Classifier:** L2-regularized Logistic Regression ($C=1.0$) solved via the L-BFGS algorithm.
- **Mathematical Formulation:**
  $$\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- **Strengths:** Millisecond inference latency, minimal memory usage, high interpretability via linear feature weights.
- **Limitations:** Incapable of long-range syntax parsing; treats non-adjacent word pairs independently; vulnerable to out-of-vocabulary variations.

---

## 9. DistilBERT Methodology
- **Backbone Checkpoint:** `distilbert-base-uncased` (6 transformer layers, 12 attention heads per layer, 768 hidden dimensions, 66M parameters).
- **Classification Head:** Linear projection on top of the pooled `[CLS]` token:
  $$P(y = c \mid \mathbf{x}) = \text{softmax}(\mathbf{W}_c \mathbf{h}_{\text{[CLS]}} + \mathbf{b}_c)$$
- **Fine-Tuning Regimen:**
  - Optimizer: AdamW with weight decay $\lambda = 0.01$.
  - Learning Rate: $\eta = 2 \times 10^{-5}$ with linear warmup (10% of steps) and linear decay.
  - Loss Function: Cross-Entropy with optional class weighting.
  - Regularization: Early stopping with patience = 2 based on validation F1 score.

---

## 10. Class Imbalance Handling
The dataset loader programmatically computes the imbalance ratio $\rho = \frac{N_{\text{majority}}}{N_{\text{minority}}}$.
- For balanced datasets ($\rho < 1.2$), standard unweighted cross-entropy loss is applied.
- If imbalance is introduced or detected, class weights are computed strictly from training labels:
  $$w_c = \frac{N_{\text{train}}}{2 \cdot N_c}$$
- For DistilBERT, weights are passed into PyTorch's `nn.CrossEntropyLoss(weight=w)` inside a custom `WeightedTrainer`.
- Test splits are never resampled, preventing distortion of realistic prevalence metrics.

---

## 11. Multi-Metric Evaluation Framework
Binary classification evaluation requires diverse, complementary metrics:
- **Accuracy:** General correctness; suitable only on balanced data.
- **Precision:** $\frac{TP}{TP + FP}$ — critical when false positives are costly.
- **Recall:** $\frac{TP}{TP + FN}$ — critical when missing positive reviews is penalizing.
- **F1 Score:** Harmonic mean of Precision and Recall.
- **Macro F1 Score:** Unweighted average of per-class F1 scores; unmasks minority class degradation.
- **ROC-AUC:** Area Under the Receiver Operating Characteristic curve; assesses rank-ordering quality across all classification thresholds.

---

## 12. Attention Visualization & Interpretability
Self-attention maps tokens to queries $\mathbf{Q}$, keys $\mathbf{K}$, and values $\mathbf{V}$:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
In `src/visualize_attention.py`, we extract the attention tensor of shape `(num_heads, seq_len, seq_len)` from layer 5 and render interactive heatmaps.
> **Scientific Caveat:** Attention matrices visualize internal representation routing within the neural network. They do not constitute absolute causal proof of human rationale, nor do they guarantee that an attended token is the sole driver of the final prediction.

---

## 13. Application Design (Streamlit)
`app.py` implements a reactive web dashboard:
- **Model Caching:** `@st.cache_resource` loads models once into GPU/RAM.
- **Interactive Review Input:** Preloaded examples or freeform typing.
- **Confidence Gauges & Probabilities:** Dynamic bars showing Negative vs Positive confidence.
- **Attention Exploration:** Selectable layer and attention head heatmaps rendered directly in-browser.
- **Graceful Fallbacks:** Guides users with exact commands if model checkpoints have not yet been trained.

---

## 14. Ethical Considerations & Limitations
1. **Domain Bias:** The model is trained on theatrical film reviews. Applying it to biomedical literature, financial sentiment, or customer service tickets may yield degraded accuracy due to domain shift.
2. **Sarcasm & Subtlety:** Sarcastic phrasing ("What a brilliant disaster") can confound both bag-of-words and small transformer models.
3. **Demographic / Cultural Biases:** Movie reviews contain references to culture, religion, and gender. Models must be monitored for spurious associations between named entities and sentiment polarity.
