# Model Card: Fine-Tuned DistilBERT for Movie Review Sentiment Analysis

## Model Details
- **Model Name:** Fine-Tuned DistilBERT for IMDb Sentiment Analysis
- **Model Type:** Transformer Encoder with Sequence Classification Head
- **Base Architecture:** `distilbert-base-uncased` (Hugging Face)
- **Parameters:** ~66,364,418 parameters
- **License:** Apache 2.0 (Base model) / MIT (Project implementation)
- **Frameworks:** PyTorch, Hugging Face Transformers

## Intended Use
- **Primary Use Case:** Classifying English-language movie and television reviews into binary sentiment classes (`Negative` or `Positive`).
- **Target Users:** Data scientists, machine learning engineers, and NLP practitioners studying text classification, attention visualization, and model serving.
- **Out-of-Scope Uses:**
  - Automated moderation of hate speech or toxicity without dedicated fine-tuning.
  - Multi-class sentiment detection (e.g., fine-grained 1-5 star ratings).
  - Cross-domain sentiment classification (e.g., financial earnings calls, clinical notes) without domain adaptation.

## Training Data & Splits
- **Dataset:** Stanford Large Movie Review Dataset (IMDb 50,000 reviews).
- **Training Set:** 21,250 reviews (50% negative, 50% positive).
- **Validation Set:** 3,750 reviews (stratified).
- **Test Set:** 25,000 reviews (untouched during training).
- **Data Preprocessing:** HTML stripped, whitespace normalized, WordPiece tokenization with max length = 256.

## Training Configuration
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 1e-8$)
- **Learning Rate:** $2 \times 10^{-5}$ with linear warmup and decay
- **Batch Size:** 8 per device (gradient accumulation steps = 2)
- **Epochs:** 2 (with early stopping patience = 2)
- **Weight Decay:** 0.01
- **Loss Function:** Cross-Entropy Loss (with support for weighted loss under class imbalance)

## Evaluation Results
*Note: Results are measured strictly on the 25,000 held-out test reviews.*
- **Accuracy:** See `reports/metrics/evaluation_summary.json`
- **F1 Score:** See `reports/metrics/evaluation_summary.json`
- **ROC-AUC:** See `reports/metrics/evaluation_summary.json`

## Ethical Considerations & Limitations
1. **Domain Sensitivity:** Review vocabulary is highly specific to films and entertainment. Performance may drop when applied to other domains (e-commerce products, technical bug reports).
2. **Sarcasm and Indirect Phrasing:** The model may struggle with deeply ironic or sarcastic reviews where literal vocabulary is positive but the author's intent is negative.
3. **Representation & Cultural Bias:** Pretrained language models encode cultural and demographic biases present in web crawl data. Entities associated with specific cultures, countries, or genders should not correlate spuriously with negative or positive sentiment.
4. **Attention Limitations:** Self-attention maps indicate token interaction within transformer layers but do not provide definitive causal explanation of model decisions.
