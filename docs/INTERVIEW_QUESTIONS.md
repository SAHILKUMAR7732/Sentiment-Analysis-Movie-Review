# NLP & Sentiment Analysis: 20 Core Interview Questions & Answers

This guide provides comprehensive, beginner-friendly explanations for 20 fundamental concepts spanning Natural Language Processing, Machine Learning, Transformer architectures, and Model Evaluation.

---

### Q1: What is Sentiment Analysis, and what makes it challenging in real-world applications?
**Answer:**
Sentiment Analysis is the computational task of identifying, extracting, and categorizing opinions and affective states from natural language text. In binary sentiment classification, texts are typically mapped to `Negative` (0) or `Positive` (1).
Real-world challenges include:
1. **Negation and Inversion:** Sentences like "The movie was not good" or "I can't say I disliked it" reverse polarity based on auxiliary words.
2. **Sarcasm and Irony:** Phrases like "What a masterpiece of utter boredom" contain positive vocabulary used with negative intent.
3. **Domain Shift:** Words have different connotations across domains. For example, "unpredictable" is positive in a suspense thriller review, but negative in an automotive safety review.

---

### Q2: What is Natural Language Processing (NLP)?
**Answer:**
NLP is an interdisciplinary subfield of artificial intelligence, computer science, and linguistics focused on enabling computers to understand, interpret, process, and generate human language. NLP bridges unstructured text data with mathematical algorithms through numerical representations (embeddings and vectors).

---

### Q3: What is Tokenization, and how does subword tokenization differ from word-level tokenization?
**Answer:**
Tokenization is the process of segmenting a sequence of characters into meaningful discrete units called *tokens*.
- **Word-Level Tokenization:** Splits on whitespace and punctuation. Drawbacks include large vocabulary sizes and an inability to handle Out-Of-Vocabulary (OOV) words or typos.
- **Subword Tokenization (e.g., WordPiece, BPE):** Decomposes rare or compound words into frequent subword units (e.g., "unpredictable" $\to$ "un", "##predict", "##able"). This keeps vocabulary compact (e.g., ~30,000 tokens in DistilBERT) while completely eliminating unknown words.

---

### Q4: How does TF-IDF work, and what are its mathematical foundations?
**Answer:**
Term Frequency-Inverse Document Frequency (TF-IDF) converts text documents into numerical vectors by balancing how often a term appears in a document against how rare it is across the entire corpus:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
Where:
- $\text{TF}(t, d)$ measures frequency of term $t$ in document $d$.
- $\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$. Rare words receive higher weights, while universally common words (like "the", "and") receive low weights.

---

### Q5: What is Logistic Regression in the context of NLP classification?
**Answer:**
Logistic Regression is a linear classification model that calculates a weighted sum of input features (such as TF-IDF scores) and passes the result through the sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ to output a probability between 0 and 1.
It is lightweight, fast to train, and interpretable because the learned weights $\mathbf{w}$ indicate the positive or negative contribution of each word or $n$-gram to the final sentiment.

---

### Q6: What are the fundamental limitations of the TF-IDF + Logistic Regression approach?
**Answer:**
1. **Bag-of-Words Assumption:** It ignores word order and syntactic structure beyond immediate $n$-grams.
2. **Polysemy:** It assigns a single fixed feature weight to each word regardless of context (e.g., "bank" of a river vs. financial "bank").
3. **High Dimensionality and Sparsity:** Vocabulary matrices are vast and mostly contain zeros, consuming memory without capturing deep semantic relationships.

---

### Q7: What is the Transformer architecture and why did it replace RNNs/LSTMs?
**Answer:**
Introduced by Vaswani et al. (2017), the Transformer relies on *self-attention mechanisms* instead of recurrence.
- **Parallelization:** Unlike RNNs which process tokens sequentially step-by-step, Transformers process all tokens in a sequence concurrently, dramatically accelerating GPU training.
- **Direct Long-Range Dependencies:** Any token can attend directly to any other token regardless of distance, mitigating the vanishing gradient problems typical of RNNs.

---

### Q8: What is DistilBERT, and how does Knowledge Distillation work?
**Answer:**
DistilBERT is a lightweight, distilled version of BERT developed by Hugging Face.
- **Mechanism:** It undergoes **Knowledge Distillation**, where a compact "student" network (6 transformer layers) is trained to mimic the full output distribution and hidden states of a larger "teacher" network (BERT-base with 12 layers).
- **Benefits:** DistilBERT retains ~97% of BERT's language understanding while having 40% fewer parameters and running 60% faster during inference.

---

### Q9: What is Fine-Tuning in Transfer Learning?
**Answer:**
Fine-tuning takes a model that has already learned general language syntax and semantics from pretraining on billions of web tokens (e.g., Wikipedia, BookCorpus) and updates its parameters on a smaller, domain-specific labeled dataset (such as IMDb movie reviews) with a task-specific classification head.

---

### Q10: What is an Attention Mask, and why is it necessary?
**Answer:**
In neural network mini-batching, reviews of varying lengths are padded with a special token (`[PAD]`) to make all sequences equal length.
The **attention mask** is a binary array (with `1` for real tokens and `0` for padding tokens) that tells the self-attention mechanism to ignore padding positions, preventing padded positions from corrupting the softmax probability distribution.

---

### Q11: How does Scaled Dot-Product Self-Attention work?
**Answer:**
For each token embedding, the model computes three vectors via learned linear projections: **Query** ($\mathbf{Q}$), **Key** ($\mathbf{K}$), and **Value** ($\mathbf{V}$).
The attention weights are computed as:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
The dot product $\mathbf{Q} \mathbf{K}^T$ measures token similarity, $\sqrt{d_k}$ scales down the magnitude to prevent vanishing gradients in softmax, and multiplying by $\mathbf{V}$ produces a context-enriched vector representation.

---

### Q12: What is Class Imbalance, and how should it be addressed?
**Answer:**
Class imbalance occurs when one class significantly outnumbers another in the training dataset (e.g., 95% negative and 5% positive).
- **Risks:** A naive model can achieve 95% accuracy by simply predicting the majority class for every sample.
- **Solutions:**
  1. **Weighted Loss:** Multiplying the loss of minority samples by an inverse-frequency weight ($w_c$) forces the optimizer to prioritize minority errors.
  2. **Stratified Sampling:** Ensuring train/val/test splits mirror the true class distribution.
  3. **Threshold Tuning:** Adjusting the decision boundary from 0.5 to balance precision and recall.

---

### Q13: What are Precision and Recall, and what is the trade-off?
**Answer:**
- **Precision:** $\frac{TP}{TP + FP}$ — Of all reviews predicted as Positive, how many were actually Positive? High precision avoids false alarms.
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ — Of all truly Positive reviews, how many did the model identify? High recall avoids missed positives.
- **Trade-off:** Raising the classification decision threshold increases precision but lowers recall, and vice-versa.

---

### Q14: What is the F1-Score, and how do Macro F1 and Weighted F1 differ?
**Answer:**
The F1-score is the harmonic mean of precision and recall:
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Macro F1:** Computes the F1 score independently for each class and calculates the unweighted mean. It treats all classes equally, making it sensitive to minority class failure.
- **Weighted F1:** Averages per-class F1 scores weighted by the number of true instances in each class.

---

### Q15: What is a Confusion Matrix?
**Answer:**
A confusion matrix is a $2 \times 2$ contingency table summarizing classification performance:
- **True Positives (TP):** Positive reviews correctly classified as Positive.
- **True Negatives (TN):** Negative reviews correctly classified as Negative.
- **False Positives (FP / Type I Error):** Negative reviews mistakenly classified as Positive.
- **False Negatives (FN / Type II Error):** Positive reviews mistakenly classified as Negative.

---

### Q16: What is ROC-AUC?
**Answer:**
- **Receiver Operating Characteristic (ROC):** A curve plotting the True Positive Rate ($\text{Recall} = \frac{TP}{TP+FN}$) against the False Positive Rate ($\text{FPR} = \frac{FP}{FP+TN}$) across every classification threshold from 0.0 to 1.0.
- **Area Under the Curve (AUC):** Measures the probability that the model will rank a randomly chosen positive sample higher than a randomly chosen negative sample. An AUC of 1.0 represents a perfect model, while 0.5 represents random guessing.

---

### Q17: What is Overfitting, and how do we prevent it in Transformer fine-tuning?
**Answer:**
Overfitting occurs when a model memorizes idiosyncrasies and noise in the training set, achieving near-zero training loss while failing to generalize to unseen test data.
**Prevention Techniques:**
1. **Weight Decay ($\lambda$):** Penalizes large model weights ($L_2$ regularization).
2. **Dropout ($p$):** Randomly deactivates neurons during forward passes.
3. **Early Stopping:** Monitors validation loss/F1 and terminates training when validation metrics stop improving.
4. **Learning Rate Warmup:** Gradually ramps up learning rate to avoid destabilizing pretrained representations early in training.

---

### Q18: What is Data Leakage, and how did this project prevent it?
**Answer:**
Data leakage occurs when information from outside the training partition (such as validation or test data) contaminates model development, yielding unrealistically optimistic evaluation scores.
**Mitigations in this Project:**
- Splitting was performed prior to computing TF-IDF vocabularies or inverse document frequencies.
- Vectorizers and scalers were fit **only** on the training set and applied via `.transform()` to validation and test data.
- Hyperparameter tuning and checkpoint selection were governed strictly by the validation split; the test split remained completely held out until final evaluation.

---

### Q19: What are the scientific limitations of Self-Attention Visualizations?
**Answer:**
While attention heatmaps display internal query-key dot-product weights, they have documented limitations:
1. **Correlation vs. Causation:** Attention weights reflect information routing inside a specific layer, but subsequent feed-forward networks and skip-connections significantly transform these features. High attention does not prove that a token was the sole causal driver of the prediction.
2. **Multi-Head Dispersion:** Different attention heads specialize in distinct grammatical phenomena (e.g., punctuation tracking, coreference, syntactic agreement), making a single head's heatmap an incomplete picture.

---

### Q20: How do you serve a Transformer model efficiently in production?
**Answer:**
1. **Model Caching:** Load model weights once into GPU/RAM on startup using caching decorators (`@st.cache_resource` in Streamlit).
2. **Precision Reduction:** Use FP16 (half-precision) or INT8 quantization (e.g., ONNX Runtime or TensorRT) to reduce latency and VRAM usage.
3. **Dynamic Batching:** Group incoming real-time requests to maximize GPU utilization.
4. **Decoupled Architecture:** Expose prediction via a REST/gRPC API (e.g., FastAPI) separate from the front-end user interface.
