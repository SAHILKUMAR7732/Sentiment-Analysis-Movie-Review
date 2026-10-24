"""Streamlit Interactive Web Application for Movie Review Sentiment Analysis.

Provides an interactive user interface to classify movie reviews, inspect class
probabilities, view DistilBERT attention heatmaps, and explore NLP pipeline dynamics.
"""

import os
import sys
import logging
from pathlib import Path

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is available in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import SentimentPredictor
from src.visualize_attention import extract_attention_matrix, plot_attention_heatmap

# Page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern typography and cards
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .prediction-card-pos {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 1.2rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .prediction-card-neg {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 1.2rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_sentiment_predictor() -> SentimentPredictor:
    """Initializes and caches the sentiment predictor."""
    return SentimentPredictor(
        model_dir=str(PROJECT_ROOT / "models" / "distilbert-sentiment"),
        baseline_dir=str(PROJECT_ROOT / "models" / "baseline")
    )


def render_sidebar(predictor: SentimentPredictor):
    """Renders the workflow sidebar and model information."""
    with st.sidebar:
        st.header("⚙️ Pipeline Architecture")

        st.markdown("""
        **Data & Processing Workflow:**
        1. 📝 **Raw Review** (Text Input)
        2. 🧹 **Text Cleaning** (HTML & Whitespace)
        3. 🔤 **WordPiece Tokenizer** (Tokens & Masks)
        4. 🤖 **DistilBERT Encoder** (6 Transformer Layers)
        5. 📊 **Classification Head** (Linear + Softmax)
        6. 🎯 **Sentiment Output** (Positive / Negative)
        """)

        st.divider()

        st.subheader("Active Model Status")
        if predictor.is_ready():
            st.success(f"**Loaded:** {predictor.model_type}")
            st.info(f"**Device:** {predictor.device.upper()}")
        else:
            st.warning("⚠️ No local model found.")
            st.caption(
                "Run training commands to generate checkpoints:\n\n"
                "`python -m src.baseline_model`\n\n"
                "`python -m src.train_distilbert`"
            )

        st.divider()
        st.subheader("IMDb Dataset Context")
        st.caption(
            "• **Source:** Stanford IMDb 50,000 Movie Reviews\n"
            "• **Labels:** Binary (Negative: 0, Positive: 1)\n"
            "• **Balance:** Even 50% split across training and test splits"
        )


def main():
    predictor = get_sentiment_predictor()
    render_sidebar(predictor)

    st.markdown('<div class="main-header">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Classify natural language movie reviews with deep learning '
        'using pretrained DistilBERT and TF-IDF baselines.</div>',
        unsafe_allow_html=True
    )

    # Preset Examples
    examples = {
        "Custom Review": "",
        "Positive Masterpiece": "The screenplay was brilliant, the cinematography was mesmerizing, and the emotional resonance kept me captivated until the very last frame.",
        "Negative Critique": "A complete waste of time. The characters were unlikable, the plot was riddled with ridiculous clichés, and the pacing was utterly unbearable.",
        "Nuanced with Negation": "The movie was not terrible, but it certainly did not live up to the massive hype surrounding its stellar cast."
    }

    selected_example = st.selectbox(
        "💡 Select an example review or write your own below:",
        options=list(examples.keys()),
        index=0
    )

    default_text = examples[selected_example]
    user_review = st.text_area(
        "Enter Movie Review:",
        value=default_text,
        height=140,
        placeholder="Type or paste a movie review here (e.g., 'An absolute cinematic triumph with stellar performances...')..."
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_clicked:
        if not user_review.strip():
            st.warning("Please provide a non-empty review text before running analysis.")
            return

        if not predictor.is_ready():
            st.error(
                "🚨 No trained model checkpoint was detected in `models/`.\n\n"
                "Please train a model using PowerShell:\n\n"
                "```powershell\n"
                "# Baseline model:\n"
                "python -m src.baseline_model\n\n"
                "# DistilBERT Transformer:\n"
                "python -m src.train_distilbert\n"
                "```"
            )
            return

        with st.spinner("Classifying review with neural network..."):
            try:
                result = predictor.predict_single(user_review)
            except Exception as e:
                st.error(f"Inference error: {e}")
                return

        sentiment = result["sentiment"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]
        model_type = result["model_type"]

        st.divider()

        # Display Prediction Result Card
        if sentiment == "Positive":
            st.markdown(f"""
            <div class="prediction-card-pos">
                <div class="metric-title">Predicted Sentiment</div>
                <div class="metric-value" style="color: #065f46;">🟢 Positive Sentiment</div>
                <p style="margin-top: 0.5rem; margin-bottom: 0; color: #047857;">
                    The model identified positive sentiment with <strong>{confidence * 100:.2f}%</strong> confidence.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card-neg">
                <div class="metric-title">Predicted Sentiment</div>
                <div class="metric-value" style="color: #991b1b;">🔴 Negative Sentiment</div>
                <p style="margin-top: 0.5rem; margin-bottom: 0; color: #b91c1c;">
                    The model identified negative sentiment with <strong>{confidence * 100:.2f}%</strong> confidence.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Class Probability Distributions
        st.subheader("Class Probabilities")
        prob_col1, prob_col2 = st.columns(2)

        with prob_col1:
            st.markdown(f"**Negative Probability: `{probabilities['Negative'] * 100:.2f}%`**")
            st.progress(float(probabilities["Negative"]))

        with prob_col2:
            st.markdown(f"**Positive Probability: `{probabilities['Positive'] * 100:.2f}%`**")
            st.progress(float(probabilities["Positive"]))

        st.caption(f"Calculated via softmax logits from **{model_type}**.")

        # Attention Heatmap Section (if DistilBERT is loaded)
        if model_type == "DistilBERT" and predictor.distilbert_model is not None:
            st.divider()
            with st.expander("🔍 Interactive Attention Visualization (Transformer Interpretability)", expanded=False):
                st.markdown(
                    "Transformer self-attention calculates dynamic pairwise affinity scores between all tokens in the sequence. "
                    "Select a layer and attention head below to inspect internal representation routing."
                )

                attn_col1, attn_col2 = st.columns(2)
                with attn_col1:
                    layer_choice = st.slider("Transformer Layer (0 = initial, 5 = final)", 0, 5, 5)
                with attn_col2:
                    head_choice = st.selectbox("Attention Head", options=["Average (Mean)"] + [f"Head {i}" for i in range(12)], index=0)

                head_arg = "mean" if head_choice == "Average (Mean)" else int(head_choice.split()[1])

                with st.spinner("Extracting attention weights..."):
                    try:
                        attn_tensor, tokens = extract_attention_matrix(
                            model=predictor.distilbert_model,
                            tokenizer=predictor.distilbert_tokenizer,
                            text=user_review,
                            layer_idx=layer_choice,
                            max_length=25,
                            device=predictor.device
                        )
                        fig = plot_attention_heatmap(
                            attention_heads=attn_tensor,
                            tokens=tokens,
                            head_idx=head_arg,
                            layer_idx=layer_choice
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.warning(f"Could not render attention heatmap: {e}")

        # Methodology Explanation Card
        st.divider()
        with st.expander("ℹ️ How Does This Model Work?", expanded=False):
            st.markdown("""
            ### Technical Overview
            - **Tokenization:** Review text is split into subword tokens using WordPiece (`[CLS]`, subwords, `[SEP]`).
            - **Contextual Embeddings:** Unlike static Word2Vec or bag-of-words TF-IDF, DistilBERT processes tokens bidirectionally.
              Words like *not*, *good*, and *great* are interpreted dynamically based on surrounding syntax.
            - **Classification:** The pooled representation (`[CLS]` token) passes through a linear layer and a softmax function
              producing a probability distribution over the classes $\\mathcal{C} = \\{\\text{Negative}, \\text{Positive}\\}$.
            """)

        # Disclaimer
        st.caption(
            "⚠️ **Disclaimer:** Machine learning predictions represent statistical inferences based on the IMDb dataset. "
            "Nuances such as sarcasm, satire, or cultural slang may lead to misclassifications."
        )


if __name__ == "__main__":
    main()
