"""Attention Visualization Module for DistilBERT.

Extracts multi-head self-attention weights from Transformer layers and
renders token-to-token attention heatmaps.

Interpretability Note:
Attention weights illustrate the contextual token-to-token routing mechanism inside
the Transformer architecture (query-key dot-product interactions). They should NOT
be interpreted as infallible causal proof of model decisions or human-aligned explanations.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.preprocessing import preprocess_for_transformer, validate_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("visualize_attention")


def extract_attention_matrix(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    layer_idx: int = -1,
    max_length: int = 40,
    device: str = "cpu"
) -> Tuple[np.ndarray, List[str]]:
    """Runs a forward pass requesting attention tensors and returns tokenized attention matrix.

    Args:
        model: Hugging Face SequenceClassification model supporting output_attentions.
        tokenizer: Hugging Face tokenizer corresponding to the model.
        text: Input review text string.
        layer_idx: Transformer layer to extract (-1 designates the final layer).
        max_length: Maximum tokens to process for clean visual layout.
        device: Device string ('cpu' or 'cuda').

    Returns:
        Tuple of (attention_tensor_numpy, list_of_token_strings).
        Shape of attention_tensor_numpy is (num_heads, seq_len, seq_len).
    """
    clean_text = preprocess_for_transformer(validate_text(text))

    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    model.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions
    if attentions is None:
        raise ValueError(
            "Model did not return attention weights. Ensure output_attentions=True is supported."
        )

    # DistilBERT has 6 layers (0 to 5)
    num_layers = len(attentions)
    if layer_idx < 0:
        actual_layer = num_layers + layer_idx
    else:
        actual_layer = layer_idx

    if actual_layer < 0 or actual_layer >= num_layers:
        raise IndexError(f"Layer index {layer_idx} out of range for model with {num_layers} layers.")

    # Shape: (batch_size, num_heads, seq_len, seq_len) -> extract batch 0
    layer_attention = attentions[actual_layer][0].cpu().numpy()

    # Convert token IDs to readable tokens
    input_ids = inputs["input_ids"][0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    return layer_attention, tokens


def plot_attention_heatmap(
    attention_heads: np.ndarray,
    tokens: List[str],
    head_idx: Union[int, str] = 0,
    layer_idx: int = -1,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Renders and optionally saves a 2D token-to-token attention heatmap.

    Args:
        attention_heads: 3D array of shape (num_heads, seq_len, seq_len).
        tokens: List of token strings corresponding to rows and columns.
        head_idx: Integer head index or 'mean' to average across all heads.
        layer_idx: Layer index displayed in the title.
        output_path: Optional file path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    if head_idx == "mean":
        matrix = np.mean(attention_heads, axis=0)
        head_title = "Average across all Heads"
    else:
        matrix = attention_heads[int(head_idx)]
        head_title = f"Head #{head_idx}"

    seq_len = len(tokens)
    fig_size = max(6, min(14, seq_len * 0.45))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    sns.heatmap(
        matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="Blues",
        cbar_kws={"label": "Self-Attention Weight"},
        ax=ax,
        square=True
    )

    ax.set_title(
        f"DistilBERT Self-Attention Matrix\n(Layer {layer_idx}, {head_title})",
        pad=15,
        fontsize=12,
        fontweight="bold"
    )
    ax.set_xlabel("Key Tokens (Attended To)", labelpad=10)
    ax.set_ylabel("Query Tokens (Attending From)", labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    # Explanatory scientific caution note
    fig.text(
        0.5, -0.05,
        "Note: Self-attention weights reflect internal token routing and contextual mixing.\n"
        "They illustrate representation dynamics rather than definitive causal feature attribution.",
        ha="center", fontsize=8, color="#555555", style="italic"
    )

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved attention heatmap to {output_path}.")

    return fig


def visualize_review_attention(
    text: str,
    model_dir: str = "models/distilbert-sentiment",
    output_path: str = "reports/figures/attention_heatmap.png",
    head_idx: Union[int, str] = 0,
    layer_idx: int = -1,
    max_length: int = 35
) -> Path:
    """End-to-end attention visualization workflow for a given review text.

    Args:
        text: Input review text.
        model_dir: Path to directory containing saved DistilBERT model.
        output_path: Destination image path.
        head_idx: Head index (0-11 for DistilBERT) or 'mean'.
        layer_idx: Transformer layer index (-1 for final layer).
        max_length: Maximum tokens to display for clean visual layout.

    Returns:
        Path to the saved figure.
    """
    model_path = Path(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if (model_path / "config.json").is_file():
        logger.info(f"Loading model from local checkpoint: {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            attn_implementation="eager"
        )
    else:
        fallback_name = "distilbert-base-uncased"
        logger.warning(
            f"Trained checkpoint not found at {model_path}. "
            f"Falling back to pretrained base checkpoint: {fallback_name} for attention structure inspection."
        )
        tokenizer = AutoTokenizer.from_pretrained(fallback_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            fallback_name,
            num_labels=2,
            attn_implementation="eager"
        )

    attention_heads, tokens = extract_attention_matrix(
        model=model,
        tokenizer=tokenizer,
        text=text,
        layer_idx=layer_idx,
        max_length=max_length,
        device=device
    )

    out_file = Path(output_path)
    plot_attention_heatmap(
        attention_heads=attention_heads,
        tokens=tokens,
        head_idx=head_idx,
        layer_idx=layer_idx,
        output_path=out_file
    )
    plt.close()
    return out_file


if __name__ == "__main__":
    sample_text = "The screenplay was brilliant and the acting was surprisingly poignant."
    visualize_review_attention(sample_text)
