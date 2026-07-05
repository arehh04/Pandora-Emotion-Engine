"""
Token-level SHAP explainability for the Fine-Tuned BERT Regressor.
Generates colour-coded HTML showing which tokens pushed the score up or down.

Usage:
    from src.explainability.shap_bert_tokens import explain_text
    html, top_tokens = explain_text("I love going to parties", model, tokenizer)
"""
import os
import numpy as np
import torch
from transformers import BertTokenizer


def _bert_predict_fn(texts: list, model, tokenizer, device: str, max_len: int = 256):
    """Wrapper for SHAP: takes list of strings, returns np.array of scores."""
    model.eval()
    scores = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            ids  = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            score, _ = model(ids, mask)
            scores.append(score.item())
    return np.array(scores)


def explain_text(text: str, model, tokenizer, device: str = "cpu",
                 max_display: int = 10) -> tuple[str, list[dict]]:
    """
    Compute token-level SHAP values for a single input text.

    Returns:
        html_str   : coloured HTML string (green = positive, red = negative)
        top_tokens : list of {token, shap_value} dicts sorted by |impact|
    """
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    # Masker and explainer
    masker   = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(
        lambda texts: _bert_predict_fn(texts, model, tokenizer, device),
        masker,
        output_names=["Extraversion Score"]
    )

    shap_values = explainer([text], batch_size=1, fixed_context=1)

    tokens = shap_values.data[0]
    values = shap_values.values[0]

    if hasattr(values, '__len__') and len(values.shape) > 1:
        values = values[:, 0]

    # Build coloured HTML
    html_parts = []
    for tok, val in zip(tokens, values):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        # Normalise value to colour intensity 0–255
        intensity = min(255, int(abs(val) / (max(abs(values)) + 1e-8) * 255))
        if val >= 0:
            bg = f"rgba(0, {intensity}, 100, 0.4)"
            border = f"rgba(0,229,100,0.6)"
        else:
            bg = f"rgba({intensity}, 0, 0, 0.4)"
            border = f"rgba(255,80,80,0.6)"

        display_tok = tok.replace("##", "")   # merge WordPiece sub-tokens visually
        html_parts.append(
            f'<span title="SHAP: {val:+.4f}" style="'
            f'background:{bg}; border:1px solid {border}; border-radius:4px; '
            f'padding:2px 5px; margin:2px; display:inline-block; '
            f'font-family:monospace; font-size:0.92rem; color:#E6F1FF;">'
            f'{display_tok}</span>'
        )

    html_str = (
        '<div style="line-height:2.4; padding:1rem; background:rgba(17,34,64,0.8); '
        'border:1px solid rgba(0,229,255,0.2); border-radius:12px;">'
        + " ".join(html_parts) +
        '</div>'
    )

    # Top tokens by |SHAP|
    token_shap = [
        {"token": t.replace("##", ""), "shap_value": float(v)}
        for t, v in zip(tokens, values)
        if t not in ("[CLS]", "[SEP]", "[PAD]")
    ]
    token_shap.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return html_str, token_shap[:max_display]


def get_attention_heatmap(text: str, model, tokenizer,
                          device: str = "cpu", layer: int = 11) -> tuple:
    """
    Extract averaged attention weights from the last Transformer layer.
    Returns (tokens, attention_matrix) where attention_matrix is (N, N).
    """
    model.eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    ids  = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        _, attentions = model(ids, mask)

    # attentions: tuple of (batch, heads, seq, seq)
    attn_layer = attentions[layer][0]           # (heads, seq, seq)
    attn_avg   = attn_layer.mean(dim=0).cpu().numpy()  # (seq, seq)

    tokens = tokenizer.convert_ids_to_tokens(ids[0].cpu().numpy())
    # Keep only non-padding tokens
    seq_len = mask.sum().item()
    tokens  = tokens[:seq_len]
    attn_avg = attn_avg[:seq_len, :seq_len]

    return tokens, attn_avg
