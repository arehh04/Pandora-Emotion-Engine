"""
Evaluate ALL models (Ridge, XGBoost, Random Forest, Fine-tuned BERT)
on the test set. Generates full comparison table + chart.

Usage:
    python -m src.evaluation.evaluate_all_models
"""
import os, json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def evaluate_classical(name: str, model_path: str, X_test, y_test) -> dict:
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    return {
        "model": name,
        "RMSE":  round(rmse(y_test, preds), 4),
        "MAE":   round(float(mean_absolute_error(y_test, preds)), 4),
        "R²":    round(float(r2_score(y_test, preds)), 4),
    }


def evaluate_bert_regressor(model_pt_path: str, X_test_bert, y_test) -> dict:
    """
    Load the fine-tuned BERT regressor and evaluate.
    Falls back gracefully if model not yet downloaded from Colab.
    """
    if not os.path.exists(model_pt_path):
        print(f"  ⚠️  Fine-tuned BERT model not found at: {model_pt_path}")
        print("  →  Please train on Colab and download bert_regressor_best.pt")
        return {"model": "Fine-tuned BERT", "RMSE": "N/A", "MAE": "N/A", "R²": "N/A"}

    import torch
    from src.models.bert_regressor import BertRegressorModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = BertRegressorModel()
    model.load_state_dict(torch.load(model_pt_path, map_location=device))
    model.eval().to(device)

    # X_test_bert shape: (N, 768) – raw BERT embeddings
    # For the fine-tuned model we need tokenised text, so we skip re-tokenisation
    # and instead use the frozen embeddings as a proxy (quick evaluation mode).
    # Full text-based evaluation requires raw text → use Colab metrics JSON if available.
    bert_metrics_path = os.path.join(MODELS_DIR, "bert_test_metrics.json")
    if os.path.exists(bert_metrics_path):
        with open(bert_metrics_path) as f:
            m = json.load(f)
        return {
            "model": "Fine-tuned BERT",
            "RMSE":  round(m["test_rmse"], 4),
            "MAE":   round(m["test_mae"],  4),
            "R²":    round(m["test_r2"],   4),
        }

    print("  ⚠️  bert_test_metrics.json not found. Using embedding-proxy evaluation.")
    # Proxy: pass frozen embeddings through the regressor head only
    with torch.no_grad():
        t = torch.tensor(X_test_bert, dtype=torch.float32).to(device)
        # Direct head pass (no BERT backbone) — approximate
        preds = (model.regressor(t).squeeze(-1) * 99.0).cpu().numpy()
    return {
        "model": "Fine-tuned BERT (proxy)",
        "RMSE":  round(rmse(y_test, preds), 4),
        "MAE":   round(float(mean_absolute_error(y_test, preds)), 4),
        "R²":    round(float(r2_score(y_test, preds)), 4),
    }


def plot_comparison(results: list[dict], save_path: str):
    models = [r["model"] for r in results]
    rmse_v = [r["RMSE"] if isinstance(r["RMSE"], float) else 0 for r in results]
    mae_v  = [r["MAE"]  if isinstance(r["MAE"],  float) else 0 for r in results]
    r2_v   = [r["R²"]   if isinstance(r["R²"],   float) else 0 for r in results]

    x      = np.arange(len(models))
    width  = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0A192F')
    for ax in axes:
        ax.set_facecolor('#112240')

    # Left: RMSE + MAE
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, rmse_v, width, label='RMSE', color='#FF6B6B', alpha=0.85)
    bars2 = ax1.bar(x + width/2, mae_v,  width, label='MAE',  color='#FFB347', alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=15, ha='right', color='#80d8ff', fontsize=9)
    ax1.set_ylabel('Error (Lower is Better)', color='#80d8ff')
    ax1.set_title('RMSE & MAE Comparison', color='#00e5ff', fontweight='bold')
    ax1.tick_params(colors='#80d8ff')
    ax1.legend(facecolor='#0A192F', labelcolor='#E6F1FF')
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{h:.2f}',
                     ha='center', va='bottom', fontsize=8, color='#E6F1FF')

    # Right: R²
    ax2 = axes[1]
    colors = ['#546e7a', '#00b0ff', '#00e5ff', '#ce93d8']
    bars3  = ax2.bar(x, r2_v, 0.5, color=colors[:len(models)], alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=15, ha='right', color='#80d8ff', fontsize=9)
    ax2.set_ylabel('R² Score (Higher is Better)', color='#80d8ff')
    ax2.set_title('R² Score Comparison', color='#00e5ff', fontweight='bold')
    ax2.tick_params(colors='#80d8ff')
    for bar in bars3:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.002, f'{h:.3f}',
                     ha='center', va='bottom', fontsize=9, color='#E6F1FF', fontweight='bold')

    plt.suptitle('All Model Performance Comparison — Pandora System', color='#00e5ff',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#0A192F')
    plt.close()
    print(f"Comparison chart saved -> {save_path}")


def plot_training_curve(log_csv: str, save_path: str):
    if not os.path.exists(log_csv):
        print(f"Training log not found: {log_csv}")
        return
    df  = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0A192F')
    for ax in axes:
        ax.set_facecolor('#112240')

    # Training loss
    axes[0].plot(df["epoch"], df["train_loss"], marker='o', color='#00e5ff', linewidth=2)
    axes[0].set_title('Training Loss per Epoch', color='#00e5ff', fontweight='bold')
    axes[0].set_xlabel('Epoch', color='#80d8ff')
    axes[0].set_ylabel('MSE Loss', color='#80d8ff')
    axes[0].tick_params(colors='#80d8ff')

    # Val RMSE
    axes[1].plot(df["epoch"], df["val_rmse"], marker='s', color='#FF6B6B', linewidth=2, label='RMSE')
    axes[1].plot(df["epoch"], df["val_r2"],   marker='^', color='#69F0AE', linewidth=2, label='R²')
    axes[1].set_title('Validation Metrics per Epoch', color='#00e5ff', fontweight='bold')
    axes[1].set_xlabel('Epoch', color='#80d8ff')
    axes[1].tick_params(colors='#80d8ff')
    axes[1].legend(facecolor='#0A192F', labelcolor='#E6F1FF')

    plt.suptitle('Fine-Tuned BERT — Training Curve', color='#00e5ff', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#0A192F')
    plt.close()
    print(f"Training curve saved -> {save_path}")


def main():
    # Load test features
    print("Loading test data...")
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"))
    y_test   = test_df["extraversion"].values
    X_test_c = test_df.drop(columns=["extraversion"]).values
    X_test_b = np.load(os.path.join(DATA_DIR, "test_bert_embeddings.npy"))
    X_test   = np.hstack((X_test_c, X_test_b))

    results = []
    print("\nEvaluating models...")

    results.append(evaluate_classical(
        "Ridge (Baseline)",
        os.path.join(MODELS_DIR, "classical_ridge_model.pkl"),
        X_test_c[:, :1018], y_test
    ))
    results.append(evaluate_classical(
        "XGBoost",
        os.path.join(MODELS_DIR, "advanced_xgboost_model_local.pkl"),
        X_test, y_test
    ))

    # Try Random Forest
    rf_path = os.path.join(MODELS_DIR, "random_forest_model_local.pkl")
    if os.path.exists(rf_path):
        results.append(evaluate_classical("Random Forest", rf_path, X_test, y_test))

    results.append(evaluate_bert_regressor(
        os.path.join(MODELS_DIR, "bert_regressor_best.pt"),
        X_test_b, y_test
    ))

    # Print table
    print("\n" + "="*60)
    print(f"{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-"*60)
    for r in results:
        print(f"{r['model']:<25} {str(r['RMSE']):>8} {str(r['MAE']):>8} {str(r['R²']):>8}")
    print("="*60)

    # Save CSV
    pd.DataFrame(results).to_csv(os.path.join(MODELS_DIR, "full_model_comparison.csv"), index=False)
    print("\nFull comparison CSV saved.")

    # Charts
    plot_comparison(results, os.path.join(MODELS_DIR, "full_model_comparison.png"))
    plot_training_curve(
        os.path.join(MODELS_DIR, "bert_training_log.csv"),
        os.path.join(MODELS_DIR, "bert_training_curve.png")
    )


if __name__ == "__main__":
    main()
