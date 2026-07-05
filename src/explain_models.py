import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    
    print("Loading model and data...")
    classical_ridge = joblib.load(os.path.join(models_dir, "classical_ridge_model.pkl"))
    tfidf = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    
    val_feat = pd.read_csv(os.path.join(data_dir, "validation_features.csv"))
    X_val = val_feat.drop(columns=['extraversion'])
    
    # Reconstruct readable feature names
    ling_emo_cols = [c for c in X_val.columns if not c.startswith('tfidf_')]
    tfidf_words = tfidf.get_feature_names_out()
    
    feature_names = ling_emo_cols + [f"TFIDF: {w}" for w in tfidf_words]
    # Ensure they match
    if len(feature_names) == len(X_val.columns):
        X_val.columns = feature_names
    else:
        print("Warning: Feature counts do not match")
        
    print("Calculating SHAP values...")
    # Background dataset for SHAP
    explainer = shap.LinearExplainer(classical_ridge, X_val, feature_names=feature_names)
    shap_values = explainer(X_val)
    
    print("Generating Plots...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_val, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "shap_summary_plot.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "shap_bar_plot.png"))
    plt.close()
    
    print("Done! SHAP plots saved to models/ directory.")

if __name__ == "__main__":
    main()
