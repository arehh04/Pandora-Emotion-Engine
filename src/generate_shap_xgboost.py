import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def generate_shap_plots():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    
    print("Loading model and data...")
    try:
        model = joblib.load(os.path.join(models_dir, "advanced_xgboost_model_local.pkl"))
        
        # Load a sample of the test set to save time on SHAP calculation
        test_df = pd.read_csv(os.path.join(data_dir, "test_features.csv"))
        y_test = test_df['extraversion'].values
        X_classical = test_df.drop(columns=['extraversion'])
        classical_cols = X_classical.columns.tolist()
        
        X_bert = np.load(os.path.join(data_dir, "test_bert_embeddings.npy"))
        bert_cols = [f"bert_{i}" for i in range(X_bert.shape[1])]
        
        X_fused = np.hstack((X_classical.values, X_bert))
        all_cols = classical_cols + bert_cols
        
        # Sample 500 rows for SHAP to avoid massive memory/time usage
        # SHAP explains the model, we just need a representative sample
        sample_idx = np.random.choice(X_fused.shape[0], 500, replace=False)
        X_sample = X_fused[sample_idx]
        y_sample = y_test[sample_idx]
        
        X_df = pd.DataFrame(X_sample, columns=all_cols)
        
        print("Calculating SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_df)
        
        # 1. Summary Plot (Beeswarm)
        print("Generating Summary Plot...")
        plt.figure()
        shap.summary_plot(shap_values, X_df, show=False, max_display=10)
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, "xgb_shap_summary.png"))
        plt.close()
        
        # 2. Bar Plot (Mean Absolute Importance)
        print("Generating Bar Plot...")
        plt.figure()
        shap.plots.bar(shap_values, show=False, max_display=10)
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, "xgb_shap_bar.png"))
        plt.close()
        
        # 3. Waterfall Plots for 3 specific cases
        print("Generating Waterfall Plots...")
        preds = model.predict(X_sample)
        
        highest_idx = np.argmax(preds)
        lowest_idx = np.argmin(preds)
        
        # Misprediction (highest absolute error)
        errors = np.abs(preds - y_sample)
        worst_idx = np.argmax(errors)
        
        plt.figure()
        shap.plots.waterfall(shap_values[highest_idx], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, "xgb_shap_waterfall_highest.png"))
        plt.close()
        
        plt.figure()
        shap.plots.waterfall(shap_values[lowest_idx], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, "xgb_shap_waterfall_lowest.png"))
        plt.close()
        
        plt.figure()
        shap.plots.waterfall(shap_values[worst_idx], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, "xgb_shap_waterfall_worst.png"))
        plt.close()
        
        print("SHAP plots generated successfully.")
        
    except Exception as e:
        print(f"Error generating SHAP: {e}")

if __name__ == "__main__":
    generate_shap_plots()
