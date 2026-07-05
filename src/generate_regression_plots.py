import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def generate_regression_plots():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    
    print("Loading test data and model...")
    # Load test data
    test_features_path = os.path.join(data_dir, "test_features.csv")
    test_bert_path = os.path.join(data_dir, "test_bert_embeddings.npy")
    
    test_df = pd.read_csv(test_features_path)
    X_test_classical = test_df.drop(columns=['extraversion']).values
    y_test = test_df['extraversion'].values
    X_test_bert = np.load(test_bert_path)
    X_test = np.hstack((X_test_classical, X_test_bert))
    
    # Load XGBoost model
    model_path = os.path.join(models_dir, "advanced_xgboost_model_local.pkl")
    model = joblib.load(model_path)
    
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    
    print("Plotting Actual vs Predicted...")
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, color='teal', edgecolor='k')
    
    # Plot perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    
    plt.title('Actual vs. Predicted Extraversion Scores', fontsize=16)
    plt.xlabel('Actual Extraversion Score (Ground Truth)', fontsize=14)
    plt.ylabel('Predicted Extraversion Score', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'actual_vs_predicted.png'))
    plt.close()
    
    print("Plotting Residuals...")
    residuals = y_test - y_pred
    
    plt.figure(figsize=(10, 8))
    sns.histplot(residuals, bins=40, kde=True, color='purple')
    plt.title('Distribution of Residuals (Errors)', fontsize=16)
    plt.xlabel('Residual (Actual - Predicted)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'residual_histogram.png'))
    plt.close()
    
    # Residual scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred, residuals, alpha=0.5, color='coral', edgecolor='k')
    plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
    plt.title('Residual Plot (Predicted vs Error)', fontsize=16)
    plt.xlabel('Predicted Extraversion Score', fontsize=14)
    plt.ylabel('Residual (Actual - Predicted)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'residual_scatter.png'))
    plt.close()
    
    print("Regression plots generated successfully in models/ directory!")

if __name__ == "__main__":
    generate_regression_plots()
