import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

def load_fused_data(data_dir, split):
    features_df = pd.read_csv(os.path.join(data_dir, f"{split}_features.csv"))
    y = features_df['extraversion'].values
    X_classical = features_df.drop(columns=['extraversion']).values
    
    X_bert = np.load(os.path.join(data_dir, f"{split}_bert_embeddings.npy"))
    
    X_fused = np.hstack((X_classical, X_bert))
    return X_fused, y

def balance_regression_data(X, y, threshold=65, oversample_factor=8):
    """Duplicates samples where y >= threshold to artificially balance the Extraversion distribution."""
    high_idx = np.where(y >= threshold)[0]
    X_high = X[high_idx]
    y_high = y[high_idx]
    
    if len(high_idx) == 0:
        return X, y
        
    X_balanced = np.vstack((X, *[X_high] * oversample_factor))
    y_balanced = np.concatenate((y, *[y_high] * oversample_factor))
    
    # Shuffle
    idx = np.random.permutation(len(y_balanced))
    return X_balanced[idx], y_balanced[idx]

if __name__ == "__main__":
    data_dir = "data"
    models_dir = "models"
    
    print("Loading fused datasets (Classical + BERT)...")
    X_train, y_train = load_fused_data(data_dir, "train")
    X_val, y_val = load_fused_data(data_dir, "validation")
    X_test, y_test = load_fused_data(data_dir, "test")
    
    print(f"Original Train size: {len(y_train)}, Mean Extraversion: {y_train.mean():.2f}")
    
    # Balance data
    X_train_bal, y_train_bal = balance_regression_data(X_train, y_train, threshold=65, oversample_factor=8)
    print(f"Balanced Train size: {len(y_train_bal)}, Mean Extraversion: {y_train_bal.mean():.2f}")
    
    print("\n--- Training XGBoost on Balanced Data ---")
    xgb = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    
    start_time = time.time()
    xgb.fit(X_train_bal, y_train_bal)
    print(f"XGBoost training completed in {time.time() - start_time:.2f}s")
    
    joblib.dump(xgb, os.path.join(models_dir, "advanced_xgboost_model_local.pkl"))
    print("Saved XGBoost to", os.path.join(models_dir, "advanced_xgboost_model_local.pkl"))
    
    print("\n--- Training Random Forest on Balanced Data ---")
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
    
    start_time = time.time()
    rf.fit(X_train_bal, y_train_bal)
    print(f"RF training completed in {time.time() - start_time:.2f}s")
    
    joblib.dump(rf, os.path.join(models_dir, "random_forest_model_local.pkl"))
    print("Saved RF to", os.path.join(models_dir, "random_forest_model_local.pkl"))
    
    print("\n--- Evaluation on Test Set (Original Distribution) ---")
    for name, model in [("XGBoost", xgb), ("Random Forest", rf)]:
        preds = model.predict(X_test)
        print(f"{name} - RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}, R2: {r2_score(y_test, preds):.4f}")
