import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

def load_fused_data(data_dir, split):
    # Load classical features (TF-IDF + Ling + Emo)
    features_df = pd.read_csv(os.path.join(data_dir, f"{split}_features.csv"))
    y = features_df['extraversion'].values
    X_classical = features_df.drop(columns=['extraversion']).values
    
    # Load BERT embeddings
    X_bert = np.load(os.path.join(data_dir, f"{split}_bert_embeddings.npy"))
    
    # Fuse them
    X_fused = np.hstack((X_classical, X_bert))
    return X_fused, y

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading fused datasets (Classical + BERT)...")
    X_train, y_train = load_fused_data(data_dir, "train")
    X_val, y_val = load_fused_data(data_dir, "validation")
    X_test, y_test = load_fused_data(data_dir, "test")
    
    print(f"Fused Feature Space: {X_train.shape[1]} dimensions (Classical + BERT)")
    
    # 1. XGBoost with RandomizedSearchCV
    print("\n--- Training XGBoost (Feature Fusion) ---")
    xgb = XGBRegressor(random_state=42, n_jobs=-1)
    
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', verbose=2, random_state=42, n_jobs=-1)
    start_time = time.time()
    xgb_search.fit(X_train, y_train)
    print(f"XGBoost tuning completed in {time.time() - start_time:.2f}s")
    print(f"Best XGBoost Params: {xgb_search.best_params_}")
    
    best_xgb = xgb_search.best_estimator_
    
    # Evaluate XGBoost
    val_preds_xgb = best_xgb.predict(X_val)
    test_preds_xgb = best_xgb.predict(X_test)
    
    print(f"XGBoost (Val)   - RMSE: {np.sqrt(mean_squared_error(y_val, val_preds_xgb)):.4f}, R2: {r2_score(y_val, val_preds_xgb):.4f}")
    print(f"XGBoost (Test)  - RMSE: {np.sqrt(mean_squared_error(y_test, test_preds_xgb)):.4f}, R2: {r2_score(y_test, test_preds_xgb):.4f}")
    
    joblib.dump(best_xgb, os.path.join(models_dir, "advanced_xgboost_model.pkl"))
    
    # 2. Random Forest
    print("\n--- Training Random Forest (Feature Fusion) ---")
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_search = RandomizedSearchCV(rf, rf_params, n_iter=5, cv=3, scoring='neg_root_mean_squared_error', verbose=2, random_state=42, n_jobs=-1)
    start_time = time.time()
    rf_search.fit(X_train, y_train)
    print(f"RF tuning completed in {time.time() - start_time:.2f}s")
    print(f"Best RF Params: {rf_search.best_params_}")
    
    best_rf = rf_search.best_estimator_
    
    # Evaluate RF
    test_preds_rf = best_rf.predict(X_test)
    print(f"RF (Test)       - RMSE: {np.sqrt(mean_squared_error(y_test, test_preds_rf)):.4f}, R2: {r2_score(y_test, test_preds_rf):.4f}")
    
    joblib.dump(best_rf, os.path.join(models_dir, "advanced_rf_model.pkl"))
    print("\nModels saved to models/ directory!")

if __name__ == "__main__":
    main()
