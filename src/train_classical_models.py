import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

def evaluate_model(model, X, y, name):
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"[{name}] RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    return rmse, mae, r2

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading 3-Layered Feature datasets...")
    train_df = pd.read_csv(os.path.join(data_dir, "train_features.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "validation_features.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_features.csv"))
    
    y_train = train_df['extraversion'].values
    X_train = train_df.drop(columns=['extraversion']).values
    
    y_val = val_df['extraversion'].values
    X_val = val_df.drop(columns=['extraversion']).values
    
    y_test = test_df['extraversion'].values
    X_test = test_df.drop(columns=['extraversion']).values
    
    print(f"Feature Space: {X_train.shape[1]} dimensions (Semantic + Lexical + Behavioral)")
    
    # ---------------------------------------------------------
    # 1. Ridge Regression
    # ---------------------------------------------------------
    print("\n--- Training Ridge Regression ---")
    start_time = time.time()
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=5)
    ridge.fit(X_train, y_train)
    print(f"Ridge training completed in {time.time() - start_time:.2f}s")
    
    evaluate_model(ridge, X_val, y_val, "Ridge (Val)")
    evaluate_model(ridge, X_test, y_test, "Ridge (Test)")
    joblib.dump(ridge, os.path.join(models_dir, "classical_ridge_model.pkl"))
    
    # ---------------------------------------------------------
    # 2. XGBoost
    # ---------------------------------------------------------
    print("\n--- Training XGBoost ---")
    xgb = XGBRegressor(random_state=42, n_jobs=-1)
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', verbose=1, random_state=42, n_jobs=-1)
    start_time = time.time()
    xgb_search.fit(X_train, y_train)
    print(f"XGBoost tuning completed in {time.time() - start_time:.2f}s")
    print(f"Best XGBoost Params: {xgb_search.best_params_}")
    
    best_xgb = xgb_search.best_estimator_
    evaluate_model(best_xgb, X_val, y_val, "XGBoost (Val)")
    evaluate_model(best_xgb, X_test, y_test, "XGBoost (Test)")
    joblib.dump(best_xgb, os.path.join(models_dir, "advanced_xgboost_model.pkl"))
    
    # ---------------------------------------------------------
    # 3. Random Forest
    # ---------------------------------------------------------
    print("\n--- Training Random Forest ---")
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_search = RandomizedSearchCV(rf, rf_params, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', verbose=1, random_state=42, n_jobs=-1)
    start_time = time.time()
    rf_search.fit(X_train, y_train)
    print(f"RF tuning completed in {time.time() - start_time:.2f}s")
    print(f"Best RF Params: {rf_search.best_params_}")
    
    best_rf = rf_search.best_estimator_
    evaluate_model(best_rf, X_val, y_val, "Random Forest (Val)")
    evaluate_model(best_rf, X_test, y_test, "Random Forest (Test)")
    joblib.dump(best_rf, os.path.join(models_dir, "advanced_rf_model.pkl"))
    
    print("\nAll models trained on 3-Layered features and saved to models/ directory!")

if __name__ == "__main__":
    main()
