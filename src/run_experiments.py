import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import time

def load_data(data_dir):
    print("Loading local training data...")
    # Load Training Data
    train_df = pd.read_csv(os.path.join(data_dir, "train_features.csv"))
    y_train = train_df['extraversion'].values
    X_train_class = train_df.drop(columns=['extraversion']).values
    X_train_bert = np.load(os.path.join(data_dir, "train_bert_embeddings.npy"))
    X_train = np.hstack((X_train_class, X_train_bert))
    
    # Load Test Data
    test_df = pd.read_csv(os.path.join(data_dir, "test_features.csv"))
    y_test = test_df['extraversion'].values
    X_test_class = test_df.drop(columns=['extraversion']).values
    X_test_bert = np.load(os.path.join(data_dir, "test_bert_embeddings.npy"))
    X_test = np.hstack((X_test_class, X_test_bert))
    
    return X_train, y_train, X_test, y_test

def run_experiments():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    
    X_train, y_train, X_test, y_test = load_data(data_dir)
    print(f"Training Data Shape: {X_train.shape}")
    
    results = []
    
    # XGBoost Configurations
    xgb_configs = [
        {"name": "XGB_Exp5_AggressiveDeep", "params": {"n_estimators": 200, "max_depth": 10, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.7, "random_state": 42, "n_jobs": -1, "tree_method": "hist"}},
        {"name": "XGB_Exp6_Balanced", "params": {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7, "gamma": 0.1, "random_state": 42, "n_jobs": -1, "tree_method": "hist"}},
    ]
    
    # Random Forest Configurations
    rf_configs = [
        {"name": "RF_Exp3_MoreTreesDeeper", "params": {"n_estimators": 300, "max_depth": 15, "min_samples_split": 5, "random_state": 42, "n_jobs": -1}},
        {"name": "RF_Exp4_ShallowRobust", "params": {"n_estimators": 500, "max_depth": 10, "min_samples_split": 10, "random_state": 42, "n_jobs": -1}},
        {"name": "RF_Exp5_UltraDeep", "params": {"n_estimators": 150, "max_depth": 20, "min_samples_split": 2, "random_state": 42, "n_jobs": -1}}, 
        {"name": "RF_Exp6_HighlyConstrained", "params": {"n_estimators": 400, "max_depth": 8, "min_samples_split": 20, "random_state": 42, "n_jobs": -1}},
    ]
    
    best_xgb_r2 = 0.5764 # Our reigning champion to beat (from Exp3)
    best_rf_r2 = 0.4670  # Our reigning champion to beat
    excel_path = os.path.join(base_dir, "experiment_results.xlsx")
    
    # Load existing results if they exist so we don't overwrite them
    if os.path.exists(excel_path):
        results = pd.read_excel(excel_path).to_dict('records')
    else:
        results = []
    
    def save_results(results_list):
        df_results = pd.DataFrame(results_list)
        df_results.to_excel(excel_path, index=False)
    
    print("\n--- Running XGBoost Experiments ---")
    for config in xgb_configs:
        print(f"\nStarting {config['name']}...")
        model = xgb.XGBRegressor(**config['params'])
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        
        preds = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print(f"Finished in {elapsed:.1f}s | RMSE: {rmse:.4f} | R2: {r2:.4f}")
        
        results.append({
            "Experiment": config["name"],
            "Model": "XGBoost",
            "Parameters": str(config["params"]),
            "RMSE": rmse,
            "R2": r2,
            "Time (s)": elapsed
        })
        save_results(results)
        
        if r2 > best_xgb_r2:
            print(f"*** NEW BEST XGBOOST MODEL! R2: {r2:.4f} ***")
            best_xgb_r2 = r2
            model_path = os.path.join(models_dir, "advanced_xgboost_model_local.pkl")
            joblib.dump(model, model_path)
    
    print("\n--- Running Random Forest Experiments ---")
    for config in rf_configs:
        print(f"\nStarting {config['name']}...")
        model = RandomForestRegressor(**config['params'])
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        
        preds = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print(f"Finished in {elapsed:.1f}s | RMSE: {rmse:.4f} | R2: {r2:.4f}")
        
        results.append({
            "Experiment": config["name"],
            "Model": "Random Forest",
            "Parameters": str(config["params"]),
            "RMSE": rmse,
            "R2": r2,
            "Time (s)": elapsed
        })
        save_results(results)
        
        if r2 > best_rf_r2:
            print(f"*** NEW BEST RF MODEL! R2: {r2:.4f} ***")
            best_rf_r2 = r2
            model_path = os.path.join(models_dir, "random_forest_model_local.pkl")
            joblib.dump(model, model_path)
            
    print(f"\nAll experiments complete! Results saved to {excel_path}")

if __name__ == "__main__":
    run_experiments()
