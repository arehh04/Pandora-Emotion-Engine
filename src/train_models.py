import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X, y, name):
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"{name} Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}\n")
    return rmse, mae, r2

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Classical Ridge Model
    # ---------------------------------------------------------
    print("--- Training Classical Ridge Model ---")
    train_feat = pd.read_csv(os.path.join(data_dir, "train_features.csv"))
    val_feat = pd.read_csv(os.path.join(data_dir, "validation_features.csv"))
    test_feat = pd.read_csv(os.path.join(data_dir, "test_features.csv"))
    
    y_train = train_feat['extraversion'].values
    X_train = train_feat.drop(columns=['extraversion']).values
    
    y_val = val_feat['extraversion'].values
    X_val = val_feat.drop(columns=['extraversion']).values
    
    y_test = test_feat['extraversion'].values
    X_test = test_feat.drop(columns=['extraversion']).values
    
    # Train
    classical_ridge = RidgeCV(alphas=np.logspace(-3, 3, 7), cv=5)
    classical_ridge.fit(X_train, y_train)
    
    # Evaluate
    evaluate_model(classical_ridge, X_val, y_val, "Classical Ridge (Validation)")
    evaluate_model(classical_ridge, X_test, y_test, "Classical Ridge (Test)")
    
    joblib.dump(classical_ridge, os.path.join(models_dir, "classical_ridge_model.pkl"))
    print("Saved classical_ridge_model.pkl\n")
    
    # ---------------------------------------------------------
    # 2. BERT Ridge Model
    # ---------------------------------------------------------
    print("--- Training BERT Ridge Model ---")
    train_bert = np.load(os.path.join(data_dir, "train_bert_embeddings.npy"))
    val_bert = np.load(os.path.join(data_dir, "validation_bert_embeddings.npy"))
    test_bert = np.load(os.path.join(data_dir, "test_bert_embeddings.npy"))
    
    # Train
    bert_ridge = RidgeCV(alphas=np.logspace(-3, 3, 7), cv=5)
    bert_ridge.fit(train_bert, y_train)
    
    # Evaluate
    evaluate_model(bert_ridge, val_bert, y_val, "BERT Ridge (Validation)")
    evaluate_model(bert_ridge, test_bert, y_test, "BERT Ridge (Test)")
    
    joblib.dump(bert_ridge, os.path.join(models_dir, "bert_ridge_model.pkl"))
    print("Saved bert_ridge_model.pkl\n")

if __name__ == "__main__":
    main()
