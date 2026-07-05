import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

def retrain_rf_locally():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    
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
    
    print(f"Training Data Shape: {X_train.shape}")
    print("Training Random Forest locally...")
    
    # Standard sensible parameters for a quick Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("Evaluating local Random Forest model...")
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Local RF Test RMSE: {rmse:.4f}")
    print(f"Local RF Test R2: {r2:.4f}")
    
    # Save the model locally so it is Windows-compatible
    model_path = os.path.join(models_dir, "random_forest_model_local.pkl")
    joblib.dump(model, model_path)
    
    print("Random Forest model successfully saved locally!")

if __name__ == "__main__":
    retrain_rf_locally()
