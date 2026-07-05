import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertModel
import spacy
from src.extract_classical_features import compute_linguistic_features, compute_emotional_features, load_nrc_lexicon

def extract_features(text, data_dir, models_dir):
    # Load models
    nlp = spacy.load("en_core_web_sm")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    nrc_lex = load_nrc_lexicon(os.path.join(data_dir, "NRC-Emotion-Lexicon-Senselevel-v0.92.txt"))
    vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'feature_scaler.pkl'))
    
    # 1. Classical Features
    # Lemmatization for TFIDF and NRC
    doc = nlp(text)
    lemmatized = " ".join([token.lemma_ for token in doc if not token.is_stop])
    
    ling_df = compute_linguistic_features([text], nlp)
    emo_df = compute_emotional_features(pd.Series([lemmatized]), nrc_lex)
    
    tfidf_vec = vectorizer.transform([lemmatized])
    tfidf_df = pd.DataFrame(tfidf_vec.toarray(), columns=[f"tfidf_{i}" for i in range(2000)])
    
    combined = pd.concat([ling_df, emo_df, tfidf_df], axis=1)
    
    # We must match the exact column order of the training data
    train_cols = pd.read_csv(os.path.join(data_dir, 'train_features.csv'), nrows=0).drop(columns=['extraversion']).columns
    
    df_classical_ordered = pd.DataFrame(columns=train_cols)
    for col in train_cols:
        df_classical_ordered.loc[0, col] = combined.get(col, 0).iloc[0] if isinstance(combined.get(col, 0), pd.Series) else combined.get(col, 0)
        
    scaled_classical = scaler.transform(df_classical_ordered)
    
    # 4. BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].numpy() # CLS token
        
    # Combine all
    X_final = np.hstack((scaled_classical, embedding))
    
    return X_final, train_cols.tolist()

def generate_use_case():
    text = "I want to spend my holiday doing my hobby which is reading"
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    
    print("Extracting features for text...")
    X_input, class_cols = extract_features(text, data_dir, models_dir)
    
    print("Loading model...")
    model_path = os.path.join(models_dir, "advanced_xgboost_model_local.pkl")
    model = joblib.load(model_path)
    
    print("Predicting...")
    pred = model.predict(X_input)[0]
    print(f"Prediction: {pred:.2f}")
    
    print("Generating SHAP waterfall...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_input)
    
    # Create feature names
    bert_cols = [f"bert_{i}" for i in range(768)]
    all_feature_names = class_cols + bert_cols
    shap_values.feature_names = all_feature_names
    
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.title(f"SHAP Waterfall: 'I want to spend my holiday doing my hobby which is reading'\nPredicted Extraversion: {pred:.2f}/5.0", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'use_case_shap.png'))
    plt.close()
    
    print("Done! Image saved to use_case_shap.png")
    
if __name__ == "__main__":
    generate_use_case()
