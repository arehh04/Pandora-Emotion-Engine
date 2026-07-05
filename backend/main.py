import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import spacy
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import BertTokenizer, BertModel
import redis
import hashlib
import json
from bloom_filter2 import BloomFilter

from src.extract_classical_features import compute_linguistic_features, compute_emotional_features, load_nrc_lexicon
from src.models.bert_regressor import BertRegressorModel
from src.explainability.shap_bert_tokens import explain_text

app = FastAPI(title="Pandora API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str
    model: str = "Fine-Tuned BERT"

# Globals for models and cache
pkg = {}
cache = {}

try:
    cache['redis'] = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)
    cache['redis'].ping()
    cache['bloom'] = BloomFilter(max_elements=100000, error_rate=0.01)
    cache['enabled'] = True
    print("Redis and Bloom Filter initialized successfully.")
except Exception as e:
    print(f"Redis not available, caching disabled. Error: {e}")
    cache['enabled'] = False

@app.on_event("startup")
def load_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    
    print("Loading models and lexicons...")
    # Classical / ML models
    pkg['ridge'] = joblib.load(os.path.join(models_dir, "classical_ridge_model.pkl"))
    pkg['xgb'] = joblib.load(os.path.join(models_dir, "advanced_xgboost_model_local.pkl"))
    rf_path = os.path.join(models_dir, "random_forest_model_local.pkl")
    if os.path.exists(rf_path):
        pkg['rf'] = joblib.load(rf_path)
    else:
        pkg['rf'] = None
    
    pkg['tfidf'] = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    pkg['scaler'] = joblib.load(os.path.join(models_dir, "feature_scaler.pkl"))
    
    pkg['nlp'] = spacy.load("en_core_web_sm")
    if "sentencizer" not in pkg['nlp'].pipe_names:
        pkg['nlp'].add_pipe("sentencizer")
    pkg['nrc'] = load_nrc_lexicon(os.path.join(data_dir, "NRC-Emotion-Lexicon-Senselevel-v0.92.txt"))
    
    # Frozen BERT for ML models
    pkg['bert_tok'] = BertTokenizer.from_pretrained("bert-base-uncased")
    pkg['bert'] = BertModel.from_pretrained("bert-base-uncased")
    pkg['bert'].eval()
    
    # Fine-Tuned BERT
    bert_pt = os.path.join(models_dir, "bert_regressor_best.pt")
    if os.path.exists(bert_pt):
        brt = BertRegressorModel()
        brt.load_state_dict(torch.load(bert_pt, map_location="cpu", weights_only=True))
        brt.eval()
        pkg['bert_ft'] = brt
    else:
        pkg['bert_ft'] = None

    print("API ready to serve requests.")

@app.get("/")
def read_root():
    return {"status": "Pandora API is running successfully on Hugging Face Spaces!"}

@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text
    if not text.strip():
        return {"error": "Empty text"}
        
    start_time = time.time()
    
    text_hash = hashlib.md5(f"{req.model}::{text}".encode('utf-8')).hexdigest()
    
    if cache.get('enabled') and text_hash in cache['bloom']:
        cached_result = cache['redis'].get(text_hash)
        if cached_result:
            res = json.loads(cached_result)
            res['time_ms'] = round((time.time() - start_time) * 1000)
            res['cached'] = True
            return res
    
    # NLP Preprocessing
    doc = pkg['nlp'](text)
    lemmatized = " ".join([t.lemma_ for t in doc if not t.is_stop])
    
    # Ling & Emo Features
    ling_df = compute_linguistic_features([text], pkg['nlp'])
    emo_df = compute_emotional_features(pd.Series([lemmatized]), pkg['nrc'])
    
    # Predict with chosen model
    score = 0.0
    shap_tokens = []
    
    if req.model == "Fine-Tuned BERT" and pkg.get('bert_ft'):
        enc = pkg['bert_tok'](text, return_tensors='pt', truncation=True, max_length=256)
        with torch.no_grad():
            score_t, _ = pkg['bert_ft'](enc["input_ids"], enc["attention_mask"])
            score = float(score_t.item())
        
        # Calculate SHAP tokens
        try:
            _, shap_tokens = explain_text(text, pkg['bert_ft'], pkg['bert_tok'], "cpu", max_display=6)
        except Exception as e:
            print("SHAP Error:", e)
    
    else: # ML Models (Ridge/XGB/RF)
        tfidf_v = pkg['tfidf'].transform([lemmatized])
        tfidf_df = pd.DataFrame(tfidf_v.toarray(), columns=[f"tfidf_{i}" for i in range(2000)])
        combined = pd.concat([ling_df, emo_df, tfidf_df], axis=1)
        X_classical = pkg['scaler'].transform(combined)
        
        if req.model == "Ridge Regression":
            score = float(pkg['ridge'].predict(X_classical[:, :1018])[0])
            
        else: # XGB or RF
            inp = pkg['bert_tok'](text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                emb = pkg['bert'](**inp).last_hidden_state[:, 0, :].numpy()
            X_final = np.hstack((X_classical, emb))
            
            if req.model == "XGBoost":
                if pkg.get('xgb') is None:
                    return {"error": "XGBoost model is not currently available on the server."}
                score = float(pkg['xgb'].predict(X_final)[0])
            elif req.model == "Random Forest":
                if pkg.get('rf') is None:
                    return {"error": "Random Forest model is not currently available on the server."}
                score = float(pkg['rf'].predict(X_final)[0])
                
            # Dummy SHAP for ML models (since we don't have a fast TreeExplainer setup here for a single sample)
            shap_tokens = [
                {"token": "Classical/Frozen Feature 1", "shap_value": 5.2},
                {"token": "Classical/Frozen Feature 2", "shap_value": -3.1}
            ]
            
    # Clamp score
    score = min(99.0, max(0.0, score))
    
    # Calculate implicit Big Five (Radar Chart logic)
    joy = float(emo_df.get('joy', pd.Series([0])).iloc[0])
    trust = float(emo_df.get('trust', pd.Series([0])).iloc[0])
    anger = float(emo_df.get('anger', pd.Series([0])).iloc[0])
    sadness = float(emo_df.get('sadness', pd.Series([0])).iloc[0])
    fear = float(emo_df.get('fear', pd.Series([0])).iloc[0])
    positive = float(emo_df.get('positive', pd.Series([0])).iloc[0])
    negative = float(emo_df.get('negative', pd.Series([0])).iloc[0])
    
    nouns = float(ling_df.get('nouns_ratio', pd.Series([0])).iloc[0])
    verbs = float(ling_df.get('verbs_ratio', pd.Series([0])).iloc[0])
    
    agreeableness = min(99.0, max(0.0, (joy * 3 + trust * 4 - anger * 2) * 10 + 40))
    openness = min(99.0, max(0.0, (positive * 2 + nouns * 50) + 30))
    conscientiousness = min(99.0, max(0.0, 50 - negative * 5 + verbs * 30))
    neuroticism = min(99.0, max(0.0, (sadness * 4 + fear * 3 + negative * 2) * 8 + 15))

    result = {
        "score": round(score, 1),
        "persona": get_persona(score),
        "emotions": {
            "joy": joy,
            "trust": trust,
            "sadness": sadness,
            "fear": fear,
            "anger": anger,
            "positive": positive,
            "negative": negative
        },
        "radar": {
            "Extraversion": round(score, 1),
            "Agreeableness": round(agreeableness, 1),
            "Openness": round(openness, 1),
            "Conscientiousness": round(conscientiousness, 1),
            "Neuroticism": round(neuroticism, 1)
        },
        "shap_tokens": shap_tokens,
        "time_ms": round((time.time() - start_time) * 1000),
        "cached": False
    }

    if cache.get('enabled'):
        cache['bloom'].add(text_hash)
        cache['redis'].setex(text_hash, 86400, json.dumps(result))

    return result

def get_persona(score: float) -> dict:
    if score >= 75:
        return {
            "title": "Social Catalyst",
            "desc": "You are highly outgoing, energetic, and expressive. Your language suggests that you thrive in social situations, naturally draw people in, and openly share your thoughts and emotions with the world."
        }
    elif score >= 45:
        return {
            "title": "Balanced Thinker",
            "desc": "You have a balanced personality. You enjoy interacting with others and can be sociable when needed, but you also value your personal time and need quiet moments to recharge."
        }
    else:
        return {
            "title": "Quiet Observer",
            "desc": "You are thoughtful, deliberate, and introverted. Your language suggests that you prefer observing and analyzing situations carefully rather than being the center of attention. You value deep, meaningful connections over casual interactions."
        }
