import streamlit as st
import joblib
import pandas as pd
import numpy as np
import spacy
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import shap
import io
import time
import warnings
warnings.filterwarnings('ignore')
from PIL import Image

# Import feature extraction helpers
from src.extract_classical_features import compute_linguistic_features, compute_emotional_features, load_nrc_lexicon

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pandora — Personality Intelligence",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# MYTHICAL ICE CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Orbitron:wght@400;700;900&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main Background ── */
.stApp {
    background: linear-gradient(135deg, #020c1b 0%, #0a192f 40%, #0d2347 70%, #061428 100%);
    background-attachment: fixed;
}

/* ── Animated Particle-like Floating Glow ── */
.stApp::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 20% 50%, rgba(0,229,255,0.04) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(100,181,246,0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(0,176,255,0.03) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* ── Hero Header ── */
.pandora-hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
    position: relative;
}
.pandora-title {
    font-family: 'Orbitron', monospace;
    font-size: 4.5rem;
    font-weight: 900;
    letter-spacing: 0.5rem;
    background: linear-gradient(90deg, #64b5f6, #00e5ff, #80d8ff, #00b0ff, #64b5f6);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s ease infinite;
    text-shadow: none;
    line-height: 1.1;
}
.pandora-subtitle {
    font-size: 1.05rem;
    color: #64b5f6;
    letter-spacing: 0.35rem;
    text-transform: uppercase;
    margin-top: 0.4rem;
    opacity: 0.85;
}
.pandora-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #00e5ff, #80d8ff, #00e5ff, transparent);
    margin: 1.5rem auto;
    max-width: 600px;
    border-radius: 4px;
    box-shadow: 0 0 12px #00e5ff88;
}
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(17, 34, 64, 0.8);
    border-radius: 16px;
    padding: 6px;
    border: 1px solid rgba(0,229,255,0.15);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #64b5f6 !important;
    font-weight: 500;
    font-size: 0.88rem;
    letter-spacing: 0.04rem;
    border-radius: 10px;
    padding: 10px 20px;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,229,255,0.2), rgba(100,181,246,0.15)) !important;
    color: #00e5ff !important;
    box-shadow: 0 0 16px rgba(0,229,255,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
    border: 1px solid rgba(0,229,255,0.4) !important;
}

/* ── Glassmorphism Cards ── */
.glass-card {
    background: rgba(17, 34, 64, 0.6);
    border: 1px solid rgba(0, 229, 255, 0.18);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    margin-bottom: 1.5rem;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(0,229,255,0.35);
    box-shadow: 0 8px 40px rgba(0,229,255,0.12), inset 0 1px 0 rgba(255,255,255,0.07);
}

/* ── Section Headers ── */
.ice-header {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #00e5ff;
    letter-spacing: 0.15rem;
    margin-bottom: 1rem;
    text-shadow: 0 0 20px rgba(0,229,255,0.6);
}
.ice-sub {
    color: #80d8ff;
    font-size: 0.95rem;
    font-weight: 400;
    line-height: 1.7;
    opacity: 0.9;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(0,229,255,0.08), rgba(100,181,246,0.05));
    border: 1px solid rgba(0,229,255,0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,229,255,0.1);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00e5ff, transparent);
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00e5ff, #80d8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    color: #64b5f6;
    font-size: 0.78rem;
    letter-spacing: 0.12rem;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* ── Score Gauge ── */
.score-bar-container {
    background: rgba(0,0,0,0.3);
    border-radius: 50px;
    height: 14px;
    border: 1px solid rgba(0,229,255,0.2);
    overflow: hidden;
    margin: 0.5rem 0;
}
.score-bar-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #0d47a1, #1565c0, #00b0ff, #00e5ff, #80d8ff);
    box-shadow: 0 0 12px rgba(0,229,255,0.7);
    transition: width 1s ease;
}

/* ── Report Card ── */
.report-card {
    background: linear-gradient(145deg, rgba(10,25,47,0.95), rgba(17,34,64,0.95));
    border: 1px solid rgba(0,229,255,0.35);
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: 0 0 60px rgba(0,229,255,0.08), 0 20px 60px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
}
.report-card::after {
    content: 'PANDORA';
    position: absolute;
    bottom: -20px; right: -10px;
    font-family: 'Orbitron', monospace;
    font-size: 6rem;
    font-weight: 900;
    color: rgba(0,229,255,0.04);
    pointer-events: none;
    user-select: none;
}
.report-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.3rem;
    color: #00e5ff;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.5rem;
}
.report-score {
    font-family: 'Orbitron', monospace;
    font-size: 5rem;
    font-weight: 900;
    line-height: 1;
    background: linear-gradient(135deg, #00e5ff 0%, #80d8ff 50%, #64b5f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    filter: drop-shadow(0 0 20px rgba(0,229,255,0.5));
}
.report-persona {
    font-size: 1.3rem;
    font-weight: 700;
    color: #80d8ff;
    letter-spacing: 0.08rem;
    margin-top: 0.5rem;
}
.report-desc {
    color: #90caf9;
    font-size: 0.92rem;
    line-height: 1.7;
    margin-top: 0.8rem;
}

/* ── Emotion Tags ── */
.emo-tag {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px;
    border: 1px solid;
    letter-spacing: 0.05rem;
}

/* ── DataFrames ── */
.dataframe { border-radius: 12px !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,180,255,0.15), rgba(0,229,255,0.1));
    color: #00e5ff !important;
    border: 1px solid rgba(0,229,255,0.5) !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.08rem;
    padding: 0.7rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,229,255,0.15);
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,229,255,0.25), rgba(0,180,255,0.2)) !important;
    box-shadow: 0 0 30px rgba(0,229,255,0.35), 0 4px 20px rgba(0,0,0,0.4) !important;
    border-color: rgba(0,229,255,0.8) !important;
    transform: translateY(-1px);
}

/* ── Text Area ── */
.stTextArea textarea {
    background: rgba(6,20,40,0.8) !important;
    border: 1px solid rgba(0,229,255,0.25) !important;
    border-radius: 12px !important;
    color: #e6f1ff !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    transition: border-color 0.3s ease;
}
.stTextArea textarea:focus {
    border-color: rgba(0,229,255,0.6) !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.15) !important;
}
.stTextArea label {
    color: #64b5f6 !important;
    font-weight: 600;
}

/* ── Info / Success / Warning boxes ── */
.stAlert { border-radius: 12px !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00e5ff !important; }

/* ── Feature table ── */
.feat-row { 
    display: flex; 
    justify-content: space-between; 
    padding: 6px 0; 
    border-bottom: 1px solid rgba(0,229,255,0.08); 
}
.feat-name { color: #90caf9; font-size: 0.88rem; }
.feat-val  { color: #00e5ff; font-weight: 600; font-size: 0.88rem; font-family: 'Orbitron', monospace; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(6, 18, 40, 0.95) !important;
    border-right: 1px solid rgba(0,229,255,0.12) !important;
}

/* ── Hide streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer     { visibility: hidden; }
header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
base_dir   = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "models")
data_dir   = os.path.join(base_dir, "data")

# ─────────────────────────────────────────────────────────────────────────────
# CACHED MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_models():
    import torch
    from transformers import BertTokenizer, BertModel

    pkg = {}
    try:
        pkg['xgb']      = joblib.load(os.path.join(models_dir, "advanced_xgboost_model_local.pkl"))
        pkg['tfidf']    = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
        pkg['scaler']   = joblib.load(os.path.join(models_dir, "feature_scaler.pkl"))
        pkg['nrc']      = load_nrc_lexicon(os.path.join(data_dir, "NRC-Emotion-Lexicon-Senselevel-v0.92.txt"))

        nlp = spacy.load("en_core_web_sm")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        pkg['nlp'] = nlp

        pkg['bert_tok'] = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model.eval()
        pkg['bert'] = bert_model

        pkg['train_cols'] = (
            pd.read_csv(os.path.join(data_dir, 'train_features.csv'), nrows=0)
              .drop(columns=['extraversion'])
              .columns
              .tolist()
        )
        pkg['ok'] = True
    except Exception as e:
        pkg['ok'] = False
        pkg['error'] = str(e)
    return pkg


def predict_and_explain(text: str, pkg: dict):
    """Full pipeline: text → extraversion score + SHAP + emotion features."""
    import torch

    nlp     = pkg['nlp']
    doc     = nlp(text)
    lemmatized = " ".join([t.lemma_ for t in doc if not t.is_stop])

    # Classical features
    ling_df = compute_linguistic_features([text], nlp)
    emo_df  = compute_emotional_features(pd.Series([lemmatized]), pkg['nrc'])
    tfidf_v = pkg['tfidf'].transform([lemmatized])
    tfidf_df = pd.DataFrame(tfidf_v.toarray(), columns=[f"tfidf_{i}" for i in range(2000)])

    combined = pd.concat([ling_df, emo_df, tfidf_df], axis=1)

    # Reorder to training column order
    df_ord = pd.DataFrame(columns=pkg['train_cols'])
    for col in pkg['train_cols']:
        series = combined.get(col, 0)
        df_ord.loc[0, col] = series.iloc[0] if isinstance(series, pd.Series) else series

    X_classical = pkg['scaler'].transform(df_ord.astype(float))

    # BERT embedding
    tokenizer  = pkg['bert_tok']
    bert_model = pkg['bert']
    inputs = tokenizer(text, return_tensors='pt', padding=True,
                       truncation=True, max_length=512)
    with torch.no_grad():
        outputs  = bert_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].numpy()

    X_final = np.hstack((X_classical, embedding))

    # Predict
    score = pkg['xgb'].predict(X_final)[0]

    # SHAP waterfall
    explainer   = shap.Explainer(pkg['xgb'])
    shap_values = explainer(X_final)
    feat_names  = pkg['train_cols'] + [f"bert_{i}" for i in range(768)]
    shap_values.feature_names = feat_names

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0A192F')
    shap.plots.waterfall(shap_values[0], max_display=12, show=False)
    plt.gcf().patch.set_facecolor('#0A192F')
    current_ax = plt.gca()
    current_ax.set_facecolor('#112240')
    current_ax.tick_params(colors='#80d8ff')
    current_ax.xaxis.label.set_color('#80d8ff')
    current_ax.yaxis.label.set_color('#80d8ff')
    for spine in current_ax.spines.values():
        spine.set_edgecolor('#00e5ff4d')
    plt.title(f"SHAP Explanation — Predicted Score: {score:.1f}",
              color='#00e5ff', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150,
                bbox_inches='tight', facecolor='#0A192F')
    buf.seek(0)
    plt.close('all')

    # Extract emotion breakdown
    emo_cols = [c for c in emo_df.columns if c in
                ['anger','anticip','disgust','fear','joy',
                 'sadness','surprise','trust','negative','positive']]
    emo_vals = {c: float(emo_df[c].iloc[0]) for c in emo_cols}

    # Linguistic summary
    ling_info = {
        'word_count':   int(ling_df.get('word_count', pd.Series([0])).iloc[0]),
        'sent_count':   int(ling_df.get('sentence_count', pd.Series([0])).iloc[0]),
        'prons_ratio':  round(float(ling_df.get('prons_ratio', pd.Series([0.0])).iloc[0]) * 100, 1),
        'nouns_ratio':  round(float(ling_df.get('nouns_ratio', pd.Series([0.0])).iloc[0]) * 100, 1),
        'verbs_ratio':  round(float(ling_df.get('verbs_ratio', pd.Series([0.0])).iloc[0]) * 100, 1),
    }

    return score, buf, emo_vals, ling_info


def persona_label(score: float):
    """Map score to persona. Target is continuous 0–99."""
    # The dataset mean is 35, std ~30. Thresholds calibrated to dataset quartiles:
    # Q1=7, Median=24, Q3=60
    if score >= 72:
        return ("Highly Extraverted", "#FF6B6B", "#FFD93D",
                "You radiate social energy! You thrive in groups, love conversation, and feel energized by the world around you. Your text signals an outgoing, expressive personality that naturally draws people in.")
    elif score >= 55:
        return ("Moderately Extraverted", "#FFB347", "#FFD93D",
                "You lean towards extraversion. Social interactions energize you and you enjoy engaging with the world, though you also appreciate quieter moments of reflection.")
    elif score >= 35:
        return ("Ambivert", "#64b5f6", "#80d8ff",
                "You sit beautifully in the middle — comfortable alone or in a crowd. Your text reflects versatility: adaptable, balanced, and deeply empathetic.")
    elif score >= 15:
        return ("Moderately Introverted", "#b39ddb", "#ce93d8",
                "You tend to recharge through solitude and deep focus. Your text suggests a preference for meaningful, one-on-one conversations over large social gatherings.")
    else:
        return ("Highly Introverted", "#80cbc4", "#4db6ac",
                "A deep, reflective thinker! You gain your energy from within, cherish your personal space, and prefer depth over breadth in all interactions.")


def emo_color(name: str):
    colors = {
        'joy': ('#FFD700', '#3d3000'), 'trust': ('#00E5FF', '#003040'),
        'anticipation': ('#FF9800', '#3d2200'), 'surprise': ('#E040FB', '#2a0040'),
        'fear': ('#78909C', '#1c2a30'), 'anger': ('#FF5252', '#3d0000'),
        'disgust': ('#66BB6A', '#0a2d0b'), 'sadness': ('#5C6BC0', '#0d1340'),
        'anticip': ('#FF9800', '#3d2200'),
        'positive': ('#69F0AE', '#002918'), 'negative': ('#FF6E6E', '#330000')
    }
    return colors.get(name, ('#E6F1FF', '#112240'))


def make_radar_chart(score, emo_vals, ling_info):
    """Generate a Big-Five-style personality radar chart."""
    joy      = emo_vals.get("joy", 0)
    trust    = emo_vals.get("trust", 0)
    sadness  = emo_vals.get("sadness", 0)
    fear     = emo_vals.get("fear", 0)
    anger    = emo_vals.get("anger", 0)
    positive = emo_vals.get("positive", 0)
    negative = emo_vals.get("negative", 0)
    nouns    = ling_info.get("nouns_ratio", 0)
    verbs    = ling_info.get("verbs_ratio", 0)
    agreeableness     = min(99, max(0, (joy * 3 + trust * 4 - anger * 2) * 10 + 40))
    openness          = min(99, max(0, (positive * 2 + nouns * 50) + 30))
    conscientiousness = min(99, max(0, 50 - negative * 5 + verbs * 30))
    neuroticism       = min(99, max(0, (sadness * 4 + fear * 3 + negative * 2) * 8 + 15))
    labels = ["Extraversion", "Agreeableness", "Openness", "Conscientiousness", "Neuroticism"]
    values = [score, agreeableness, openness, conscientiousness, neuroticism]
    vals_n = [v / 99.0 for v in values] + [score / 99.0]
    import numpy as np, matplotlib.pyplot as plt, io
    angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0A192F")
    ax.set_facecolor("#112240")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25", "50", "75", "99"], color="#4a7fa5", fontsize=7)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="#80d8ff", fontsize=9, fontweight="600")
    ax.spines["polar"].set_color("#00e5ff33")
    ax.grid(color="#00e5ff26", linewidth=0.8)
    ax.fill(angles, vals_n, alpha=0.25, color="#00e5ff")
    ax.plot(angles, vals_n, linewidth=2, color="#00e5ff")
    ax.scatter(angles[0], vals_n[0], s=80, color="#FFD93D", zorder=5)
    ax.set_title("Personality Profile (Estimated)", color="#00e5ff", fontsize=10, fontweight="bold", pad=18)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#0A192F")
    buf.seek(0)
    plt.close("all")
    return buf


def compare_all_scores(text, pkg):
    """Run Ridge, XGBoost, RF, and optionally fine-tuned BERT, return scores list."""
    import torch, os, joblib, numpy as np, pandas as pd
    from src.extract_classical_features import compute_linguistic_features, compute_emotional_features
    nlp        = pkg["nlp"]
    doc        = nlp(text)
    lemmatized = " ".join([t.lemma_ for t in doc if not t.is_stop])
    ling_df    = compute_linguistic_features([text], nlp)
    emo_df     = compute_emotional_features(pd.Series([lemmatized]), pkg["nrc"])
    tfidf_v    = pkg["tfidf"].transform([lemmatized])
    tfidf_df   = pd.DataFrame(tfidf_v.toarray(), columns=[f"tfidf_{i}" for i in range(2000)])
    combined   = pd.concat([ling_df, emo_df, tfidf_df], axis=1)
    df_ord     = pd.DataFrame(columns=pkg["train_cols"])
    for col in pkg["train_cols"]:
        series = combined.get(col, 0)
        df_ord.loc[0, col] = series.iloc[0] if isinstance(series, pd.Series) else series
    X_classical = pkg["scaler"].transform(df_ord.astype(float))
    tok = pkg["bert_tok"]; bm = pkg["bert"]
    inp = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        emb = bm(**inp).last_hidden_state[:, 0, :].numpy()
    X_final = np.hstack((X_classical, emb))
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    results = []
    results.append({"model": "Ridge (Baseline)", "score": round(float(pkg["ridge"].predict(X_classical)[0]), 2), "color": "#546e7a"})
    results.append({"model": "XGBoost",          "score": round(float(pkg["xgb"].predict(X_final)[0]), 2),      "color": "#00b0ff"})
    rf_path = os.path.join(models_dir, "random_forest_model_local.pkl")
    if os.path.exists(rf_path):
        rf_score = joblib.load(rf_path).predict(X_final)[0]
        results.append({"model": "Random Forest", "score": round(float(rf_score), 2), "color": "#00e5ff"})
    bert_pt = os.path.join(models_dir, "bert_regressor_best.pt")
    if os.path.exists(bert_pt):
        try:
            from src.models.bert_regressor import BertRegressorModel
            brt = BertRegressorModel()
            brt.load_state_dict(torch.load(bert_pt, map_location="cpu"))
            brt.eval()
            enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                brt_score, _ = brt(enc["input_ids"], enc["attention_mask"])
            results.append({"model": "Fine-tuned BERT", "score": round(float(brt_score.item()), 2), "color": "#ce93d8"})
        except Exception:
            pass
    return results



# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pandora-hero">
    <div class="pandora-title">PANDORA</div>
    <div class="pandora-subtitle">✦ Personality Intelligence System ✦</div>
    <div class="pandora-divider"></div>
    <div style="color:#4a7fa5; font-size:0.82rem; letter-spacing:0.1rem;">
        FEATURE FUSION · BERT · XGBOOST · SHAP EXPLAINABILITY
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "❄️  Overview",
    "🤖  BERT Embedding",
    "📊  Data Analysis",
    "📈  Model Results",
    "🔮  Live Analysis"
])


# ═══════════════════════════════ TAB 1 – OVERVIEW ════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.05, 1], gap="large")

    with col_left:
        st.markdown("""
        <div class="glass-card">
            <div class="ice-header">🎯 Mission</div>
            <div class="ice-sub">
            Pandora is an advanced computational psychology system that predicts the
            <strong style="color:#00e5ff;">Extraversion</strong> personality dimension directly
            from short text. It fuses classical NLP signals with deep contextual understanding
            from Google's BERT Transformer to achieve superior accuracy over linear baselines.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div class="ice-header">🧬 Feature Fusion Pipeline</div>
            <table style="width:100%; border-collapse:collapse;">
                <tr>
                    <td style="padding:10px 0; border-bottom:1px solid rgba(0,229,255,0.1); color:#00e5ff; font-weight:600; width:40%;">
                        🔤 TF-IDF Bigrams
                    </td>
                    <td style="padding:10px 0; border-bottom:1px solid rgba(0,229,255,0.1); color:#90caf9; font-size:0.88rem;">
                        2,000-dim sparse lexical matrix
                    </td>
                </tr>
                <tr>
                    <td style="padding:10px 0; border-bottom:1px solid rgba(0,229,255,0.1); color:#00e5ff; font-weight:600;">
                        💠 NRC Emotion Lexicon
                    </td>
                    <td style="padding:10px 0; border-bottom:1px solid rgba(0,229,255,0.1); color:#90caf9; font-size:0.88rem;">
                        10 discrete emotional dimensions
                    </td>
                </tr>
                <tr>
                    <td style="padding:10px 0; border-bottom:1px solid rgba(0,229,255,0.1); color:#00e5ff; font-weight:600;">
                        📐 Linguistic POS
                    </td>
                    <td style="padding:10px 0; border-bottom:1px solid rgba(0,229,255,0.1); color:#90caf9; font-size:0.88rem;">
                        Noun/Verb/Adj/Pronoun ratios
                    </td>
                </tr>
                <tr>
                    <td style="padding:10px 0; color:#80d8ff; font-weight:700; font-size:1rem;">
                        🤖 BERT Embeddings
                    </td>
                    <td style="padding:10px 0; color:#80d8ff; font-size:0.88rem;">
                        768-dim deep semantic context
                    </td>
                </tr>
            </table>
            <div style="margin-top:1.2rem; padding:0.8rem 1rem; background:rgba(0,229,255,0.06); border-radius:10px; border-left:3px solid #00e5ff;">
                <span style="color:#64b5f6; font-size:0.85rem;">
                📐 Total Feature Space: <strong style="color:#00e5ff; font-family:'Orbitron';">2,776 dimensions</strong> per input.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="glass-card">
            <div class="ice-header">🏆 Key Results</div>
        """, unsafe_allow_html=True)

        results = [
            ("Ridge Regression", "Baseline", "29.18", "0.103", "#546e7a"),
            ("XGBoost", "Feature Fusion", "28.29", "0.158", "#00b0ff"),
            ("Random Forest ★", "Feature Fusion", "28.22", "0.162", "#00e5ff"),
        ]
        for name, feat, rmse, r2, col in results:
            border = "2px solid #00e5ff" if "★" in name else "1px solid rgba(0,229,255,0.15)"
            glow = "box-shadow:0 0 20px rgba(0,229,255,0.2);" if "★" in name else ""
            st.markdown(f"""
            <div style="background:rgba(17,34,64,0.7); border:{border}; {glow}
                        border-radius:14px; padding:1rem 1.2rem; margin-bottom:0.8rem; position:relative;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="color:{col}; font-weight:700; font-size:0.95rem;">{name}</span>
                        <div style="color:#4a7fa5; font-size:0.75rem; margin-top:2px;">{feat}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="color:#ff8a65; font-size:0.82rem;">RMSE <strong style="color:#ffccbc;">{rmse}</strong></div>
                        <div style="color:#64b5f6; font-size:0.82rem;">R² <strong style="color:{col};">{r2}</strong></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div class="ice-header">👩‍💻 System Info</div>
            <div class="feat-row"><span class="feat-name">Dataset</span><span class="feat-val">Pandora</span></div>
            <div class="feat-row"><span class="feat-name">Training Samples</span><span class="feat-val">12,837</span></div>
            <div class="feat-row"><span class="feat-name">Test Samples</span><span class="feat-val">3,210</span></div>
            <div class="feat-row"><span class="feat-name">BERT Model</span><span class="feat-val">bert-base-uncased</span></div>
            <div class="feat-row"><span class="feat-name">CV Folds</span><span class="feat-val">5</span></div>
            <div class="feat-row"><span class="feat-name">Best Model</span><span class="feat-val">Random Forest</span></div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════ TAB 2 – BERT EMBEDDING ══════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ice-header" style="text-align:center; font-size:1.6rem; margin-bottom:0.5rem;">🤖 BERT Semantic Embedding</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:#4a7fa5; font-size:0.88rem; margin-bottom:1.5rem; letter-spacing:0.06rem;">How Google\'s Transformer converts your text into a 768-dimensional personality fingerprint</div>', unsafe_allow_html=True)

    # ── Pipeline steps ──
    steps = [
        ("01", "Raw Text Input", "Your raw sentence enters the pipeline unchanged.",
         "e.g. \"I love reading alone at home\"", "#00e5ff"),
        ("02", "Tokenisation", "BERT\'s WordPiece tokenizer splits the text into sub-word tokens and prepends the special [CLS] classification token.",
         "[CLS] i love reading alone at home [SEP]", "#80d8ff"),
        ("03", "12-Layer Transformer", "Each of the 12 Transformer layers applies Multi-Head Self-Attention — mathematically weighing every word against every other word to build contextual understanding.",
         "Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V", "#64b5f6"),
        ("04", "[CLS] Pooling", "The hidden state of the [CLS] token after all 12 layers is extracted. This single vector encodes the entire sentence meaning.",
         "shape: (1, 768) — one float per dimension", "#b39ddb"),
        ("05", "768-D Output Vector", "The final embedding is a row of 768 decimal numbers. This is the \"mathematical thought\" of BERT — capturing sarcasm, tone, and personality cues invisible to keyword matching.",
         "[0.312, -0.841, 0.075, 1.203, ...] × 768 values", "#ce93d8"),
        ("06", "Feature Fusion", "The 768 BERT dims are horizontally stacked (np.hstack) with the 2,008 classical features → a 2,776-dim vector that feeds into XGBoost.",
         "X_final = [X_classical (2008) | X_bert (768)] = 2,776 dims", "#FFD93D"),
    ]

    for step_num, title, desc, code, color in steps:
        st.markdown(f"""
        <div style="display:flex; gap:1.2rem; margin-bottom:1rem; align-items:flex-start;">
            <div style="min-width:44px; height:44px; border-radius:50%; border:2px solid {color};
                        display:flex; align-items:center; justify-content:center;
                        font-family:\'Orbitron\',monospace; font-size:0.75rem; font-weight:700;
                        color:{color}; background:rgba(0,0,0,0.3); flex-shrink:0;">{step_num}</div>
            <div style="flex:1; background:rgba(17,34,64,0.5); border:1px solid rgba(0,229,255,0.1);
                        border-left:3px solid {color}; border-radius:12px; padding:1rem 1.2rem;">
                <div style="color:{color}; font-weight:700; font-size:0.95rem; margin-bottom:0.3rem;">{title}</div>
                <div style="color:#90caf9; font-size:0.87rem; line-height:1.6; margin-bottom:0.5rem;">{desc}</div>
                <code style="background:rgba(0,0,0,0.4); color:#a8dadc; font-size:0.8rem; padding:3px 8px;
                             border-radius:6px; border:1px solid rgba(0,229,255,0.15);">{code}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Live embedding heatmap for a typed sentence ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="ice-header" style="font-size:1rem;">⚡ Live Embedding Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="ice-sub" style="font-size:0.85rem; margin-bottom:1rem;">Type a sentence and see the first 64 dimensions of its BERT embedding visualised as a heatmap.</div>', unsafe_allow_html=True)

    emb_text = st.text_input("Enter a sentence:", value="I love reading books alone at home", key="bert_vis_input")
    if st.button("🔬 Generate Embedding", key="emb_btn"):
        with st.spinner("Running BERT..."):
            try:
                import torch
                m = load_all_models()
                if m.get('ok'):
                    tok = m['bert_tok']
                    bm  = m['bert']
                    inp = tok(emb_text, return_tensors='pt', truncation=True, max_length=512)
                    with torch.no_grad():
                        out = bm(**inp)
                        vec = out.last_hidden_state[:, 0, :].numpy()[0]  # (768,)

                    # Plot first 128 dims as heatmap grid (16×8)
                    grid = vec[:128].reshape(8, 16)
                    fig, ax = plt.subplots(figsize=(12, 4))
                    fig.patch.set_facecolor('#0A192F')
                    ax.set_facecolor('#0A192F')
                    im = ax.imshow(grid, cmap='coolwarm', aspect='auto', vmin=-2, vmax=2)
                    ax.set_title(f'First 128 of 768 BERT Dimensions  —  "{emb_text}"',
                                 color='#00e5ff', fontsize=11, fontweight='bold', pad=10)
                    ax.set_xlabel('Dimension Index (0–15 per row)', color='#80d8ff', fontsize=9)
                    ax.set_ylabel('Row group (×16)', color='#80d8ff', fontsize=9)
                    ax.tick_params(colors='#80d8ff', labelsize=8)
                    for spine in ax.spines.values():
                        spine.set_edgecolor('rgba(0,229,255,0.2)')
                    cbar = fig.colorbar(im, ax=ax, fraction=0.03)
                    cbar.ax.yaxis.set_tick_params(color='#80d8ff')
                    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#80d8ff')
                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0A192F')
                    buf.seek(0)
                    plt.close('all')
                    st.image(buf, use_container_width=True, caption="Red = high activation | Blue = low activation | Each cell = one latent dimension")

                    # Stats
                    c_a, c_b, c_c, c_d = st.columns(4)
                    c_a.metric("Dimensions", "768")
                    c_b.metric("Max Activation", f"{vec.max():.3f}")
                    c_c.metric("Min Activation", f"{vec.min():.3f}")
                    c_d.metric("Mean Activation", f"{vec.mean():.3f}")
            except Exception as e:
                st.error(f"Embedding error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════ TAB 3 – EDA ════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ice-header" style="text-align:center; font-size:1.6rem; margin-bottom:1.5rem;">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        try:
            img = Image.open(os.path.join(models_dir, "eda_target_dist.png"))
            st.image(img, caption="Distribution of Extraversion Scores", use_container_width=True)
            st.markdown("""
            <div style="background:rgba(0,229,255,0.06); border-left:3px solid #00e5ff; border-radius:8px;
                        padding:0.7rem 1rem; margin-top:0.8rem; color:#90caf9; font-size:0.85rem; line-height:1.6;">
            <strong style="color:#00e5ff;">Key Finding:</strong> Extraversion scores are
            <strong style="color:#80d8ff;">right-skewed continuous values (0–99)</strong> with a 
            mean of <strong style="color:#00e5ff;">~35</strong> and median of ~24 — confirming most 
            users in the Pandora dataset lean introvert, with extreme extraverts being rarer outliers.
            </div>
            """, unsafe_allow_html=True)
        except:
            st.warning("Target distribution image not found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        try:
            img = Image.open(os.path.join(models_dir, "eda_emo_corr.png"))
            st.image(img, caption="Emotion-Extraversion Correlation Matrix", use_container_width=True)
            st.markdown("""
            <div style="background:rgba(0,229,255,0.06); border-left:3px solid #00e5ff; border-radius:8px;
                        padding:0.7rem 1rem; margin-top:0.8rem; color:#90caf9; font-size:0.85rem; line-height:1.6;">
            <strong style="color:#00e5ff;">Key Finding:</strong> Joy & Anticipation show mild 
            positive correlation with Extraversion. Sadness & Fear show mild negative correlation, 
            aligning with psychological literature on personality and affect.
            </div>
            """, unsafe_allow_html=True)
        except:
            st.warning("Correlation matrix not found.")
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════ TAB 4 – RESULTS ════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ice-header" style="text-align:center; font-size:1.6rem; margin-bottom:1.5rem;">📈 Model Performance & Interpretability</div>', unsafe_allow_html=True)

    # ─ Metric cards ─
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    # Try to load from CSV for live accuracy
    _comp_csv = os.path.join(models_dir, "full_model_comparison.csv")
    if os.path.exists(_comp_csv):
        _cdf = pd.read_csv(_comp_csv)
        _best = _cdf.loc[_cdf["RMSE"] == _cdf["RMSE"].min()].iloc[0]
        _best_rmse = str(_best["RMSE"])
        _best_r2   = str(_best["R²"])
        _best_name = _best["model"]
    else:
        _best_rmse = "28.22"; _best_r2 = "0.162"; _best_name = "Random Forest"

    _metrics_top = [
        ("Best RMSE",   _best_rmse, _best_name),
        ("Best R²",    _best_r2,   _best_name),
        ("Models Tested", "4",     "Ridge · XGB · RF · BERT"),
        ("Feature Dims",  "2,776",  "Total Vector Space"),
    ]
    for col_obj, (label, val, sub) in zip([c1,c2,c3,c4], _metrics_top):
        with col_obj:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div style="color:#4a7fa5; font-size:0.72rem; margin-top:0.3rem;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─ Full 4-model table ─
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="ice-header" style="font-size:1rem;">🧠 All Models Comparison</div>', unsafe_allow_html=True)
    if os.path.exists(_comp_csv):
        st.dataframe(pd.read_csv(_comp_csv), use_container_width=True, hide_index=True)
    else:
        _static = pd.DataFrame([
            {"Model": "Ridge Regression (Baseline)", "Features": "Classical Only", "RMSE": 29.18, "MAE": "—", "R²": 0.103},
            {"Model": "XGBoost",                    "Features": "Fusion (BERT+Classical)", "RMSE": 28.29, "MAE": "—", "R²": 0.158},
            {"Model": "Random Forest",              "Features": "Fusion (BERT+Classical)", "RMSE": 28.22, "MAE": "—", "R²": 0.162},
            {"Model": "Fine-tuned BERT ★",         "Features": "End-to-End BERT",       "RMSE": "(Colab)", "MAE": "(Colab)", "R²": "(Colab)"},
        ])
        st.dataframe(_static, use_container_width=True, hide_index=True)
        st.info("Fine-tuned BERT results will appear here after you complete Colab training and place `bert_test_metrics.json` in the models/ folder.")
    st.markdown('</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")
    with col_l:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        try:
            img = Image.open(os.path.join(models_dir, "full_model_comparison.png"))
            st.image(img, caption="All Models Performance Comparison (RMSE, MAE, R²)", use_container_width=True)
        except:
            try:
                img = Image.open(os.path.join(models_dir, "model_comparison.png"))
                st.image(img, caption="Model Performance Comparison (RMSE & R²)", use_container_width=True)
            except:
                st.warning("Model comparison chart not found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        # Training curve (from Colab)
        _curve_path = os.path.join(models_dir, "bert_training_curve.png")
        if os.path.exists(_curve_path):
            img = Image.open(_curve_path)
            st.image(img, caption="Fine-Tuned BERT Training Curve", use_container_width=True)
        else:
            try:
                img = Image.open(os.path.join(models_dir, "actual_vs_predicted.png"))
                st.image(img, caption="Actual vs Predicted Extraversion Scores", use_container_width=True)
            except:
                st.warning("Training curve / prediction plot not found.")
            st.markdown("""
            <div style="background:rgba(0,229,255,0.06); border-left:3px solid #FFD93D; border-radius:8px;
                        padding:0.7rem 1rem; margin-top:0.8rem; color:#90caf9; font-size:0.85rem;">
            <strong style="color:#FFD93D;">⏳ Pending:</strong> Fine-tuned BERT training curve will
            appear here after running <code>src/models/train_bert_regressor_colab.py</code> on Colab
            and placing <code>bert_training_curve.png</code> in <code>models/</code>.
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col_l2, col_r2 = st.columns([1, 1], gap="large")
    with col_l2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        try:
            img = Image.open(os.path.join(models_dir, "xgb_shap_summary.png"))
            st.image(img, caption="Global SHAP Feature Importance (Beeswarm)", use_container_width=True)
        except:
            st.warning("SHAP summary not found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        try:
            img = Image.open(os.path.join(models_dir, "xgb_shap_bar.png"))
            st.image(img, caption="SHAP Mean Absolute Importance (Bar)", use_container_width=True)
        except:
            st.warning("SHAP bar chart not found.")
        st.markdown("""
        <div style="background:rgba(0,229,255,0.06); border-left:3px solid #00e5ff;
                    border-radius:8px; padding:0.8rem 1rem; margin-top:0.8rem;
                    color:#90caf9; font-size:0.85rem; line-height:1.6;">
        <strong style="color:#00e5ff;">SHAP Finding:</strong> BERT semantic embeddings
        dominated 12 of the Top 15 most influential features — confirming that deep
        contextual meaning far outweighs explicit word counting for personality prediction.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════ TAB 5 – LIVE DEMO ════════════════════════════
with tab5:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.5rem;">
        <div class="ice-header" style="font-size:1.6rem;">🔮 Live Personality Analysis</div>
        <div style="color:#4a7fa5; font-size:0.88rem; margin-top:0.3rem; letter-spacing:0.08rem;">
            Multi-Model Analysis · SHAP Explainability · Personality Radar Chart
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load status
    with st.spinner("❄️ Initialising Pandora Intelligence Engine..."):
        models = load_all_models()

    if not models.get('ok'):
        st.error(f"⚠️ Model loading failed: {models.get('error', 'Unknown error')}")
        st.info("Make sure `advanced_xgboost_model_local.pkl`, `tfidf_vectorizer.pkl`, `feature_scaler.pkl`, and BERT weights are all available.")
        st.stop()

    # ─ Check for fine-tuned BERT ─
    _bert_pt = os.path.join(models_dir, "bert_regressor_best.pt")
    _bert_ready = os.path.exists(_bert_pt)

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; background:rgba(0,229,255,0.06);
                border:1px solid rgba(0,229,255,0.2); border-radius:10px; padding:0.6rem 1.2rem;
                margin-bottom:1.5rem;">
        <span style="color:#00e5ff; font-size:1rem;">✅</span>
        <span style="color:#64b5f6; font-size:0.88rem;">
            Pandora Intelligence Engine ready — XGBoost + BERT loaded.
            {'Fine-tuned BERT ★ also available!' if {_bert_ready} else 'Fine-tuned BERT: ⏳ Awaiting Colab training.'}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ─ Input + Model Selector ─
    col_inp, col_opt = st.columns([1.5, 1], gap="large")

    with col_inp:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        text_input = st.text_area(
            "✍️ Enter your text below",
            height=160,
            placeholder="Describe your thoughts, a recent experience, or anything on your mind...",
            key="main_input"
        )
        st.caption("💡 Longer texts (3–5 sentences) yield more accurate results.")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            analyze_btn = st.button("🔮 Analyze Personality", use_container_width=True)
        with col_b2:
            compare_btn = st.button("📊 Compare All Models", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_opt:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="ice-header" style="font-size:1rem;">⚙️ Analysis Options</div>', unsafe_allow_html=True)

        _model_opts = ["XGBoost (Feature Fusion)", "Random Forest (Feature Fusion)", "Ridge Regression (Baseline)"]
        if _bert_ready:
            _model_opts.insert(0, "🌟 Fine-tuned BERT (Best)")
        model_choice = st.selectbox("🤖 Select Model", _model_opts)

        show_radar   = st.checkbox("🕸️ Show Personality Radar Chart", value=True)
        show_compare = st.checkbox("📊 Show All-Model Scores Side-by-Side", value=False)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card" style="margin-top:0;">
            <div style="color:#00e5ff; font-size:0.78rem; letter-spacing:0.1rem; font-weight:700; margin-bottom:0.6rem;">💡 TIPS</div>
            <div style="color:#90caf9; font-size:0.85rem; line-height:1.7;">
                ❄ Write in first-person for best results<br>
                ❄ Include social preferences or activities<br>
                ❄ 3–5 sentences gives BERT enough context<br>
                ❄ Try the same text on multiple models!
            </div>
        </div>
        """, unsafe_allow_html=True)
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("❄️ Pandora is analyzing your personality signature..."):
                try:
                    score, shap_buf, emo_vals, ling_info = predict_and_explain(text_input, models)
                except Exception as e:
                    st.error(f"Analysis error: {e}")
                    st.stop()

            persona, c1_, c2_, desc = persona_label(score)
            pct = min(max(score / 99.0, 0), 1.0)
            bar_pct = int(pct * 100)

            st.markdown("<br>", unsafe_allow_html=True)

            # ─── REPORT CARD ───
            st.markdown('<div class="report-card">', unsafe_allow_html=True)

            rc_left, rc_right = st.columns([1.1, 1], gap="large")

            with rc_left:
                st.markdown(f"""
                <div class="report-header">PANDORA PERSONALITY REPORT</div>
                <div class="report-score">{score:.1f}</div>
                <div style="color:#4a7fa5; font-size:0.75rem; letter-spacing:0.1rem; margin-top:0.2rem;">
                    EXTRAVERSION SCORE / 100
                </div>
                <div class="report-persona" style="
                    background: linear-gradient(90deg, {c1_}, {c2_});
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; margin-top:0.7rem;">
                    {persona}
                </div>
                <div class="report-desc">{desc}</div>
                """, unsafe_allow_html=True)

                # Score bar
                st.markdown(f"""
                <div style="margin-top:1.2rem;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="color:#4a7fa5; font-size:0.75rem; letter-spacing:0.08rem;">INTROVERTED</span>
                        <span style="color:#4a7fa5; font-size:0.75rem; letter-spacing:0.08rem;">EXTRAVERTED</span>
                    </div>
                    <div class="score-bar-container">
                        <div class="score-bar-fill" style="width:{bar_pct}%;"></div>
                    </div>
                    <div style="text-align:center; color:#64b5f6; font-size:0.78rem; margin-top:4px;">
                        {bar_pct}th percentile
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with rc_right:
                # Emotional breakdown
                st.markdown("""
                <div style="margin-bottom:0.8rem;">
                    <div style="color:#00e5ff; font-size:0.78rem; letter-spacing:0.12rem;
                                font-family:'Orbitron', monospace; text-transform:uppercase;
                                font-weight:700; margin-bottom:0.6rem;">
                        Emotional Signature
                    </div>
                """, unsafe_allow_html=True)

                emo_html = ""
                for name, val in emo_vals.items():
                    if val > 0:
                        fg, bg = emo_color(name)
                        emo_html += (
                            f'<span class="emo-tag" style="color:{fg}; '
                            f'background:{bg}; border-color:{fg}40;">'
                            f'{name.capitalize()} {int(val)}</span>'
                        )
                if not emo_html:
                    emo_html = '<span style="color:#4a7fa5; font-size:0.85rem;">No strong emotional signals detected.</span>'
                st.markdown(emo_html + "</div>", unsafe_allow_html=True)

                # Linguistic breakdown
                st.markdown("""
                <div style="margin-top:1.2rem; margin-bottom:0.5rem;">
                    <div style="color:#00e5ff; font-size:0.78rem; letter-spacing:0.12rem;
                                font-family:'Orbitron', monospace; text-transform:uppercase;
                                font-weight:700; margin-bottom:0.6rem;">
                        Linguistic Profile
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="feat-row"><span class="feat-name">Word Count</span><span class="feat-val">{ling_info['word_count']}</span></div>
                <div class="feat-row"><span class="feat-name">Sentences</span><span class="feat-val">{ling_info['sent_count']}</span></div>
                <div class="feat-row"><span class="feat-name">Pronoun Ratio</span><span class="feat-val">{ling_info['prons_ratio']}%</span></div>
                <div class="feat-row"><span class="feat-name">Noun Ratio</span><span class="feat-val">{ling_info['nouns_ratio']}%</span></div>
                <div class="feat-row"><span class="feat-name">Verb Ratio</span><span class="feat-val">{ling_info['verbs_ratio']}%</span></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)  # /report-card

            # ─── SHAP WATERFALL ───
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="glass-card">
                <div class="ice-header" style="font-size:1rem; margin-bottom:0.5rem;">
                    ⚡ SHAP Explanation — Feature Contribution Waterfall
                </div>
                <div class="ice-sub" style="font-size:0.83rem; margin-bottom:1rem;">
                    Shows exactly which features pushed the prediction <strong style="color:#69F0AE;">UP ↑</strong> 
                    (extraverted signals) or <strong style="color:#FF6E6E;">DOWN ↓</strong> (introverted signals) 
                    from the baseline mean score.
                </div>
            """, unsafe_allow_html=True)
            st.image(shap_buf, use_container_width=True, caption="Top 12 SHAP Feature Contributions")
            st.markdown('</div>', unsafe_allow_html=True)

            st.balloons()
