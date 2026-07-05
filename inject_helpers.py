"""
Helper script: inject radar chart and compare_all_scores helpers into streamlit_app.py
Run once: python inject_helpers.py
"""
HELPERS = '''

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
    ax.spines["polar"].set_color("rgba(0,229,255,0.2)")
    ax.grid(color="rgba(0,229,255,0.15)", linewidth=0.8)
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

'''

with open("streamlit_app.py", "r", encoding="utf-8") as f:
    content = f.read()

MARKER = "    return colors.get(name, ('#E6F1FF', '#112240'))\n"
if "make_radar_chart" in content:
    print("Helpers already injected.")
elif MARKER in content:
    content = content.replace(MARKER, MARKER + HELPERS, 1)
    with open("streamlit_app.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Helpers injected successfully.")
else:
    print("ERROR: Marker not found in file.")
    # show context around emo_color
    idx = content.find("emo_color")
    print(repr(content[idx:idx+200]))
