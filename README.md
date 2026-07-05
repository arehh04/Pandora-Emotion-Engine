# Predicting Extraversion from Text: A Feature Fusion Approach 🧠
**Final Year Project (FYP) by Nasyrah**

This repository contains the end-to-end machine learning pipeline for predicting Extraversion personality scores from raw text. The project utilizes a "Feature Fusion" approach, combining Classical NLP heuristics (linguistics & emotions) with Deep Learning semantics (BERT).

---

## 🚀 Project Architecture

### 1. Feature Extraction Pipeline
The pipeline analyzes raw text by mathematically fusing two distinct branches of natural language processing:
- **Classical Branch:**
  - **Linguistic Module:** Analyzes grammatical structures (Noun, Verb, Adjective ratios) using `SpaCy`.
  - **Emotional Module:** Extracts 10 discrete emotional frequencies (e.g., Joy, Anger) via the **NRC Emotion Lexicon**.
  - **N-Grams:** Generates `TF-IDF` vectors for the top 2,000 word bigrams.
- **Deep Learning Branch:**
  - **Semantic Module:** Extracts a 768-dimensional contextual embedding using Google's pre-trained `bert-base-uncased` transformer model.

### 2. Machine Learning Modeling (Regression)
The fused feature matrix is normalized via `StandardScaler` and fed into continuous regression algorithms:
- **Baseline Model:** Ridge Regression (Linear)
- **Advanced Models:** XGBoost & Random Forest (Non-Linear Ensemble Trees)

*Hyperparameter tuning was conducted via `RandomizedSearchCV` (3-Fold CV) on Google Colab GPUs.*

### 3. Explainability & EDA
- **SHAP (SHapley Additive exPlanations):** Used to unpack the "black box" of the models and determine global feature importance.
- **EDA Visualizations:** Target distributions, semantic heatmaps, and linguistic bigram rankings are automatically generated prior to model training.

---

## 🛠️ Repository Structure
```
📁 Nasyrah FYP/
│
├── 📁 data/                  # Raw datasets, NRC Lexicon, and extracted feature CSVs/NPYs
├── 📁 models/                # Trained .pkl models, TF-IDF vectorizers, and EDA/SHAP images
├── 📁 notebooks/             # Colab-ready Jupyter notebooks for heavy GPU training
├── 📁 src/
│   ├── extract_classical_features.py  # NLP pipeline (SpaCy, NRC, TF-IDF)
│   ├── train_advanced_models.py       # XGBoost/RF training logic
│   └── generate_eda.py                # EDA and visualization generation
│
├── streamlit_app.py          # Interactive Academic Presentation Dashboard
└── requirements.txt          # Python dependencies
```

---

## 💻 How to Run the Dashboard Locally

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd "Nasyrah FYP"
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Launch the Streamlit Dashboard:**
   ```bash
   streamlit run streamlit_app.py
   ```
   *The dashboard will automatically open in your web browser at `http://localhost:8501`, featuring 4 academic tabs: Project Overview, Exploratory Data Analysis, Model Performance, and a Live Interactive Demo.*

---

## 📊 Results Summary
By utilizing Feature Fusion and XGBoost, the model achieved a **>50% improvement** in predictive correlation (R²) compared to the linear baseline, proving that advanced non-linear algorithms combined with Deep Learning contextual embeddings can capture psychological nuances missed by traditional frequency-based NLP.
