# Pandora: The Emotion Engine 🧠

Pandora is an end-to-end machine learning system designed to predict a continuous psychological trait (Extraversion) directly from unstructured text. By moving beyond simple keyword matching, Pandora utilizes a robust **Feature Fusion** methodology that combines Classical Natural Language Processing (NLP) heuristics with the semantic depth of Deep Learning Transformer architectures.

---

## 📖 Project Overview (A to Z)

Psychological trait prediction from text is a complex regression problem. Words alone lack context, and context alone lacks explicit emotional definition. This project bridges that gap by extracting multiple dimensions of a user's text and predicting a continuous Extraversion score (0.0 to 100.0).

### 1. Data Preprocessing
Raw textual data undergoes extensive cleaning before touching any models:
* Removal of URLs, user tags, and special characters.
* Tokenization and Lemmatization via **SpaCy** (`en_core_web_sm`).
* Stop-word removal to isolate syntactically meaningful tokens.

### 2. Feature Extraction (The Fusion Approach)
To capture both explicit grammar and implicit meaning, the pipeline extracts three distinct feature sets:
1. **Linguistic Features:** Grammatical structures are quantified, extracting the ratios of Nouns, Verbs, and Adjectives.
2. **Emotional Features:** The **NRC Emotion Lexicon** is mapped against the text to extract frequencies across 10 discrete emotional categories (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation, Positive, Negative).
3. **Semantic Embeddings:** A pre-trained `bert-base-uncased` Transformer model extracts a 768-dimensional dense vector representing the deep contextual meaning of the text.

### 3. Machine Learning Modeling (Regression)
The extracted features were fed into a hierarchy of continuous regression models to evaluate performance:
* **Baseline Linear Model:** Ridge Regression.
* **Non-Linear Ensembles:** XGBoost and Random Forest algorithms trained on the fused feature space (TF-IDF + Linguistics + Emotions + Frozen BERT Embeddings).
* **State-of-the-Art Deep Learning:** A custom **Fine-Tuned BERT Regressor**, where a regression head was attached directly to the BERT architecture, allowing the model to dynamically update its attention weights specifically for Extraversion during training.

### 4. Explainability & Interpretability
Machine Learning should not be a black box. Pandora implements **SHAP (SHapley Additive exPlanations)** to unpack model decisions.
* **Token-Level SHAP:** The system breaks down the input text and assigns a positive or negative impact score to every single word, allowing users to see *exactly* which words influenced the AI to predict high or low extraversion.

---

## 📊 Evaluation & Results

The models were rigorously evaluated on an unseen test set using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and the Coefficient of Determination (R²). 

| Model Engine | RMSE | MAE | R² Score |
| :--- | :---: | :---: | :---: |
| **Ridge Regression** (Baseline) | ~27.42 | ~21.50 | 0.20 |
| **Random Forest** (Fused Features) | ~26.85 | ~21.10 | 0.24 |
| **XGBoost** (Fused Features) | ~26.30 | ~20.75 | 0.27 |
| **Fine-Tuned BERT Regressor** | **25.58** | **19.83** | **0.31** |

**Conclusion:** The Fine-Tuned BERT architecture significantly outperformed all classical and ensemble methods. Achieving an R² of 0.31 purely from short-form text is a highly significant result in the domain of continuous psychological regression.

---

## 🚀 System Architecture

Pandora is split into two robust, production-ready microservices.

### 1. The Backend (FastAPI + Caching)
A high-performance Python inference engine designed for Docker deployments.
* **FastAPI:** Serves the `/predict` REST endpoint.
* **Caching Layer:** Implements a pure-Python **Bloom Filter** for lightning-fast duplication checks, paired with **Redis** to cache model outputs and SHAP explanations. This prevents expensive BERT re-computation for previously seen text.
* **Orchestration:** Utilizes `supervisord` to run NGINX, Redis, and FastAPI simultaneously, acting as a reverse-proxy load balancer.

### 2. The Frontend (SvelteKit 5)
A beautiful, highly interactive web dashboard.
* Built on **Svelte 5** utilizing reactive `$state` runes.
* Features a dynamic, cinematic 24-second space/galaxy canvas animation.
* Automatically visualizes SHAP values and maps the prediction to an interactive SVG Radar Chart for implicit Big Five Personality traits.
* Generates downloadable PDF reports.

---

## 💻 Local Development Setup

### 1. Starting the Backend API
1. Open a terminal in the root directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # (or .\.venv\Scripts\activate on Windows)
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
4. Start the FastAPI server (make sure you have Redis installed locally if you want caching):
   ```bash
   uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
   ```

### 2. Starting the Frontend UI
1. Open a *second* terminal.
2. Navigate to the frontend directory:
   ```bash
   cd frontend
   npm install
   ```
3. Start the Svelte development server:
   ```bash
   npm run dev
   ```
4. Open your browser to `http://localhost:5173`.

---

## 🌍 Production Deployment

### Backend (Hugging Face Docker Spaces)
The backend is optimized for Hugging Face Docker Spaces.
1. Create a new Hugging Face Space (Select **Docker** SDK).
2. Clone your space and copy the repository contents (excluding heavy dataset CSVs).
3. Push to Hugging Face. The Space will read the `Dockerfile`, install NGINX and Redis, and use `supervisord` to manage the API internally on port `7860`.

### Frontend (Vercel)
The SvelteKit application is pre-configured with `@sveltejs/adapter-vercel`.
1. Ensure the `API_URL` in `frontend/src/routes/+page.svelte` points to your backend URL.
2. Import the GitHub repository into [Vercel](https://vercel.com/). Vercel will automatically detect the SvelteKit project and deploy the frontend to a global edge network.
