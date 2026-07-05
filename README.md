# Pandora: The Emotion Engine 🧠
**Final Year Project (FYP) by Nasyrah**

Pandora is an end-to-end machine learning system that predicts Extraversion personality scores from raw text. The project utilizes a "Feature Fusion" approach, combining Classical NLP heuristics (linguistics & emotions) with Deep Learning semantics (Fine-Tuned BERT).

This repository contains the full source code for the **SvelteKit Frontend** UI and the **FastAPI Backend** inference engine.

---

## 🚀 Project Architecture

### 1. The Backend (FastAPI + Deep Learning)
The backend is a high-performance Python inference engine designed for production deployment via Docker.
* **FastAPI:** Serves the `/predict` REST endpoint.
* **Feature Fusion:**
  * **Classical Branch:** Analyzes grammatical structures (Noun/Verb ratios via `SpaCy`) and extracts 10 emotional frequencies via the **NRC Emotion Lexicon**.
  * **Deep Learning Branch:** Utilizes a custom **Fine-Tuned BERT Regressor** (`bert-base-uncased`) to extract deep semantic dimensions and contextual embeddings.
* **Caching Layer:**
  * Implements an in-memory **Bloom Filter** for lightning-fast duplication checks.
  * Uses **Redis** to cache model outputs and SHAP explanations, preventing expensive re-computation of the BERT model for previously seen text.
* **Orchestration:** Uses `supervisord` inside Docker to run NGINX, Redis, and FastAPI simultaneously, acting as a reverse-proxy load balancer.

### 2. The Frontend (SvelteKit 5)
The frontend is a beautifully crafted, highly interactive web dashboard.
* Built on **Svelte 5** (utilizing reactive `$state` runes).
* Features a dynamic 24-second infinite space/galaxy canvas animation.
* Automatically visualizes **SHAP (SHapley Additive exPlanations)** token values to highlight exactly which words influenced the model's prediction.
* Maps the prediction to implicit Big Five Personality traits via an interactive SVG Radar Chart.
* Generates downloadable PDF reports ("Seal into Parchment") summarizing the live analysis.

---

## 🛠️ Repository Structure
```text
📁 Pandora-Emotion-Engine/
│
├── 📁 backend/               # FastAPI application logic and API endpoints
├── 📁 frontend/              # SvelteKit 5 User Interface
├── 📁 src/                   # Core ML logic (Feature Extraction, Training, SHAP)
├── 📁 models/                # Trained .pkl models, TF-IDF vectorizers, and PyTorch weights
├── 📁 data/                  # Lexicons (Note: Large training datasets are excluded via gitignore)
├── 📁 notebooks/             # Colab-ready Jupyter notebooks for heavy GPU training
│
├── Dockerfile                # Multi-service container setup for Hugging Face Spaces
├── supervisord.conf          # Process manager configuration
├── nginx.conf                # NGINX reverse-proxy configuration
└── requirements.txt          # Python dependencies
```

---

## 💻 Local Development Setup

### 1. Starting the Backend API
1. Open a terminal in the root directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```
3. Install the dependencies and required SpaCy models:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
4. Start the FastAPI server (make sure you have Redis installed locally if you want caching, otherwise it gracefully disables it):
   ```bash
   uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
   ```

### 2. Starting the Frontend UI
1. Open a *second* terminal.
2. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
3. Install Node.js dependencies:
   ```bash
   npm install
   ```
4. Start the Svelte development server:
   ```bash
   npm run dev
   ```
5. Open your browser to `http://localhost:5173`.

---

## 🌍 Production Deployment

### Backend (Hugging Face Docker Spaces)
The backend is optimized for Hugging Face Docker Spaces (which only expose a single port: 7860).
1. Create a new Hugging Face Space (Select **Docker** SDK).
2. Clone your space and copy the repository contents (excluding heavy dataset CSVs).
3. Push to Hugging Face. The Space will read the `Dockerfile`, install NGINX and Redis, and use `supervisord` to manage the API internally.

### Frontend (Vercel)
The SvelteKit application is pre-configured with `@sveltejs/adapter-vercel`.
1. Ensure the `API_URL` in `frontend/src/routes/+page.svelte` points to your Hugging Face space URL (e.g., `https://your-username-pandora.hf.space/predict`).
2. Import the GitHub repository into [Vercel](https://vercel.com/). Vercel will automatically detect the SvelteKit project and deploy the frontend to a global edge network.
