# Chapter 3: Methodology

## 3.1 Introduction
This chapter details the end-to-end machine learning architecture developed to predict human Extraversion from short text snippets. Given the high noise inherent in human behavioral data, a "Feature Fusion" methodology was designed to mathematically bridge classical NLP heuristics and modern Deep Learning contextual semantics.

## 3.2 Research Methodology (DSR Framework)
The development of the ExtraVerse system follows the Design Science Research (DSR) methodology. This framework is highly suitable for this thesis as it focuses on the creation and evaluation of an innovative IT artifact (the predictive pipeline and UI) intended to solve an identified scientific problem.

### Research Objectives (RO)
The research methodology is guided by three primary Research Objectives (RO), which dictate the phases of the project:
* **RO I:** To collect textual data, preprocess it, and engineer an integrated feature representation (Feature Fusion combining classical and semantic NLP).
* **RO II:** To develop, tune, and formally evaluate machine learning models (Ridge Regression, XGBoost, Random Forest) for predicting Extraversion scores.
* **RO III:** To analyze the predictive decisions using explainable AI (SHAP) and deploy the finalized model into an interactive web system (ExtraVerse).

### DSR, WBS, and Deliverable Alignment
The execution of this project is organized into an 8-phase Work Breakdown Structure (WBS). The following table meticulously aligns each standard DSR cycle with the corresponding WBS phases, activities, exact deliverables, and fulfilling Research Objectives:

| DSR Phase | WBS Phase & Key Activities | Deliverable | Research Objective (RO) |
|-----------|----------------------------|-------------|-------------------------|
| **1. Problem Identification** | **Phase 1: Preliminary Study**<br>- Review personality prediction literature<br>- Survey NLP and ML modeling techniques | Research Framework | **RO I** |
| **2. Objectives of a Solution** | **Phase 1: Preliminary Study**<br>- Define candidate feature categories<br>- Finalize research taxonomy | Research Framework | **RO I** |
| **3. Design and Development** | **Phase 2: Data Collection & Preprocessing**<br>- Clean, tokenize, and lemmatize Pandora dataset<br>**Phase 3: Feature Engineering**<br>- Extract TF-IDF, Linguistic, NRC, and BERT features<br>**Phase 4: Model Development**<br>- Train and fine-tune Ridge, XGBoost, and RF models<br>**Phase 7: System Development**<br>- Build Gradio UI interface and inference pipeline | Cleaned Dataset<br><br>Feature Representations<br><br>Developed Models<br><br>ExtraVerse System | **RO I**<br><br>**RO I**<br><br>**RO II**<br><br>**RO III** |
| **4. Demonstration** | **Phase 7: System Development**<br>- Package and deploy interactive ExtraVerse web app | ExtraVerse System | **RO III** |
| **5. Evaluation** | **Phase 5: Model Evaluation**<br>- Compute RMSE, $R^2$ metrics and compare model performance<br>**Phase 6: Explainability Analysis**<br>- Compute SHAP values and generate local/global feature importance plots | Evaluation Results<br><br>SHAP Results | **RO II**<br><br>**RO III** |
| **6. Communication** | **Phase 8: Documentation**<br>- Consolidate findings against P1-P4<br>- Write thesis chapters and build defense slide deck | Complete Thesis Document and Slides | **All ROs** |

## 3.3 System Architecture
The predictive pipeline operates in five distinct phases:
1. **Data Preprocessing:** Cleaning and normalizing raw user text.
2. **Feature Extraction (Parallel):** Simultaneously running Classical extraction (TF-IDF, NRC, POS) and Semantic extraction (BERT).
3. **Feature Fusion:** Concatenating the branches into a high-dimensional feature space ($d = 2,776$).
4. **Machine Learning Modeling:** Feeding the normalized data into linear (Ridge) and non-linear ensemble (XGBoost, Random Forest) algorithms.
5. **Interpretability (SHAP):** Unpacking the model decisions to evaluate feature importance.

## 3.4 Data Preprocessing
Raw textual data from the Pandora dataset was normalized using the `SpaCy` natural language processing library. The text underwent lowercase conversion to ensure uniformity. Stop-words (e.g., "the", "and") were removed to reduce dimensionality. Finally, words were lemmatized to their base dictionary form (e.g., "running" $\rightarrow$ "run") to ensure semantic consistency across grammatical tenses.

## 3.5 Feature Extraction

### 3.5.1 Classical Lexical & Emotional Features
To capture explicit word usage, Term Frequency-Inverse Document Frequency (TF-IDF) was applied to extract the top 2,000 word bigrams. TF-IDF evaluates how important a word is to a document in a collection or corpus, formulated as:
$$ TFIDF(t, d, D) = tf(t, d) \cdot \log\left(\frac{N}{|\{d \in D : t \in d\}|}\right) $$
Where $tf$ is the term frequency, $N$ is the total number of documents, and the denominator is the number of documents containing term $t$.

Additionally, emotional frequencies were mapped using the **NRC Emotion Lexicon**, counting the occurrences of 10 discrete emotional states (Joy, Anger, Trust, etc.). Linguistic structures were measured by calculating the ratios of Parts-of-Speech (POS) tags such as Nouns, Verbs, and Adjectives.

### 3.5.2 Deep Learning Contextual Extraction (BERT)
To capture the implicit psychological context of the text, Google's pre-trained Transformer model (`bert-base-uncased`) was utilized. BERT relies on the Multi-Head Self-Attention mechanism, defined mathematically as:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
Where $Q$, $K$, and $V$ represent the Query, Key, and Value matrices respectively, and $d_k$ is the scaling factor.

> [!NOTE] 
> **Illustration Guide: The BERT Dimension Extraction**
> *(Use this text to draw your architectural diagram for this section!)*
> 
> **Step 1 (Input):** The raw sentence (e.g., "I went to the party") enters the BERT Tokenizer.
> **Step 2 (Self-Attention):** The Transformer layers mathematically weigh the importance of every word against every other word in the sentence (context).
> **Step 3 (Pooling):** The output from the special `[CLS]` token is extracted.
> **Step 4 (Output Vector):** The model outputs a single **768-dimensional dense vector** (an array of 768 decimal numbers). This single vector acts as the "mathematical thought" or fingerprint of the entire sentence, capturing sarcasm, tone, and deep meaning.

### 3.5.3 Feature Fusion
The explicit classical features ($d = 2018$) and the implicit BERT embeddings ($d = 768$) were concatenated via a horizontal matrix stack into a unified feature space ($N = 16,047$ rows, $d = 2,776$ columns). A `StandardScaler` was applied to ensure the massive TF-IDF frequencies did not overpower the small, dense BERT decimals.

## 3.6 Machine Learning Algorithms

### 3.6.1 Ridge Regression (Baseline)
Ridge Regression applies L2 regularization to standard linear regression to prevent overfitting in high-dimensional spaces. The objective is to minimize the cost function:
$$ J(\theta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \theta_j^2 $$
Where $\lambda$ is the penalty term shrinking the coefficients $\theta$.

### 3.6.2 XGBoost (Advanced Non-Linear)
eXtreme Gradient Boosting (XGBoost) sequentially builds decision trees to correct the residual errors of previous trees. It was chosen for its superior handling of high-dimensional, sparse data (like TF-IDF). Its objective function includes a regularization term $\Omega$ to control tree complexity:
$$ \mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) $$

### 3.6.3 Random Forest (Advanced Ensemble)
Random Forest utilizes bootstrap aggregating (bagging) to build hundreds of deep decision trees independently and averages their predictions. This ensemble technique reduces variance and prevents overfitting, making it highly robust to noise in behavioral data.

## 3.7 Experimental Setup and Hyperparameter Tuning
To ensure robust and unbiased model evaluation, the dataset was partitioned into an 80% training set and a 20% unseen test set. The training set was utilized to fit the algorithms and identify the optimal configuration for the complex non-linear models. 

Hyperparameter tuning was conducted using a grid search methodology combined with 5-fold cross-validation. This approach systematically evaluates various parameter combinations to minimize the Root Mean Squared Error (RMSE) on the validation folds, thereby actively preventing overfitting to the training data.

For the primary XGBoost architecture, the experimental tuning process resulted in the following optimal configurations:
* **n_estimators (300):** A relatively high number of boosting rounds allows the model to learn complex latent relationships slowly.
* **learning_rate (0.05):** A conservative learning rate prevents the model from over-correcting residuals too rapidly.
* **max_depth (3):** Restricting tree depth acts as a strong regularizer, stopping the model from memorizing the highly noisy textual training data.
* **subsample & colsample_bytree (1.0):** Utilizing the full set of rows and columns per tree was mathematically optimal given the already high dimensionality of the Feature Fusion space.

Conversely, the Ridge baseline model was optimized primarily on its L2 regularization strength ($\alpha$), while the Random Forest ensemble was heavily tuned for tree count and maximum depth.

## 3.8 Evaluation Metrics
Model performance was evaluated on a held-out test set using two standard regression metrics.

**Root Mean Squared Error (RMSE):**
Measures the standard deviation of the prediction errors (residuals). Lower values indicate better fit.
$$ RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $$

**Coefficient of Determination ($R^2$):**
Measures the proportion of the variance in the dependent variable (Extraversion) that is predictable from the independent variables. Higher values indicate a stronger correlation.
$$ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} $$

## 3.9 Interpretability (SHAP)
To unpack the "black box" decisions of the XGBoost model, SHAP (SHapley Additive exPlanations) values were employed. Grounded in cooperative game theory, SHAP calculates the marginal contribution of a feature $i$ across all possible feature subsets:
$$ \phi_i(f, x) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} \left[ f(S \cup \{i\}) - f(S) \right] $$
This provides a unified measure of global and local feature importance, allowing us to scientifically prove which Feature Fusion group (BERT vs. TF-IDF) drove the final prediction.
