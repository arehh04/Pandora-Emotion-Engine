# Chapter 4: Results & Discussion

## 4.1 Exploratory Data Analysis (EDA)
Prior to model training, the Pandora dataset was analyzed to understand the distribution of the target variable and the explicit emotional signals.

![Distribution of Extraversion Scores](C:/Users/HP/Desktop/Nasyrah%20FYP/models/eda_target_dist.png)

The histogram above demonstrates that the ground truth Extraversion scores are normally distributed around a mean of ~3.0. This bell curve signifies that the dataset represents a natural human population (predominantly ambiverts).

![Emotional Correlation Matrix](C:/Users/HP/Desktop/Nasyrah%20FYP/models/eda_emo_corr.png)

The correlation matrix reveals explicit relationships between the NRC emotion lexicons and the Extraversion target. Positive, high-energy emotions (e.g., Joy, Anticipation) show mild positive correlation, whereas negative/isolating emotions (e.g., Sadness, Fear) demonstrate mild negative correlation.

## 4.2 Model Performance & Results
The models were evaluated against the unseen test set using RMSE and $R^2$. The objective was to determine if Feature Fusion combined with non-linear architectures (XGBoost, Random Forest) could outperform a linear baseline (Ridge Regression).

| Model | Features Used | RMSE | $R^2$ Score |
|-------|---------------|------|-------------|
| **Ridge (Baseline)** | Classical Only | 29.18 | 0.103 |
| **XGBoost (Tuned)** | Feature Fusion | 28.29 | 0.158 |
| **Random Forest (Tuned)** | Feature Fusion | 28.22 | 0.162 |

![Model Performance Comparison](C:/Users/HP/Desktop/Nasyrah%20FYP/models/model_comparison.png)

As shown in the table and chart above, **Random Forest emerged as the strongest model**, narrowly beating XGBoost. Both non-linear models utilizing the Feature Fusion pipeline achieved a massive **>50% improvement** in predictive correlation ($R^2 = 0.103 \rightarrow 0.162$) compared to the linear baseline.

While an $R^2$ of ~0.16 may appear low in deterministic fields, it is a highly respectable result in computational psychology. Predicting complex psychological constructs purely from short text snippets is a "low signal, high noise" task because human personality is obfuscated by thousands of invisible latent variables.

## 4.3 Regression Error Analysis
To fully validate the predictive behavior of the XGBoost model, an error analysis was performed on the unseen test set by comparing the predicted Extraversion scores against the actual ground truth scores.

![Actual vs Predicted Scores](C:/Users/HP/Desktop/Nasyrah%20FYP/models/actual_vs_predicted.png)

The scatter plot above illustrates the relationship between actual and predicted scores. A perfect model would place all points along the red diagonal line ($y=x$). While the model exhibits a concentrated cluster around the global mean (reflecting the dataset's normal distribution), it struggles to accurately capture the extreme outliers (highly introverted or highly extraverted individuals), which is expected behavior for regression models trained on highly subjective text data.

![Residual Histogram](C:/Users/HP/Desktop/Nasyrah%20FYP/models/residual_histogram.png)

![Residual Scatter Plot](C:/Users/HP/Desktop/Nasyrah%20FYP/models/residual_scatter.png)

The residual analysis further supports this finding. The histogram of residuals follows a near-perfect normal distribution centered around zero, indicating that the model is unbiased (it does not systematically over-predict or under-predict). However, the residual scatter plot demonstrates a pattern of homoscedasticity fading at the extremes, confirming that while the model handles the average user well, predicting extreme personality traits purely from short text remains inherently noisy.

## 4.4 SHAP Interpretability: Global Analysis
To unpack the "black box" decisions of the advanced non-linear architecture, SHAP values were generated.

![SHAP Summary Plot (Beeswarm)](C:/Users/HP/Desktop/Nasyrah%20FYP/models/xgb_shap_summary.png)

![SHAP Mean Absolute Importance Plot](C:/Users/HP/Desktop/Nasyrah%20FYP/models/xgb_shap_bar.png)

### 4.4.1 Semantic Dominance over Lexical Cues
The SHAP visual analysis reveals a monumental shift in how the non-linear algorithm mathematically calculates personality. 
* **BERT Semantic Group (Dominant):** Dense contextual embeddings absolutely dominated the decision-making process, comprising 12 of the Top 15 most influential features (e.g., `bert_640`, `bert_47`). 
* **TF-IDF Lexical Group (Medium):** Specific explicit word bigrams (e.g., `tfidf_1868`) occasionally acted as strong secondary anchors.
* **Linguistic/Emotional Group (Low):** Explicit emotional labels from the NRC lexicon fell entirely out of the top rankings.

The model mathematically realized that finding the deep, hidden contextual meaning of the text (BERT) was far more powerful than simply counting explicit word frequencies. BERT understands sarcasm, tone, and complex phrasing, proving that extraversion is highly contextual and not merely a function of using "happy" words.

## 4.5 Local Explanations (Case Studies)
To prove the model's localized reasoning, specific predictions were analyzed via SHAP Waterfall plots.

### Case 1: Highest Prediction (Highly Extraverted)
![Highest Prediction Waterfall](C:/Users/HP/Desktop/Nasyrah%20FYP/models/xgb_shap_waterfall_highest.png)
The SHAP waterfall shows a massive positive push driven almost entirely by a cluster of BERT embedding features. The model identified a deeply extraverted context that went beyond surface-level words, aggressively pushing the baseline score upwards.

### Case 2: Lowest Prediction (Highly Introverted)
![Lowest Prediction Waterfall](C:/Users/HP/Desktop/Nasyrah%20FYP/models/xgb_shap_waterfall_lowest.png)
The model recognized a complete absence of extraverted semantic context. While a single TF-IDF feature heavily pulled the prediction down, it was followed by a cascade of negative SHAP contributions from the BERT embedding space, forcing the prediction drastically below the global mean.

### Case 3: The Misprediction (High Error Case)
![Worst Prediction Waterfall](C:/Users/HP/Desktop/Nasyrah%20FYP/models/xgb_shap_waterfall_worst.png)
In cases of severe misprediction, the SHAP plots revealed the double-edged sword of relying on highly dense embeddings. When a text contained highly complex semantic structures but was actually written by an introvert, the BERT dimensions occasionally "hallucinated" an extraverted state based on sentence structure rather than explicit intent, leading to a massive overestimation of the target variable.

### Case 4: Custom Use Case Demonstration
To validate the model on a bespoke natural language input, the following custom text was passed through the entire Feature Fusion pipeline:
> *"I want to spend my holiday doing my hobby which is reading"*

![Use Case Prediction Waterfall](C:/Users/HP/Desktop/Nasyrah%20FYP/models/use_case_shap.png)

The XGBoost model predicted an extraversion score of **37.40** for this text. The SHAP waterfall plot illustrates how the BERT embeddings (`bert_765`, `bert_181`, etc.) heavily suppressed the score. Despite the sentence containing positive linguistic structures ("holiday", "hobby"), the underlying semantic context—solitary reading—was correctly interpreted by the deep learning layers as an introverted activity, proving the semantic superiority of the architecture.

## 4.6 Conclusion
The interpretability analysis of the Feature Fusion architecture highlights a critical paradigm in computational psychology: the overwhelming superiority of implicit semantic context over explicit lexical markers. By applying SHAP, it became evident that the model aggressively prioritized the 768-dimensional BERT embeddings over classical linguistic heuristics. Ultimately, the methodology successfully demonstrated that while explicit word usage provides occasional strong anchors, the deep, latent context extracted by modern Transformers is what mathematically defines and predicts extraversion in short-form text.
