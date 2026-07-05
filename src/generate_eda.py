import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

def generate_eda(data_dir="data", output_dir="models"):
    print("Loading data for EDA...")
    # Load training features
    train_path = os.path.join(data_dir, "train_features.csv")
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return
        
    df = pd.read_csv(train_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Target Distribution (Histogram)
    print("Generating Target Distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['extraversion'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Extraversion Scores (Training Data)', fontsize=16)
    plt.xlabel('Extraversion Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_target_dist.png'))
    plt.close()
    
    # 2. Emotional Features Correlation Matrix
    print("Generating Emotional Correlation Matrix...")
    # The emotional columns are specifically named, let's grab them by checking what's left after linguistic and tfidf
    ling_cols = ['word_count', 'sentence_count', 'punct_count', 'nouns_ratio', 'verbs_ratio', 'adjs_ratio', 'advs_ratio', 'prons_ratio', 'extraversion']
    emo_cols = [c for c in df.columns if not c.startswith('tfidf_') and not c.startswith('bert_') and c not in ling_cols]
    
    if emo_cols:
        cols_to_corr = ['extraversion'] + emo_cols
        corr_matrix = df[cols_to_corr].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-0.2, vmax=0.2)
        plt.title('Correlation: Emotions vs Extraversion', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eda_emo_corr.png'))
        plt.close()
    
    # 3. Top TF-IDF Bigrams by Mean Score
    print("Generating Top Words/Bigrams plot...")
    tfidf_cols = [c for c in df.columns if c.startswith('tfidf_')]
    if tfidf_cols:
        try:
            # We need the vectorizer to map feature indices to actual words
            vectorizer = joblib.load(os.path.join(output_dir, "tfidf_vectorizer.pkl"))
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate mean TF-IDF score for each term across all documents
            mean_tfidf = df[tfidf_cols].mean().values
            
            # Sort by mean score
            top_indices = np.argsort(mean_tfidf)[-15:] # Top 15
            top_words = [feature_names[i] for i in top_indices]
            top_scores = [mean_tfidf[i] for i in top_indices]
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x=top_scores, y=top_words, palette='viridis')
            plt.title('Top 15 Most Frequent Linguistic Patterns (Bigrams)', fontsize=16)
            plt.xlabel('Average TF-IDF Score', fontsize=14)
            plt.ylabel('Word / Bigram', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'eda_top_words.png'))
            plt.close()
        except Exception as e:
            print(f"Could not generate word plot: {e}")
            
    print(f"EDA Generation complete. Images saved to {output_dir}/")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_eda(
        data_dir=os.path.join(base_dir, "data"),
        output_dir=os.path.join(base_dir, "models")
    )
