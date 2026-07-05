import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import spacy

def load_nrc_lexicon(filepath):
    emotions_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                word_sense, emotion, score = parts
                word = word_sense.split('--')[0]
                if int(score) == 1:
                    if word not in emotions_dict:
                        emotions_dict[word] = set()
                    emotions_dict[word].add(emotion)
    return emotions_dict

def compute_linguistic_features(texts, nlp):
    features = []
    # enable parser and ner for sentence boundary detection and full pos tagging if needed, 
    # but for sentence boundary we just need parser or sentencizer. Let's add a sentencizer.
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
        
    for doc in nlp.pipe(texts, batch_size=256, disable=['parser', 'ner']):
        word_count = len(doc)
        sentence_count = len(list(doc.sents)) if word_count > 0 else 1
        punct_count = sum(1 for token in doc if token.is_punct)
        
        # POS tags
        nouns = sum(1 for token in doc if token.pos_ == "NOUN")
        verbs = sum(1 for token in doc if token.pos_ == "VERB")
        adjs = sum(1 for token in doc if token.pos_ == "ADJ")
        advs = sum(1 for token in doc if token.pos_ == "ADV")
        prons = sum(1 for token in doc if token.pos_ == "PRON")
        
        features.append({
            'word_count': word_count,
            'sentence_count': sentence_count,
            'punct_count': punct_count,
            'nouns_ratio': nouns / word_count if word_count > 0 else 0,
            'verbs_ratio': verbs / word_count if word_count > 0 else 0,
            'adjs_ratio': adjs / word_count if word_count > 0 else 0,
            'advs_ratio': advs / word_count if word_count > 0 else 0,
            'prons_ratio': prons / word_count if word_count > 0 else 0
        })
    return pd.DataFrame(features)

def compute_emotional_features(tokens_series, nrc_dict):
    emotions_list = ['anger', 'anticip', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust']
    features = []
    
    for text in tokens_series:
        tokens = text.split() if isinstance(text, str) else []
        counts = {emo: 0 for emo in emotions_list}
        total_words = len(tokens)
        
        for token in tokens:
            if token in nrc_dict:
                for emo in nrc_dict[token]:
                    if emo in counts:
                        counts[emo] += 1
                        
        # Normalize
        if total_words > 0:
            for emo in counts:
                counts[emo] /= total_words
                
        features.append(counts)
    
    return pd.DataFrame(features)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load NRC
    nrc_path = os.path.join(data_dir, "NRC-Emotion-Lexicon-Senselevel-v0.92.txt")
    print("Loading NRC Lexicon...")
    nrc_dict = load_nrc_lexicon(nrc_path)
    
    # 2. Setup TF-IDF
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # Process Train first to fit TF-IDF and Scaler
    print("Processing Train Set...")
    train_clean = pd.read_csv(os.path.join(data_dir, "train_clean.csv"))
    train_tokens = pd.read_csv(os.path.join(data_dir, "train_tokens.csv"))
    
    train_tfidf_matrix = tfidf.fit_transform(train_tokens['lemmatized_tokens'].fillna(''))
    train_tfidf_df = pd.DataFrame(train_tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(2000)])
    
    train_ling_df = compute_linguistic_features(train_clean['bert_text'].fillna(''), nlp)
    train_emo_df = compute_emotional_features(train_tokens['lemmatized_tokens'], nrc_dict)
    
    train_combined = pd.concat([train_ling_df, train_emo_df, train_tfidf_df], axis=1)
    train_combined['extraversion'] = train_tokens['extraversion']
    
    scaler = StandardScaler()
    feature_cols = [c for c in train_combined.columns if c != 'extraversion']
    train_combined[feature_cols] = scaler.fit_transform(train_combined[feature_cols])
    
    train_combined.to_csv(os.path.join(data_dir, "train_features.csv"), index=False)
    joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "feature_scaler.pkl"))
    print("Saved train_features.csv")
    
    # Process Validation and Test
    for split in ['validation', 'test']:
        print(f"Processing {split.capitalize()} Set...")
        clean_df = pd.read_csv(os.path.join(data_dir, f"{split}_clean.csv"))
        tokens_df = pd.read_csv(os.path.join(data_dir, f"{split}_tokens.csv"))
        
        tfidf_matrix = tfidf.transform(tokens_df['lemmatized_tokens'].fillna(''))
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(2000)])
        
        ling_df = compute_linguistic_features(clean_df['bert_text'].fillna(''), nlp)
        emo_df = compute_emotional_features(tokens_df['lemmatized_tokens'], nrc_dict)
        
        combined = pd.concat([ling_df, emo_df, tfidf_df], axis=1)
        combined['extraversion'] = tokens_df['extraversion']
        combined[feature_cols] = scaler.transform(combined[feature_cols])
        
        combined.to_csv(os.path.join(data_dir, f"{split}_features.csv"), index=False)
        print(f"Saved {split}_features.csv")

if __name__ == "__main__":
    main()
