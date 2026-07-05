import os
import re
import pandas as pd
import spacy

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers (keeping only letters and spaces)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def preprocess_dataset(input_file, output_clean_file, output_tokens_file, nlp):
    print(f"Processing {input_file}...")
    df = pd.read_csv(input_file)
    
    # Keep only required columns if they exist
    cols = [c for c in df.columns if c.lower() in ['text', 'extraversion']]
    df = df[cols]
    
    # Clean text for BERT (needs raw case/punctuation sometimes, but the WBS specified raw-case variant kept for BERT)
    # Wait, for BERT we usually keep punctuation. But for classical we remove it.
    # Let's save a "bert_clean" (only URLs removed) and a "classical_tokens"
    df['bert_text'] = df['text'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', str(x), flags=re.MULTILINE).strip())
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Drop rows where text is empty
    df = df[df['clean_text'].str.len() > 0]
    
    print("Tokenizing and lemmatizing...")
    # Tokenize and lemmatize
    tokens_list = []
    for doc in nlp.pipe(df['clean_text'], batch_size=256, disable=['parser', 'ner']):
        tokens = [token.lemma_ for token in doc if not token.is_stop]
        tokens_list.append(" ".join(tokens))
    
    df['lemmatized_tokens'] = tokens_list
    
    # Save BERT version (original case, just no URLs)
    bert_df = df[['bert_text', 'extraversion']] if 'extraversion' in df.columns else df[['bert_text']]
    bert_df.to_csv(output_clean_file, index=False)
    
    # Save classical tokens version
    tokens_df = df[['lemmatized_tokens', 'extraversion']] if 'extraversion' in df.columns else df[['lemmatized_tokens']]
    tokens_df.to_csv(output_tokens_file, index=False)
    print(f"Done. Saved to {output_clean_file} and {output_tokens_file}\n")

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        
    for split in ['train', 'validation', 'test']:
        in_file = os.path.join(data_dir, f"{split}_set.csv")
        if os.path.exists(in_file):
            preprocess_dataset(
                in_file,
                os.path.join(data_dir, f"{split}_clean.csv"),
                os.path.join(data_dir, f"{split}_tokens.csv"),
                nlp
            )
        else:
            print(f"Skipping {split}, file not found.")

if __name__ == "__main__":
    main()
