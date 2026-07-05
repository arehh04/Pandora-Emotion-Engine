import os
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Extracting BERT embeddings using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()
    
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        print(f"Processing {split} set...")
        clean_file = os.path.join(data_dir, f"{split}_clean.csv")
        if not os.path.exists(clean_file):
            print(f"File not found: {clean_file}")
            continue
            
        df = pd.read_csv(clean_file)
        
        # We process in batches to avoid OOM
        batch_size = 32
        all_embeddings = []
        
        texts = df['bert_text'].fillna('').tolist()
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)
                
        if all_embeddings:
            embeddings_matrix = np.vstack(all_embeddings)
            save_path = os.path.join(data_dir, f"{split}_bert_embeddings.npy")
            np.save(save_path, embeddings_matrix)
            print(f"Saved {split}_bert_embeddings.npy with shape {embeddings_matrix.shape}")

if __name__ == "__main__":
    main()
