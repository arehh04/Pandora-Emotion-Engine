import os
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel


def embed_texts(texts, tokenizer, model, device, batch_size=32):
    """Embed a list of strings into [CLS]-token vectors using the given tokenizer/model.

    Returns an (N, hidden_size) array, or shape (0, 0) if texts is empty.
    """
    if not texts:
        return np.zeros((0, 0))

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting BERT embeddings using device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.to(device)
    model.eval()

    splits = ["train", "validation", "test"]

    for split in splits:
        print(f"Processing {split} set...")
        clean_file = os.path.join(data_dir, f"{split}_clean.csv")
        if not os.path.exists(clean_file):
            print(f"File not found: {clean_file}")
            continue

        df = pd.read_csv(clean_file)
        texts = df["bert_text"].fillna("").tolist()

        embeddings_matrix = embed_texts(texts, tokenizer, model, device, batch_size=32)
        save_path = os.path.join(data_dir, f"{split}_bert_embeddings.npy")
        np.save(save_path, embeddings_matrix)
        print(f"Saved {split}_bert_embeddings.npy with shape {embeddings_matrix.shape}")

    # Augmented training rows need their own embeddings (new paraphrased texts
    # that don't exist in train_clean.csv), consumed by the retrained ML-prior tool.
    augmented_file = os.path.join(data_dir, "train_augmented.csv")
    if os.path.exists(augmented_file):
        print("Processing train_augmented set...")
        aug_df = pd.read_csv(augmented_file)
        texts = aug_df["bert_text"].fillna("").tolist()

        embeddings_matrix = embed_texts(texts, tokenizer, model, device, batch_size=32)
        save_path = os.path.join(data_dir, "train_augmented_bert_embeddings.npy")
        np.save(save_path, embeddings_matrix)
        print(f"Saved train_augmented_bert_embeddings.npy with shape {embeddings_matrix.shape}")
    else:
        print(f"File not found: {augmented_file} (skipping augmented embeddings)")


if __name__ == "__main__":
    main()
