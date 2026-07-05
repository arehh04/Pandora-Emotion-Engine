# ─────────────────────────────────────────────────────────────────────────────
# Google Colab Fine-Tuning Script for BERT Regressor
# Run this entire file on Colab (Runtime → Run all)
# After completion, download: /content/bert_regressor_best.pt
# Place it in:  Nasyrah FYP/models/bert_regressor_best.pt
# ─────────────────────────────────────────────────────────────────────────────

# ── STEP 0: Install dependencies ──────────────────────────────────────────────
# !pip install transformers torch scikit-learn pandas numpy tqdm -q

import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── STEP 1: Mount Drive and load data ─────────────────────────────────────────
# Upload train_features.csv, val_features.csv, test_features.csv,
# train_bert_embeddings.npy, etc. to /content/ OR mount Google Drive.

# For simplicity, we use the raw CSV text column + extraversion target.
# If you have the raw CSV with 'text' column, load that.
# If you only have train_features.csv (no text), use the approach below.

TRAIN_CSV = "/content/train_set.csv"    # Must have columns: text, extraversion
VAL_CSV   = "/content/validation_set.csv"
TEST_CSV  = "/content/test_set.csv"

# ── STEP 2: Dataset class ─────────────────────────────────────────────────────
class ExtraversionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256):
        self.texts  = df["text"].astype(str).tolist()
        self.labels = df["extraversion"].astype(float).tolist()
        self.tok    = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.float32)
        }

# ── STEP 3: Model ─────────────────────────────────────────────────────────────
class BertRegressorModel(nn.Module):
    def __init__(self, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden = self.bert.config.hidden_size  # 768
        self.regressor = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           output_attentions=True)
        cls    = out.last_hidden_state[:, 0, :]
        score  = self.regressor(cls).squeeze(-1) * 99.0
        return score, out.attentions

# ── STEP 4: Training loop ─────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    criterion  = nn.MSELoss()
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        preds, _ = model(ids, mask)
        loss     = criterion(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            preds, _ = model(ids, mask)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].numpy())
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    mae  = mean_absolute_error(all_labels, all_preds)
    r2   = r2_score(all_labels, all_preds)
    return rmse, mae, r2

# ── STEP 5: Main training ─────────────────────────────────────────────────────
EPOCHS     = 3
BATCH_SIZE = 16
LR         = 2e-5
MAX_LEN    = 256

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_df = pd.read_csv(TRAIN_CSV)[["text", "extraversion"]].dropna()
val_df   = pd.read_csv(VAL_CSV)[["text", "extraversion"]].dropna()
test_df  = pd.read_csv(TEST_CSV)[["text", "extraversion"]].dropna()

train_ds = ExtraversionDataset(train_df, tokenizer, MAX_LEN)
val_ds   = ExtraversionDataset(val_df,   tokenizer, MAX_LEN)
test_ds  = ExtraversionDataset(test_df,  tokenizer, MAX_LEN)

# Calculate sample weights to correct Introvert Dataset Bias
from torch.utils.data import WeightedRandomSampler
train_labels = train_df["extraversion"].values
# Give 8x weight to high extraversion scores (>= 65) to balance the long tail
sample_weights = [8.0 if l >= 65 else 1.0 for l in train_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = BertRegressorModel().to(DEVICE)

optimizer  = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_dl) * EPOCHS
scheduler  = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

log = []
best_val_rmse = float("inf")
best_model_path = "/content/bert_regressor_best.pt"

print(f"\n{'='*60}")
print(f"  Training BERT Regressor  |  Device: {DEVICE}")
print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
print(f"{'='*60}\n")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss = train_epoch(model, train_dl, optimizer, scheduler, DEVICE)
    val_rmse, val_mae, val_r2 = eval_epoch(model, val_dl, DEVICE)
    elapsed = time.time() - t0

    log.append({
        "epoch": epoch,
        "train_loss": round(train_loss, 4),
        "val_rmse":   round(val_rmse, 4),
        "val_mae":    round(val_mae, 4),
        "val_r2":     round(val_r2, 4)
    })

    print(f"Epoch {epoch}/{EPOCHS}  |  "
          f"Loss: {train_loss:.4f}  |  "
          f"Val RMSE: {val_rmse:.4f}  |  "
          f"Val MAE: {val_mae:.4f}  |  "
          f"Val R²: {val_r2:.4f}  |  "
          f"Time: {elapsed:.0f}s")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✅ Best model saved at epoch {epoch} (RMSE={val_rmse:.4f})")

# Save training log
log_df = pd.DataFrame(log)
log_df.to_csv("/content/bert_training_log.csv", index=False)
print("\nTraining log saved to /content/bert_training_log.csv")

# ── STEP 6: Final test evaluation ─────────────────────────────────────────────
print("\nLoading best model for final test evaluation...")
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
test_rmse, test_mae, test_r2 = eval_epoch(model, test_dl, DEVICE)
print(f"\n{'='*60}")
print(f"  FINAL TEST RESULTS — Fine-Tuned BERT Regressor")
print(f"  RMSE : {test_rmse:.4f}")
print(f"  MAE  : {test_mae:.4f}")
print(f"  R²   : {test_r2:.4f}")
print(f"{'='*60}\n")

metrics = {"test_rmse": test_rmse, "test_mae": test_mae, "test_r2": test_r2}
with open("/content/bert_test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Done! Download these files:")
print("  /content/bert_regressor_best.pt     ← model weights")
print("  /content/bert_training_log.csv      ← loss curve data")
print("  /content/bert_test_metrics.json     ← final metrics")
print("\nPlace all files into:  Nasyrah FYP/models/")
