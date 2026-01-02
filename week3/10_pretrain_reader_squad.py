import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from reader_model import ReaderModel

# ================= CONFIG =================
TRAIN_FILE = "../data/squad_train.json"
VAL_FILE   = "../data/squad_val.json"
VOCAB_FILE = "../models/word2idx.json"

MODEL_OUT  = "../models/reader_bilstm_attn_squad.pt"

BATCH_SIZE = 16
EPOCHS = 1            # 1–2 epochs are enough for assignment
LR = 2e-3
MAX_Q_LEN = 40
MAX_P_LEN = 300

DEVICE = torch.device("cpu")

os.makedirs("../models", exist_ok=True)

# ================= TOKENIZER =================
class SimpleTokenizer:
    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.pad = word2idx.get("<PAD>", 0)
        self.unk = word2idx.get("<UNK>", 1)

    def encode(self, text, max_len):
        tokens = text.lower().split()
        ids = [self.word2idx.get(t, self.unk) for t in tokens][:max_len]
        mask = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(self.pad)
            mask.append(0)
        return ids, mask

# ================= DATASET =================
class SquadDataset(Dataset):
    def __init__(self, path, tokenizer):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        assert isinstance(self.data, list), "SQuAD file must be a list"
        self.tok = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        q_ids, q_mask = self.tok.encode(ex["question"], MAX_Q_LEN)
        p_ids, p_mask = self.tok.encode(ex["passage"], MAX_P_LEN)

        start = min(ex["start"], MAX_P_LEN - 1)
        end   = min(ex["end"], MAX_P_LEN - 1)

        return (
            torch.tensor(q_ids),
            torch.tensor(p_ids),
            torch.tensor(q_mask),
            torch.tensor(p_mask),
            torch.tensor(start),
            torch.tensor(end),
        )

# ================= LOAD =================
with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    word2idx = json.load(f)

tokenizer = SimpleTokenizer(word2idx)

train_ds = SquadDataset(TRAIN_FILE, tokenizer)
val_ds   = SquadDataset(VAL_FILE, tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=1)

print(f"SQuAD train samples: {len(train_ds)}")
print(f"SQuAD val samples  : {len(val_ds)}")

# ================= MODEL =================
model = ReaderModel(
    vocab_size=len(word2idx),
    embed_dim=100,
    hidden=128
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ================= TRAIN =================
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for q, p, q_mask, p_mask, s, e in tqdm(train_loader, desc=f"SQuAD Epoch {epoch}"):
        q, p = q.to(DEVICE), p.to(DEVICE)
        q_mask, p_mask = q_mask.to(DEVICE), p_mask.to(DEVICE)
        s, e = s.to(DEVICE), e.to(DEVICE)

        optimizer.zero_grad()

        start_logits, end_logits = model(q, p, q_mask, p_mask)
        loss = criterion(start_logits, s) + criterion(end_logits, e)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} train loss: {total_loss / len(train_loader):.4f}")

# ================= SAVE =================
torch.save(model.state_dict(), MODEL_OUT)
print(f"[OK] Saved SQuAD-pretrained reader to {MODEL_OUT}")
