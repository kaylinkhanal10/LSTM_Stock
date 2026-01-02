import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter

from reader_model import ReaderModel

# ---------------- CONFIG ----------------
TRAIN_FILE = "../data/reader_train.json"
VAL_FILE   = "../data/reader_val.json"
VOCAB_FILE = "../models/word2idx.json"

MODEL_OUT  = "../models/reader_bilstm_attn.pt"

BATCH_SIZE = 16
EPOCHS = 6
LR = 2e-3
MAX_Q_LEN = 40
MAX_P_LEN = 300

DEVICE = torch.device("cpu")  # assignment-safe

os.makedirs("../models", exist_ok=True)

# ---------------- TOKENIZER ----------------
class SimpleTokenizer:
    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.unk = word2idx.get("<unk>", 1)
        self.pad = word2idx.get("<pad>", 0)

    def encode(self, text, max_len):
        tokens = text.lower().split()
        ids = [self.word2idx.get(t, self.unk) for t in tokens][:max_len]
        mask = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(self.pad)
            mask.append(0)
        return ids, mask

# ---------------- DATASET ----------------
class ReaderDataset(Dataset):
    def __init__(self, path, tokenizer):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tok = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        q_ids, q_mask = self.tok.encode(ex["question"], MAX_Q_LEN)
        p_ids, p_mask = self.tok.encode(ex["passage"], MAX_P_LEN)

        start = ex["start"]
        end   = ex["end"]

        # Clamp spans if tokenization shortened passage
        start = min(start, MAX_P_LEN - 1)
        end   = min(end, MAX_P_LEN - 1)

        return (
            torch.tensor(q_ids),
            torch.tensor(p_ids),
            torch.tensor(q_mask),
            torch.tensor(p_mask),
            torch.tensor(start),
            torch.tensor(end),
            ex["answer"]
        )

# ---------------- METRICS ----------------
def compute_em_f1(pred, gold):
    pred = pred.lower().split()
    gold = gold.lower().split()

    if not pred or not gold:
        return 0, 0

    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())

    if num_same == 0:
        return 0, 0

    precision = num_same / len(pred)
    recall = num_same / len(gold)
    f1 = 2 * precision * recall / (precision + recall)

    em = int(" ".join(pred) == " ".join(gold))
    return em, f1

# ---------------- LOAD ----------------
with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    word2idx = json.load(f)

tokenizer = SimpleTokenizer(word2idx)

train_ds = ReaderDataset(TRAIN_FILE, tokenizer)
val_ds   = ReaderDataset(VAL_FILE, tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=1)

print(f"Train samples: {len(train_ds)}")
print(f"Val samples  : {len(val_ds)}")

# ---------------- MODEL ----------------
model = ReaderModel(
    vocab_size=len(word2idx),
    embed_dim=100,
    hidden=128
).to(DEVICE)

model.load_state_dict(
    torch.load("../models/reader_bilstm_attn_squad.pt", map_location=DEVICE)
)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_f1 = 0.0

# ---------------- TRAIN ----------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for q, p, q_mask, p_mask, s, e, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
        q, p = q.to(DEVICE), p.to(DEVICE)
        q_mask, p_mask = q_mask.to(DEVICE), p_mask.to(DEVICE)
        s, e = s.to(DEVICE), e.to(DEVICE)

        optimizer.zero_grad()

        start_logits, end_logits = model(q, p, q_mask, p_mask)

        loss = criterion(start_logits, s) + criterion(end_logits, e)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} train loss: {total_loss/len(train_loader):.4f}")

    # -------- VALIDATION --------
    model.eval()
    ems, f1s = [], []

    with torch.no_grad():
        for q, p, q_mask, p_mask, s, e, gold in val_loader:
            q, p = q.to(DEVICE), p.to(DEVICE)
            q_mask, p_mask = q_mask.to(DEVICE), p_mask.to(DEVICE)

            start_logits, end_logits = model(q, p, q_mask, p_mask)

            ps = torch.argmax(start_logits, dim=1).item()
            pe = torch.argmax(end_logits, dim=1).item()

            if ps > pe:
                pred = ""
            else:
                tokens = p[0, ps:pe+1]
                inv = {v: k for k, v in word2idx.items()}
                pred = " ".join(inv.get(i.item(), "") for i in tokens)

            em, f1 = compute_em_f1(pred, gold[0])
            ems.append(em)
            f1s.append(f1)

    avg_em = sum(ems) / len(ems)
    avg_f1 = sum(f1s) / len(f1s)

    print(f"Val EM: {avg_em:.4f} | Val F1: {avg_f1:.4f}")

    if avg_f1 > best_f1:
        best_f1 = avg_f1
        torch.save(model.state_dict(), MODEL_OUT)
        print("✔ Saved best reader model")

print("\nTraining complete.")
print(f"Best Val F1: {best_f1:.4f}")
