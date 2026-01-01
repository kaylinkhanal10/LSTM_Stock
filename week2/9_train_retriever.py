import json
import os
import re
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

from retriever_model import BiLSTMEncoder, SimpleTokenizer

# ================= CONFIG =================
PASSAGES_FILE = "../data/passages_train.json"
TRIPLETS_FILE = "../data/retriever_triplets_train.json"
GLOVE_FILE    = "../embeddings/glove.6B.100d.txt"

MODEL_DIR = "../models"
MODEL_FILE = os.path.join(MODEL_DIR, "retriever_bilstm.pt")
VOCAB_FILE = os.path.join(MODEL_DIR, "word2idx.json")

EMB_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 5
MARGIN = 1.0
DEVICE = "cpu"

random.seed(42)
torch.manual_seed(42)
os.makedirs(MODEL_DIR, exist_ok=True)

# ================= LOAD DATA =================
with open(PASSAGES_FILE, "r", encoding="utf-8") as f:
    passages = json.load(f)

with open(TRIPLETS_FILE, "r", encoding="utf-8") as f:
    triplets = json.load(f)

print(f"Loaded passages : {len(passages)}")
print(f"Loaded triplets : {len(triplets)}")

# ================= BUILD VOCAB =================
def build_vocab(passages, min_freq=2):
    counter = Counter()
    for p in passages:
        tokens = re.findall(r"\w+", p["content"].lower())
        counter.update(tokens)

    vocab = ["<PAD>", "<UNK>"]
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab.append(word)

    return {w: i for i, w in enumerate(vocab)}

word2idx = build_vocab(passages)

with open(VOCAB_FILE, "w", encoding="utf-8") as f:
    json.dump(word2idx, f, indent=2)

print(f"[OK] Saved word2idx.json with {len(word2idx)} tokens")

# ================= LOAD GLOVE =================
def load_glove(glove_path, word2idx, emb_dim):
    embeddings = np.random.normal(scale=0.6, size=(len(word2idx), emb_dim))
    embeddings[0] = 0.0  # PAD

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            word = parts[0]
            if word in word2idx:
                embeddings[word2idx[word]] = np.asarray(parts[1:], dtype=np.float32)

    return torch.tensor(embeddings, dtype=torch.float)

embeddings = load_glove(GLOVE_FILE, word2idx, EMB_DIM)

# ================= TOKENIZER =================
tokenizer = SimpleTokenizer(word2idx)

# ================= DATASET =================
class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]
        return (
            tokenizer.encode(t["query"]),
            tokenizer.encode(t["positive"]),
            tokenizer.encode(t["negative_1"]),
        )

dataset = TripletDataset(triplets)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ================= MODEL =================
model = BiLSTMEncoder(
    vocab_size=len(word2idx),
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    embeddings=embeddings
).to(DEVICE)

criterion = nn.TripletMarginLoss(margin=MARGIN)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ================= TRAIN =================
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0

    for q, p, n in tqdm(loader, desc=f"Epoch {epoch+1}"):
        q, p, n = q.to(DEVICE), p.to(DEVICE), n.to(DEVICE)

        optimizer.zero_grad()

        q_vec = model(q)
        p_vec = model(p)
        n_vec = model(n)

        loss = criterion(q_vec, p_vec, n_vec)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")

# ================= SAVE MODEL =================
torch.save(model.state_dict(), MODEL_FILE)
print("[OK] Saved retriever model.")
