import json
import torch
import faiss
import numpy as np
from tqdm import tqdm
import os

from retriever_model import BiLSTMEncoder, SimpleTokenizer

# ---------------- CONFIG ----------------
PASSAGES_FILE = "../data/passages_trainval.json"
VAL_TRIPLETS  = "../data/retriever_triplets_val.json"
WORD2IDX_FILE = "../models/word2idx.json"
MODEL_FILE    = "../models/retriever_bilstm.pt"

EMB_DIM = 100
HIDDEN_DIM = 128
TOP_K = 20
DEVICE = "cpu"

# ---------------- LOAD ----------------
with open(PASSAGES_FILE, "r", encoding="utf-8") as f:
    passages = json.load(f)

with open(VAL_TRIPLETS, "r", encoding="utf-8") as f:
    val_triplets = json.load(f)

with open(WORD2IDX_FILE, "r", encoding="utf-8") as f:
    word2idx = json.load(f)

print(f"Loaded passages : {len(passages)}")
print(f"Loaded val queries : {len(val_triplets)}")

tokenizer = SimpleTokenizer(word2idx)

# ---------------- MODEL ----------------
model = BiLSTMEncoder(
    vocab_size=len(word2idx),
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- ENCODE PASSAGES ----------------
def encode_texts(texts, batch_size=32):
    vectors = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            ids = torch.stack([tokenizer.encode(t) for t in batch]).to(DEVICE)
            vec = model(ids)
            vectors.append(vec.cpu().numpy())
    return np.vstack(vectors)

print("Encoding passages...")
passage_texts = [p["content"] for p in passages]
passage_vecs = encode_texts(passage_texts)

# ---------------- FAISS INDEX ----------------
dim = passage_vecs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(passage_vecs)

print(f"FAISS index built with {index.ntotal} vectors")

# ---------------- EVALUATION ----------------
def recall_mrr(index, queries, passages, k=20):
    recall_hits = 0
    mrr_total = 0.0

    passage_article_ids = [p["article_id"] for p in passages]

    for q in tqdm(queries):
        query_vec = encode_texts([q["query"]])
        D, I = index.search(query_vec, k)

        retrieved_ids = [passage_article_ids[i] for i in I[0]]
        gold_id = q["meta"]["article_id"]

        if gold_id in retrieved_ids:
            recall_hits += 1
            rank = retrieved_ids.index(gold_id) + 1
            mrr_total += 1.0 / rank

    recall = recall_hits / len(queries) if queries else 0.0
    mrr = mrr_total / len(queries) if queries else 0.0
    return recall, mrr

print("Running Recall@20 / MRR evaluation...")
recall20, mrr = recall_mrr(index, val_triplets, passages, TOP_K)

print("\n---- RETRIEVER EVALUATION ----")
print(f"Recall@{TOP_K}: {recall20:.4f}")
print(f"MRR          : {mrr:.4f}")
