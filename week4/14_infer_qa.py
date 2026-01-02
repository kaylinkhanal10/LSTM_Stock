import sys
import os
import argparse
import json
import torch
import numpy as np

# -------------------------------------------------
# PATH FIX (import models from previous weeks)
# -------------------------------------------------
sys.path.append(os.path.abspath("../week2"))
sys.path.append(os.path.abspath("../week3"))

from retriever_model import BiLSTMEncoder
from reader_model import ReaderModel

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
PASSAGES_FILE = "../data/passages_trainval.json"
WORD2IDX_FILE = "../models/word2idx.json"

RETRIEVER_MODEL = "../models/retriever_bilstm.pt"
# READER_MODEL    = "../models/reader_bilstm_attn.pt"
READER_MODEL    = "../models/reader_bilstm_attn_squad.pt"

EMB_DIM = 100
HIDDEN_DIM = 128

TOP_K = 20
TOP_M = 5
CONF_THRESH = 0.05
MAX_LEN = 300

DEVICE = torch.device("cpu")

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def tokenize(text, word2idx, max_len=MAX_LEN):
    tokens = text.lower().split()
    ids = [word2idx.get(t, 1) for t in tokens][:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main(question):
    print(f"\nQuestion: {question}\n")

    passages = load_json(PASSAGES_FILE)
    word2idx = load_json(WORD2IDX_FILE)

    # ================= RETRIEVER =================
    retriever = BiLSTMEncoder(
        vocab_size=len(word2idx),
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM
    )
    retriever.load_state_dict(
        torch.load(RETRIEVER_MODEL, map_location=DEVICE)
    )
    retriever.eval()

    # Encode passages
    p_vecs = []
    with torch.no_grad():
        for p in passages:
            ids = tokenize(p["content"], word2idx).unsqueeze(0)
            vec = retriever(ids).squeeze(0)
            p_vecs.append(vec.numpy())

    p_vecs = np.stack(p_vecs)

    # Encode query
    with torch.no_grad():
        q_ids = tokenize(question, word2idx).unsqueeze(0)
        q_vec = retriever(q_ids).squeeze(0).numpy()

    # Cosine similarity
    sims = np.dot(p_vecs, q_vec)
    topk_idx = sims.argsort()[-TOP_K:][::-1]
    candidates = [passages[i] for i in topk_idx[:TOP_M]]

    # ================= READER =================
    reader = ReaderModel(
        len(word2idx),
        EMB_DIM,
        HIDDEN_DIM
    )
    reader.load_state_dict(
        torch.load(READER_MODEL, map_location=DEVICE)
    )
    reader.eval()

    best = None

    with torch.no_grad():
        for p in candidates:
            # Tokenize separately (REQUIRED)
            q_ids = tokenize(question, word2idx).unsqueeze(0)
            p_ids = tokenize(p["content"], word2idx).unsqueeze(0)

            q_mask = (q_ids != 0).long()
            p_mask = (p_ids != 0).long()

            start_logits, end_logits = reader(
                p_ids,
                q_ids,
                p_mask,
                q_mask
            )

            s = start_logits.argmax(dim=1).item()
            e = end_logits.argmax(dim=1).item()

            if e < s:
                continue

            tokens = p["content"].split()
            if e >= len(tokens):
                continue

            answer = " ".join(tokens[s:e + 1])
            conf = (start_logits.max() + end_logits.max()).item() / 2

            if best is None or conf > best["confidence"]:
                best = {
                    "answer": answer,
                    "confidence": conf,
                    "source": p["source"],
                    "title": p["title"],
                    "url": p["url"],
                    "date": p.get("publishedISO")
                }

    # ================= OUTPUT =================
    if best is None or best["confidence"] < CONF_THRESH:
        print("Answer: not found\n")
        print("Top sources:")
        for i, p in enumerate(candidates, 1):
            print(f"{i}. {p['source']} – {p['title']} ({p.get('publishedISO')})")
        return

    print(f"Answer     : {best['answer']}")
    print(f"Confidence : {best['confidence']:.3f}")
    print(f"Source     : {best['source']}")
    print(f"Date       : {best['date']}")
    print(f"URL        : {best['url']}")

# -------------------------------------------------
# ENTRY
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    main(args.question)
