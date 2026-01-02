import sys
import os
import json
import random
import time
import torch
import numpy as np

# -------------------------------------------------
# PATH FIX
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
READER_MODEL    = "../models/reader_bilstm_attn.pt"

EMB_DIM = 100
HIDDEN_DIM = 128

TOP_K = 20
TOP_M = 5
MAX_LEN = 300
CONF_THRESH = 0.25

PROBES = 5000   # increase to 5000 if you want
DEVICE = torch.device("cpu")

LOG_FILE = "random_probe_log.txt"

random.seed(42)
torch.manual_seed(42)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

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
def main():
    open(LOG_FILE, "w").close()

    passages = load_json(PASSAGES_FILE)
    word2idx = load_json(WORD2IDX_FILE)

    retriever = BiLSTMEncoder(
        vocab_size=len(word2idx),
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM
    )
    retriever.load_state_dict(
        torch.load(RETRIEVER_MODEL, map_location=DEVICE)
    )
    retriever.eval()

    reader = ReaderModel(
        vocab_size=len(word2idx),
        embed_dim=EMB_DIM,
        hidden=HIDDEN_DIM
    )
    reader.load_state_dict(
        torch.load(READER_MODEL, map_location=DEVICE)
    )
    reader.eval()

    # Encode all passages once
    print("Encoding passages...")
    p_vecs = []
    with torch.no_grad():
        for p in passages:
            ids = tokenize(p["content"], word2idx).unsqueeze(0)
            vec = retriever(ids).squeeze(0)
            p_vecs.append(vec.numpy())
    p_vecs = np.stack(p_vecs)

    hits_retriever = 0
    hits_reader = 0

    for i in range(PROBES):
        start = time.time()

        # ---------------- RANDOM PICK ----------------
        gold = random.choice(passages)
        gold_id = gold["passage_id"]

        # Friendly query (title is closest to training distribution)
        query = gold["title"]

        # ---------------- RETRIEVER ----------------
        with torch.no_grad():
            q_ids = tokenize(query, word2idx).unsqueeze(0)
            q_vec = retriever(q_ids).squeeze(0).numpy()

        sims = np.dot(p_vecs, q_vec)
        topk_idx = sims.argsort()[-TOP_K:][::-1]
        retrieved = [passages[i] for i in topk_idx]

        retrieved_ids = [p["passage_id"] for p in retrieved]

        retriever_hit = gold_id in retrieved_ids
        if retriever_hit:
            hits_retriever += 1

        # ---------------- READER ----------------
        reader_hit = False

        if retriever_hit:
            candidates = retrieved[:TOP_M]

            with torch.no_grad():
                for p in candidates:
                    q_ids = tokenize(query, word2idx).unsqueeze(0)
                    p_ids = tokenize(p["content"], word2idx).unsqueeze(0)

                    q_mask = (q_ids != 0).long()
                    p_mask = (p_ids != 0).long()

                    start_logits, end_logits = reader(
                        p_ids, q_ids, p_mask, q_mask
                    )

                    conf = (start_logits.max() + end_logits.max()).item() / 2
                    if conf >= CONF_THRESH:
                        reader_hit = True
                        break

        if reader_hit:
            hits_reader += 1

        elapsed = time.time() - start

        log(
            f"[{i+1}/{PROBES}] "
            f"retriever={'✔' if retriever_hit else '✘'} | "
            f"reader={'✔' if reader_hit else '✘'} | "
            f"time={elapsed:.2f}s | "
            f"query='{query[:80]}'"
        )

        print(
            f"[{i+1}/{PROBES}] "
            f"retriever={'✔' if retriever_hit else '✘'} "
            f"reader={'✔' if reader_hit else '✘'} "
            f"time={elapsed:.2f}s"
        )

    # ---------------- SUMMARY ----------------
    retriever_recall = hits_retriever / PROBES
    reader_success   = hits_reader / PROBES

    print("\n---- RANDOM PROBE SUMMARY ----")
    print(f"Probes run        : {PROBES}")
    print(f"Retriever Recall@{TOP_K}: {retriever_recall:.4f}")
    print(f"Reader Success    : {reader_success:.4f}")

    log("\n---- SUMMARY ----")
    log(f"Retriever Recall@{TOP_K}: {retriever_recall:.4f}")
    log(f"Reader Success        : {reader_success:.4f}")

# -------------------------------------------------
# ENTRY
# -------------------------------------------------
if __name__ == "__main__":
    main()
