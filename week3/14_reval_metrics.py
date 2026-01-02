import json
import torch
from collections import Counter
from tqdm import tqdm

from reader_model import ReaderModel

# ---------------- CONFIG ----------------
VAL_FILE   = "../data/reader_val.json"
VOCAB_FILE = "../models/word2idx.json"
MODEL_FILE = "../models/reader_bilstm_attn.pt"

MAX_Q_LEN = 40
MAX_P_LEN = 300
DEVICE = "cpu"

# ---------------- TOKENIZER ----------------
class SimpleTokenizer:
    def __init__(self, word2idx):
        self.w2i = word2idx
        self.pad = word2idx.get("<pad>", 0)
        self.unk = word2idx.get("<unk>", 1)

    def encode(self, text, max_len):
        tokens = text.lower().split()
        ids = [self.w2i.get(t, self.unk) for t in tokens][:max_len]
        mask = [1]*len(ids)
        while len(ids) < max_len:
            ids.append(self.pad)
            mask.append(0)
        return torch.tensor(ids), torch.tensor(mask)

# ---------------- METRICS ----------------
def em_f1(pred, gold):
    p = pred.lower().split()
    g = gold.lower().split()
    if not p or not g:
        return 0, 0

    common = Counter(p) & Counter(g)
    same = sum(common.values())

    if same == 0:
        return 0, 0

    prec = same / len(p)
    rec = same / len(g)
    f1 = 2 * prec * rec / (prec + rec)
    em = int(" ".join(p) == " ".join(g))
    return em, f1

# ---------------- LOAD ----------------
with open(VAL_FILE) as f:
    data = json.load(f)

with open(VOCAB_FILE) as f:
    word2idx = json.load(f)

tok = SimpleTokenizer(word2idx)

model = ReaderModel(
    vocab_size=len(word2idx),
    embed_dim=100,
    hidden=128
)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.to(DEVICE)
model.eval()

ems, f1s = [], []

# ---------------- EVAL ----------------
with torch.no_grad():
    for ex in tqdm(data):
        q, qm = tok.encode(ex["question"], MAX_Q_LEN)
        p, pm = tok.encode(ex["passage"], MAX_P_LEN)

        q, p = q.unsqueeze(0), p.unsqueeze(0)
        qm, pm = qm.unsqueeze(0), pm.unsqueeze(0)

        start, end = model(q, p, qm, pm)

        s = start.argmax(1).item()
        e = end.argmax(1).item()

        if s > e:
            pred = ""
        else:
            inv = {v:k for k,v in word2idx.items()}
            pred = " ".join(inv.get(i.item(), "") for i in p[0,s:e+1])

        em, f1 = em_f1(pred, ex["answer"])
        ems.append(em)
        f1s.append(f1)

print("\n---- READER RE-EVALUATION ----")
print(f"EM : {sum(ems)/len(ems):.4f}")
print(f"F1 : {sum(f1s)/len(f1s):.4f}")
