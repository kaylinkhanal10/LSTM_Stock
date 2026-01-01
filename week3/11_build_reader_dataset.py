import json
import re
import os
import hashlib
from collections import defaultdict
from tqdm import tqdm

# ---------------- CONFIG ----------------
ART_TRAIN = "../data/articles_train.json"
ART_VAL   = "../data/articles_val.json"
PASSAGES  = "../data/passages_trainval.json"

OUT_TRAIN = "../data/reader_train.json"
OUT_VAL   = "../data/reader_val.json"

MIN_ANSWER_LEN = 3
MAX_EXAMPLES_PER_ARTICLE = 3

os.makedirs("../data", exist_ok=True)

# ---------------- HELPERS ----------------
def normalize(text):
    return re.sub(r"\s+", " ", text).strip()

def first_sentence(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return parts[0] if parts else ""

def article_id(url):
    return hashlib.sha1(url.encode("utf-8")).hexdigest()

def find_answer_span(answer, passage):
    idx = passage.lower().find(answer.lower())
    if idx == -1:
        return None
    return idx, idx + len(answer)

# ---------------- LOAD ----------------
with open(PASSAGES, "r", encoding="utf-8") as f:
    passages = json.load(f)

with open(ART_TRAIN, "r", encoding="utf-8") as f:
    train_articles = json.load(f)

with open(ART_VAL, "r", encoding="utf-8") as f:
    val_articles = json.load(f)

print(f"Loaded passages : {len(passages)}")
print(f"Loaded train articles : {len(train_articles)}")
print(f"Loaded val articles   : {len(val_articles)}")

# Index passages by article_id
article_passages = defaultdict(list)
for p in passages:
    article_passages[p["article_id"]].append(p)

# ---------------- BUILD DATASET ----------------
def build_dataset(articles):
    dataset = []

    for art in tqdm(articles):
        aid = article_id(art["link"])
        art_passages = article_passages.get(aid, [])
        if not art_passages:
            continue

        questions = [
            normalize(art["title"]),
            normalize(first_sentence(art["content"]))
        ]

        candidates = re.findall(
            r"\b[A-Z][a-zA-Z]{2,}|\b\d+[,\d]*\b",
            art["content"]
        )

        added = 0

        for q in questions:
            for ans in candidates:
                if len(ans) < MIN_ANSWER_LEN:
                    continue

                for p in art_passages:
                    span = find_answer_span(ans, p["content"])
                    if not span:
                        continue

                    start, end = span
                    dataset.append({
                        "question": q,
                        "passage": p["content"],
                        "start": start,
                        "end": end,
                        "answer": p["content"][start:end],
                        "article_id": p["article_id"],
                        "url": p["url"],
                        "date": p["publishedISO"]
                    })

                    added += 1
                    if added >= MAX_EXAMPLES_PER_ARTICLE:
                        break
                if added >= MAX_EXAMPLES_PER_ARTICLE:
                    break
            if added >= MAX_EXAMPLES_PER_ARTICLE:
                break

    return dataset

# ---------------- RUN ----------------
train_data = build_dataset(train_articles)
val_data   = build_dataset(val_articles)

with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open(OUT_VAL, "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=2, ensure_ascii=False)

print("\n---- READER DATASET SUMMARY ----")
print(f"Train examples : {len(train_data)}")
print(f"Val examples   : {len(val_data)}")
print(f"Saved to       : {OUT_TRAIN}, {OUT_VAL}")
