import json
import os
import hashlib
from datetime import datetime, date
from datasketch import MinHash, MinHashLSH
import spacy
import re

# ================= CONFIG =================
ARTICLES_FILE = "../data/articles_trainval.json"

OUT_FINAL = "../data/passages_trainval.json"

MAX_PASSAGES_PER_ARTICLE = 2
CHUNK_TARGET = 300
OVERLAP = 60

LSH_THRESHOLD = 0.85
NUM_PERM = 128

os.makedirs("data", exist_ok=True)

nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])

# ================= HELPERS =================
def sha1(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def article_id(url):
    return sha1(url)

def passage_id(url, idx):
    return sha1(f"{url}:{idx}")

def days_ago(iso):
    try:
        d = datetime.strptime(iso, "%Y-%m-%d").date()
        return (date.today() - d).days
    except:
        return None

def split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]

def tokenize(text):
    return [t.text for t in nlp(text) if not t.is_space]

def extract_entities(text):
    return list(set(f"{e.text} ({e.label_})" for e in nlp(text).ents))

# ================= LOAD =================
with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
    articles = json.load(f)

print(f"Loaded articles (train+val): {len(articles)}")

# ================= BUILD PASSAGES =================
raw = []

for art in articles:
    text = (art.get("content") or "").strip()
    if not text:
        continue

    paras = split_paragraphs(text)
    tokens = [t for p in paras for t in tokenize(p)]

    chunks = []

    if len(tokens) <= CHUNK_TARGET:
        chunks.append(" ".join(tokens))
    else:
        first = tokens[:CHUNK_TARGET]
        second = tokens[max(0, CHUNK_TARGET - OVERLAP): CHUNK_TARGET * 2]
        chunks.append(" ".join(first))
        if len(second) > 80:
            chunks.append(" ".join(second))

    for i, chunk in enumerate(chunks[:MAX_PASSAGES_PER_ARTICLE]):
        raw.append({
            "passage_id": passage_id(art["link"], i),
            "article_id": article_id(art["link"]),
            "source": art["source"],
            "title": art["title"],
            "url": art["link"],
            "publishedISO": art.get("publishedISO"),
            "days_ago": days_ago(art.get("publishedISO")),
            "content": chunk,
            "entities": extract_entities(chunk),
            "section": "Finance/News"
        })

print(f"Raw passages: {len(raw)}")

# ================= DEDUP =================
lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
final = []

for p in raw:
    m = MinHash(num_perm=NUM_PERM)
    for w in set(p["content"].lower().split()):
        m.update(w.encode("utf-8"))

    if lsh.query(m):
        continue

    lsh.insert(p["passage_id"], m)
    final.append(p)

with open(OUT_FINAL, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

print(f"Final passages (train+val): {len(final)}")
print(f"Saved to {OUT_FINAL}")
