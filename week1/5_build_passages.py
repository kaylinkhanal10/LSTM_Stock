import json
import os
import hashlib
from datetime import datetime, date
from datasketch import MinHash, MinHashLSH
import spacy
import re

# ================= CONFIG =================
ARTICLES_FILE = "../data/articles_train.json"

OUT_RAW   = "../data/passages_train_raw.json"
OUT_FINAL = "../data/passages_train.json"

# Pragmatic settings for short finance news
MAX_PASSAGES_PER_ARTICLE = 2
CHUNK_TARGET = 300        # soft target, not strict
OVERLAP = 60

LSH_THRESHOLD = 0.85      # conservative dedup
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
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return paras

def tokenize(text):
    return [t.text for t in nlp(text) if not t.is_space]

def extract_entities(text):
    doc = nlp(text)
    return list(set(f"{e.text} ({e.label_})" for e in doc.ents))

# ================= LOAD =================
with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
    articles = json.load(f)

print(f"Loaded articles: {len(articles)}")

# ================= BUILD RAW PASSAGES =================
raw_passages = []

for art in articles:
    text = (art.get("content") or "").strip()
    if not text:
        continue

    paras = split_paragraphs(text)
    para_tokens = [tokenize(p) for p in paras]
    total_tokens = sum(len(p) for p in para_tokens)

    chunks = []

    # ---- Case 1: short article → single passage
    if total_tokens <= CHUNK_TARGET:
        chunks.append((" ".join([t for p in para_tokens for t in p])))

    # ---- Case 2: longer article → up to 2 passages
    else:
        flat = [t for p in para_tokens for t in p]

        first = flat[:CHUNK_TARGET]
        second = flat[max(0, CHUNK_TARGET - OVERLAP): CHUNK_TARGET * 2]

        chunks.append(" ".join(first))
        if len(second) > 80:   # avoid trivial second chunk
            chunks.append(" ".join(second))

    # cap passages per article
    chunks = chunks[:MAX_PASSAGES_PER_ARTICLE]

    for idx, chunk_text in enumerate(chunks):
        raw_passages.append({
            "passage_id": passage_id(art["link"], idx),
            "article_id": article_id(art["link"]),
            "source": art.get("source"),
            "title": art.get("title"),
            "url": art.get("link"),
            "publishedISO": art.get("publishedISO"),
            "days_ago": days_ago(art.get("publishedISO")),
            "content": chunk_text,
            "entities": extract_entities(chunk_text),
            "section": "Finance/News"
        })

with open(OUT_RAW, "w", encoding="utf-8") as f:
    json.dump(raw_passages, f, indent=2, ensure_ascii=False)

print(f"Raw passages: {len(raw_passages)}")

# ================= LIGHT DEDUP =================
lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
final = []
dups = 0

for p in raw_passages:
    m = MinHash(num_perm=NUM_PERM)
    for w in set(p["content"].lower().split()):
        m.update(w.encode("utf-8"))

    if lsh.query(m):
        dups += 1
        continue

    lsh.insert(p["passage_id"], m)
    final.append(p)

with open(OUT_FINAL, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

# ================= AUDIT =================
print("\n----- PASSAGE SUMMARY -----")
print(f"Passages before dedup : {len(raw_passages)}")
print(f"Passages after dedup  : {len(final)}")
print(f"Duplicates removed    : {dups}")
print(f"Avg passages/article  : {len(final) / max(1, len(articles)):.2f}")
print(f"Saved to              : {OUT_FINAL}")
