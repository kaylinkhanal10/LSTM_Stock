import json
import random
import hashlib
from collections import defaultdict

# ================= PATHS =================
PASSAGES_FILE = "../data/passages_trainval.json"   # ✅ FIXED
TRAIN_ARTS    = "../data/articles_train.json"
VAL_ARTS      = "../data/articles_val.json"

OUT_TRAIN = "../data/retriever_triplets_train.json"
OUT_VAL   = "../data/retriever_triplets_val.json"

random.seed(42)

# ================= HELPERS =================
def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def article_id(url):
    # MUST match build_passages hashing
    return hashlib.sha1(url.encode("utf-8")).hexdigest()

# ================= LOAD =================
train_articles = load(TRAIN_ARTS)
val_articles   = load(VAL_ARTS)
passages       = load(PASSAGES_FILE)

print(f"Loaded passages       : {len(passages)}")

# ================= INDEX PASSAGES =================
article_to_passages = defaultdict(list)
for p in passages:
    article_to_passages[p["article_id"]].append(p)

# Keep all articles that have ≥1 passage
usable_articles = {
    aid: ps for aid, ps in article_to_passages.items() if len(ps) >= 1
}

print(f"Articles with passages: {len(usable_articles)}")

all_passages = passages

# ================= BUILD TRIPLETS =================
def build_triplets(articles):
    triplets = []

    for art in articles:
        aid = article_id(art["link"])

        if aid not in usable_articles:
            continue

        ps = usable_articles[aid]

        # Article-centric positive
        positive = random.choice(ps)

        # Two negatives from other articles
        negatives = []
        while len(negatives) < 2:
            neg = random.choice(all_passages)
            if neg["article_id"] != aid:
                negatives.append(neg)

        triplets.append({
            "query": art["title"],
            "positive": positive["content"],
            "negative_1": negatives[0]["content"],
            "negative_2": negatives[1]["content"],
            "meta": {
                "article_id": aid,
                "source": positive["source"],
                "url": positive["url"]
            }
        })

    return triplets

# ================= RUN =================
train_triplets = build_triplets(train_articles)
val_triplets   = build_triplets(val_articles)

with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    json.dump(train_triplets, f, indent=2, ensure_ascii=False)

with open(OUT_VAL, "w", encoding="utf-8") as f:
    json.dump(val_triplets, f, indent=2, ensure_ascii=False)

# ================= AUDIT =================
print("\n---- TRIPLET SUMMARY ----")
print(f"Train triplets : {len(train_triplets)}")
print(f"Val triplets   : {len(val_triplets)}")

if train_triplets:
    print(f"Avg triplets/article (train): {len(train_triplets) / len(usable_articles):.2f}")
