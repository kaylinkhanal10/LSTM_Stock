import json
import os
import random
from collections import defaultdict

INPUT_FILE = "data/articles_all.json"
OUT_DIR = "data"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42  # deterministic

os.makedirs(OUT_DIR, exist_ok=True)

def load_articles(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def stratified_split(articles):
    """
    Split articles stratified by source to keep publisher balance.
    """
    by_source = defaultdict(list)
    for a in articles:
        by_source[a.get("source", "unknown")].append(a)

    train, val, test = [], [], []

    for src, items in by_source.items():
        random.shuffle(items)

        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

        print(f"{src}: total={n}, train={n_train}, val={n_val}, test={n - n_train - n_val}")

    return train, val, test

def main():
    random.seed(SEED)

    articles = load_articles(INPUT_FILE)
    print(f"Total articles loaded: {len(articles)}")

    train, val, test = stratified_split(articles)

    save(os.path.join(OUT_DIR, "articles_train.json"), train)
    save(os.path.join(OUT_DIR, "articles_val.json"), val)
    save(os.path.join(OUT_DIR, "articles_test.json"), test)

    print("---- SPLIT SUMMARY ----")
    print(f"Train: {len(train)}")
    print(f"Val  : {len(val)}")
    print(f"Test : {len(test)}")

if __name__ == "__main__":
    main()
