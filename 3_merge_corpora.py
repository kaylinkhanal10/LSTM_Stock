import json
import os
import hashlib
from urllib.parse import urlparse, parse_qs

SHARESANSAR_FILE = "sharesansar_corpus.json"
MEROLAGANI_FILE = "merolagani_corpus.json"

OUT_DIR = "data"
OUT_FILE = os.path.join(OUT_DIR, "articles_all.json")
os.makedirs(OUT_DIR, exist_ok=True)

def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def article_uid(article):
    """
    Conservative article identity.
    Removes only exact duplicates, not similar stories.
    """
    src = article.get("source", "")
    link = article.get("link", "")

    if "merolagani" in src.lower():
        parsed = urlparse(link)
        q = parse_qs(parsed.query)
        if "newsID" in q:
            return f"merolagani:{q['newsID'][0]}"

    if "sharesansar" in src.lower():
        parsed = urlparse(link)
        return f"sharesansar:{parsed.path}"

    # fallback: hash source+link
    raw = f"{src}|{link}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def merge():
    ss = load_json(SHARESANSAR_FILE)
    ml = load_json(MEROLAGANI_FILE)

    print(f"Loaded {len(ss)} ShareSansar articles")
    print(f"Loaded {len(ml)} Merolagani articles")

    merged = []
    seen_ids = set()
    dropped = 0

    for art in ss + ml:
        uid = article_uid(art)

        if uid in seen_ids:
            dropped += 1
            continue

        seen_ids.add(uid)

        merged.append({
            "title": art.get("title"),
            "link": art.get("link"),
            "source": art.get("source"),
            "publishedText": art.get("publishedText"),
            "publishedISO": art.get("publishedISO"),
            "publishedTimestamp": art.get("publishedTimestamp"),
            "content": art.get("content"),
        })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print("---- MERGE SUMMARY ----")
    print(f"Final merged articles : {len(merged)}")
    print(f"Dropped exact dups    : {dropped}")
    print(f"Saved to              : {OUT_FILE}")

if __name__ == "__main__":
    merge()
