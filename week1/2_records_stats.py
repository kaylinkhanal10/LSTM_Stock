import json

def count(path):
    with open(path, "r", encoding="utf-8") as f:
        return len(json.load(f))

sharesansar = count("sharesansar_corpus.json")
merolagani  = count("merolagani_corpus.json")

print(f"Sharesansar articles : {sharesansar}")
print(f"Merolagani articles  : {merolagani}")
print(f"TOTAL articles       : {sharesansar + merolagani}")
