import json

with open("../data/articles_train.json", "r", encoding="utf-8") as f:
    train = json.load(f)

with open("../data/articles_val.json", "r", encoding="utf-8") as f:
    val = json.load(f)

merged = train + val

with open("../data/articles_trainval.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"Train articles : {len(train)}")
print(f"Val articles   : {len(val)}")
print(f"Combined total : {len(merged)}")
print("Saved to ../data/articles_trainval.json")
