import json
import random
import os
import re

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
RAW_TRAIN = "../data/train-v1.1.json"
RAW_DEV   = "../data/squad_raw_val.json"

OUT_TRAIN = "../data/squad_train.json"
OUT_VAL   = "../data/squad_val.json"

MAX_CONTEXT_LEN = 300      # keep aligned with reader
MAX_QUESTION_LEN = 40

random.seed(42)
os.makedirs("../data", exist_ok=True)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def normalize(text):
    return re.sub(r"\s+", " ", text).strip()

def clip_text(text, max_len):
    tokens = text.split()
    return " ".join(tokens[:max_len])

# -------------------------------------------------
# LOAD RAW SQuAD
# -------------------------------------------------
def load_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["data"]

# -------------------------------------------------
# BUILD FLAT EXAMPLES
# -------------------------------------------------
def build_examples(raw_data):
    examples = []

    for article in raw_data:
        for para in article["paragraphs"]:
            context = normalize(para["context"])
            context = clip_text(context, MAX_CONTEXT_LEN)

            for qa in para["qas"]:
                if qa.get("is_impossible"):
                    continue

                question = normalize(qa["question"])
                question = clip_text(question, MAX_QUESTION_LEN)

                for ans in qa["answers"]:
                    answer = normalize(ans["text"])
                    start  = ans["answer_start"]
                    end    = start + len(answer)

                    # Safety check
                    if start < 0 or end > len(context):
                        continue

                    examples.append({
                        "question": question,
                        "passage": context,
                        "start": start,
                        "end": end,
                        "answer": answer
                    })

    return examples

# -------------------------------------------------
# RUN
# -------------------------------------------------
print("Loading raw SQuAD...")
raw_train = load_raw(RAW_TRAIN)
raw_dev   = load_raw(RAW_DEV)

print("Building flat examples...")
train_examples = build_examples(raw_train)
val_examples   = build_examples(raw_dev)

random.shuffle(train_examples)
random.shuffle(val_examples)

# -------------------------------------------------
# SAVE
# -------------------------------------------------
with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    json.dump(train_examples, f, indent=2, ensure_ascii=False)

with open(OUT_VAL, "w", encoding="utf-8") as f:
    json.dump(val_examples, f, indent=2, ensure_ascii=False)

print("\n---- SQuAD PREP SUMMARY ----")
print(f"Train examples : {len(train_examples)}")
print(f"Val examples   : {len(val_examples)}")
print(f"Saved to       : {OUT_TRAIN}, {OUT_VAL}")
