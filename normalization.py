import json
import re
import spacy
from datasketch import MinHash, MinHashLSH

# Load spaCy for tokenization and Entity Recognition (Metadata requirement)
nlp = spacy.load("en_core_web_sm", disable=["parser"])

def normalize_text(text):
    """Normalize text: fix Unicode, remove boilerplate, preserve paragraph boundaries."""
    # Fix Unicode (smart quotes, etc.)
    text = text.encode("ascii", "ignore").decode("utf-8")
    # Fix extra whitespace but keep paragraph boundaries (\n\n)
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove any non-essential characters but keep punctuation
    return text.strip()

def chunk_text(doc, chunk_size=300, overlap=50):
    """Chunk into 200-400 token passages with overlap."""
    tokens = [token for token in doc]
    passages = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i : i + chunk_size]
        if len(chunk) < 50: # Ignore very small trailing fragments
            continue
        passages.append(chunk)
    return passages

def process_corpus(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    processed_passages = []
    lsh = MinHashLSH(threshold=0.9, num_perm=128) # Deduplication threshold
    passage_id = 0
    duplicate_count = 0

    print("Processing articles...")
    for art in articles:
        # 1. Normalization
        clean_content = normalize_text(art['content'])
        doc = nlp(clean_content)
        
        # 2. Chunking
        chunks = chunk_text(doc)

        for chunk in chunks:
            chunk_text_str = "".join([t.text_with_ws for t in chunk])
            
            # 3. Deduplication (MinHash)
            m = MinHash(num_perm=128)
            for token in set([t.text.lower() for t in chunk]):
                m.update(token.encode('utf8'))
            
            # Check if nearly identical passage exists
            is_dup = lsh.query(m)
            if is_dup:
                duplicate_count += 1
                continue
            
            lsh.insert(f"p_{passage_id}", m)

            # 4. Metadata Labelling (Entities)
            # Tagging organizations and money for metadata requirement
            entities = [f"{ent.text} ({ent.label_})" for ent in doc.ents if ent.label_ in ["ORG", "MONEY", "GPE"]]

            processed_passages.append({
                "id": passage_id,
                "content": chunk_text_str,
                "metadata": {
                    "source": art['source'],
                    "url": art['link'],
                    "date": art.get('publishedISO'),
                    "headline": art['title'],
                    "entities": list(set(entities))[:10] # Top 10 unique entities
                }
            })
            passage_id += 1

    # 5. Save Final Result
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_passages, f, indent=4)

    print(f"Finished!")
    print(f"Total Passages Created: {len(processed_passages)}")
    print(f"Duplicates Removed: {duplicate_count}")
    print(f"Deduplication Rate: {(duplicate_count/(len(processed_passages)+duplicate_count))*100:.2f}%")

if __name__ == "__main__":
    process_corpus("sharesansar_corpus.json", "processed_passages.json")