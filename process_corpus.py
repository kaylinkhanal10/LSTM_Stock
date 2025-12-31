import json
import spacy
from datasketch import MinHash, MinHashLSH

# Load spaCy for tokenization and Entity Recognition
nlp = spacy.load("en_core_web_sm")

def process_corpus(input_file):
    with open(input_file, 'r') as f:
        articles = json.load(f)
    
    processed_passages = []
    lsh = MinHashLSH(threshold=0.9, num_perm=128) # For deduplication
    
    passage_id_counter = 0

    for art in articles:
        text = art['content']
        doc = nlp(text)
        tokens = [token.text for token in doc]
        
        # 1. CHUNKING: 300 tokens with 50 overlap
        chunk_size = 300
        overlap = 50
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = " ".join(chunk_tokens)
            
            # Stop if the chunk is too small
            if len(chunk_tokens) < 50: continue

            # 2. DEDUPLICATION (MinHash)
            m = MinHash(num_perm=128)
            for word in set(chunk_tokens):
                m.update(word.encode('utf8'))
            
            # Check if this chunk is a duplicate
            is_dup = lsh.query(m)
            if not is_dup:
                lsh.insert(f"p_{passage_id_counter}", m)
                
                # 3. METADATA LABELLING (NER)
                # Process the chunk for entities
                chunk_doc = nlp(chunk_text)
                entities = [f"{ent.text} ({ent.label_})" for ent in chunk_doc.ents]
                
                # Create the final passage object
                passage = {
                    "passage_id": passage_id_counter,
                    "article_title": art['title'],
                    "url": art['link'],
                    "date": art.get('publishedISO'),
                    "content": chunk_text,
                    "entities": list(set(entities)), # Metadata: entities
                    "section": "Finance/News"     # Metadata: section
                }
                processed_passages.append(passage)
                passage_id_counter += 1

    return processed_passages

# Save the final result
# final_data = process_corpus("sharesansar_corpus.json")