import json
import numpy as np
import keras
from keras.layers import TextVectorization

# 1. Load your processed passages
with open('processed_passages.json', 'r', encoding='utf-8') as f:
    passages = json.load(f)

# 2. Extract content for the vectorizer
all_text = [p['content'] for p in passages]

# 3. Create the Vectorizer (Modern replacement for Tokenizer)
# - output_sequence_length=300: Automatically pads/truncates to 300 tokens
# - output_mode='int': Converts words to integers
vectorizer = TextVectorization(
    max_tokens=None, 
    standardize='lower_and_strip_punctuation',
    output_mode='int', 
    output_sequence_length=300
)

# 4. Adapt the vectorizer to your news data (Builds the vocabulary)
print("Building vocabulary from news corpus...")
vectorizer.adapt(np.array(all_text))

# 5. Extract and Save Vocabulary
# Keras 3 vocab: Index 0 is '', Index 1 is '[UNK]'
vocab = vectorizer.get_vocabulary()
word_index = {word: i for i, word in enumerate(vocab)}

with open('vocabulary.json', 'w', encoding='utf-8') as f:
    json.dump(word_index, f, indent=4)

# 6. Convert text to padded integer sequences
print("Converting passages to sequences...")
padded_passages = vectorizer(np.array(all_text)).numpy()

# 7. Save the sequences for model training
np.save('passage_sequences.npy', padded_passages)

print("-" * 30)
print(f"Success!")
print(f"Vocabulary Size: {len(vocab)}")
print(f"Padded Passages Shape: {padded_passages.shape}")
print("-" * 30)