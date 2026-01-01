# 📰 LSTM-Based Question Answering on News Articles

**Team Name:** Kaylin Khanal  
**Project Type:** Neural Information Retrieval + Extractive Question Answering  
**Model Stack:** BiLSTM Retriever + LSTM Attention Reader  
**Language:** Python (PyTorch)

---

## 1. Project Overview

This project implements a **classical neural Question Answering (QA) pipeline** over real-world news articles.  
The system retrieves relevant news passages using a **BiLSTM-based dense retriever**, then extracts short answers using an **LSTM + attention-based reader**.

### System Goals
- [x] Ingest and clean hundreds of news articles  
- [x] Retrieve relevant passages efficiently  
- [x] Extract factual answers with citation  
- [x] Return **“not found”** when confidence is low (explicit requirement)

> ⚠️ Transformer-based models are intentionally avoided to align with course constraints and emphasize traditional neural IR pipelines.

---

## 2. System Architecture

User Question
↓
[ Tokenization ]
↓
[ BiLSTM Retriever ]
↓ (Top-K passages)
[ Re-ranking ]
↓ (Top-M passages)
[ LSTM + Attention Reader ]
↓
Answer Span + Confidence
↓
Citation (URL, Title, Date)

markdown
Copy code

---

## 3. Data Sources

### News Providers
- **ShareSansar** — Finance, Market, Economy  
- **Merolagani** — Finance, Economy, Market  

### Corpus Statistics

| Source        | Articles |
|---------------|----------|
| ShareSansar   | 3,926    |
| Merolagani    | 1,216    |
| **Total**     | **5,142** |

✔ Articles are merged conservatively (exact duplicates only).

---

## 4. Data Processing Pipeline (Week 1)

### Implemented Steps
- [x] Crawl & ingest articles (URL, title, date, source)
- [x] Unicode normalization & boilerplate removal
- [x] Passage chunking (200–400 tokens, 50 overlap)
- [x] Near-duplicate removal using **MinHash + LSH**
- [x] Metadata tagging (publisher, date, entities, section)

### Outputs
- `articles_all.json`
- `articles_train.json`
- `articles_val.json`
- `articles_test.json`
- `passages_train.json`
- `passages_trainval.json`

### Passage Statistics
- **Final training passages:** ~3,400  
- **Deduplication reduction:** ~9–10%  
- **Avg passages/article:** ~1.0  

---

## 5. Retriever Module (Week 2)

### Model
- BiLSTM Encoder
- Static embeddings (GloVe 100d)
- Mean pooling over hidden states
- Cosine similarity for retrieval

### Training Setup
- **Query:** Article title / headline  
- **Positive:** Passage from same article  
- **Negatives:** Random passages from other articles  
- **Loss:** Triplet loss  
- **Device:** CPU (assignment-safe)

### Outputs
- `models/retriever_bilstm.pt`
- `models/word2idx.json`
- `data/retriever_triplets_train.json`
- `data/retriever_triplets_val.json`

### Retrieval Metrics

| Metric     | Value |
|------------|-------|
| Recall@20  | ~0.05 |
| MRR        | ~0.02 |

> These results are expected for BiLSTM retrievers using static embeddings.

---

## 6. Reader Module (Week 3)

### Model
- BiLSTM passage encoder
- Bi-attention between question and passage
- Start/end span prediction
- Extractive QA only

### Training
- [x] Distant supervision (string match in passages)
- [x] Cross-entropy loss
- [x] Early stopping based on validation F1

### Best Model (Epoch 1)

| Metric | Value |
|-------|-------|
| EM    | 0.1177 |
| F1    | 0.2347 |

### Outputs
- `models/reader_bilstm_attn.pt`
- `data/reader_train.json`
- `data/reader_val.json`

---

## 7. Inference Pipeline (Week 4)

### Inference Steps
- [x] Encode question with retriever
- [x] Retrieve Top-K passages (K = 20)
- [x] Re-rank Top-M passages (M = 5)
- [x] Apply reader on candidates
- [x] Select best answer span by confidence
- [x] Apply fallback if confidence < threshold

### Fallback Behavior
If no confident answer is found:
- Output: **“not found”**
- Display top source citations

✔ This behavior is explicitly required by the assignment.

---

## 8. How to Run the System

### Setup
```bash
pip install -r requirements.txt
Run Inference
bash
Copy code
python week4/14_infer_qa.py --question "What dividend did Agricultural Development Bank announce?"
Example Output
yaml
Copy code
Answer     : 13 % dividend
Confidence : 0.41
Source     : sharesansar
Date       : 2025-11-06
URL        : https://www.sharesansar.com/...
9. Repository Structure
pgsql
Copy code
.
├── data/
│   ├── articles_*.json
│   ├── passages_*.json
│   ├── retriever_triplets_*.json
│   └── reader_*.json
│
├── models/
│   ├── retriever_bilstm.pt
│   ├── reader_bilstm_attn.pt
│   └── word2idx.json
│
├── week1/   # Data ingestion & passage building
├── week2/   # Retriever training & FAISS evaluation
├── week3/   # Reader training
├── week4/   # End-to-end inference
│
└── README.md

