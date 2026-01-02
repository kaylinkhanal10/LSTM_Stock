# 📰 LSTM-Based Question Answering on News Articles

**Team Member:** Kaylin Khanal  , Saroj Dahal
**Project Type:** Neural Information Retrieval + Extractive Question Answering  
**Model Stack:** BiLSTM Retriever + LSTM Attention Reader  
**Language:** Python (PyTorch)

---

## 1. Project Overview

This project implements a **classical neural Question Answering (QA) pipeline** over real-world news articles.  
The system retrieves relevant news passages using a **BiLSTM-based dense retriever**, then extracts short answers using an **LSTM + attention-based reader**.

The implementation intentionally avoids transformer-based models and focuses on understanding **traditional neural IR pipelines**, as required by the assignment.

### System Goals
- [x] Ingest and clean hundreds of news articles  
- [x] Retrieve relevant passages efficiently  
- [x] Extract factual answers with citation  
- [x] Return **“not found”** when confidence is low (explicit requirement)

---

## 2. System Architecture

User Question  
↓  
Tokenization  
↓  
BiLSTM Retriever  
↓ (Top-K passages)  
Re-ranking  
↓ (Top-M passages)  
LSTM + Attention Reader  
↓  
Answer Span + Confidence  
↓  
Citation (URL, Title, Date)

---

## 3. Data Sources

### News Providers
- **ShareSansar** — Finance, Market, Economy  
- **Merolagani** — Finance, Economy, Market  

### Corpus Statistics

| Source      | Articles |
|-------------|----------|
| ShareSansar | 3,926    |
| Merolagani  | 1,216    |
| **Total**   | **5,142** |

Articles are merged conservatively (exact duplicates only).

---

## 4. Data Processing Pipeline (Week 1)

### Implemented Steps
- [x] Crawl & ingest articles (URL, title, date, source)
- [x] Unicode normalization & boilerplate removal
- [x] Passage chunking (200–400 tokens with overlap)
- [x] Near-duplicate removal using **MinHash + LSH**
- [x] Named Entity Recognition using **spaCy**
- [x] Metadata tagging (publisher, date, entities, section)

### Outputs
- `articles_all.json`
- `articles_train.json`
- `articles_val.json`
- `articles_test.json`
- `passages_train.json`
- `passages_trainval.json`

### Passage Statistics
- Final training passages: ~3,400  
- Deduplication reduction: ~9–10%  
- Avg passages per article: ~1.0  

---

## 5. Retriever Module (Week 2)

### Model
- BiLSTM encoder
- Static word embeddings (GloVe 100d)
- Mean pooling over hidden states
- Cosine similarity for retrieval

### Training Setup
- **Query:** Article title / headline  
- **Positive:** Passage from the same article  
- **Negatives:** Random passages from other articles  
- **Loss:** Triplet margin loss  
- **Device:** CPU (assignment-safe)

### Outputs
- `models/retriever_bilstm.pt`
- `models/word2idx.json`
- `data/retriever_triplets_train.json`
- `data/retriever_triplets_val.json`

### Retrieval Metrics

| Metric    | Value |
|----------|-------|
| Recall@20 | ~1–5% |
| MRR      | ~0.02 |

These results are expected for a BiLSTM retriever trained with static embeddings and title-based supervision.

---

## 6. Reader Module (Week 3)

### Model
- BiLSTM passage encoder
- Bi-attention between question and passage
- Start/end span prediction heads
- Extractive QA only

### Training Strategy
- [x] Distant supervision (answer string matching in passages)
- [x] Cross-entropy loss for start/end indices
- [x] Validation using EM and F1

### Best Model Performance

| Metric | Value |
|------|-------|
| EM   | 0.1177 |
| F1   | 0.2347 |

### Outputs
- `models/reader_bilstm_attn.pt`
- `data/reader_train.json`
- `data/reader_val.json`

---

## 7. Reader Pretraining with SQuAD (Extension)

The reader was additionally **pretrained on SQuAD v1.1**, as allowed by the assignment.

### SQuAD Dataset
- Training examples: ~87,500
- Validation examples: ~34,600

### Purpose
- Learn generic span-extraction behavior
- Improve answer boundary detection
- Transfer knowledge to news domain via fine-tuning

### Result
- Reader behavior became more stable
- Overall end-to-end QA performance remains limited by retriever recall

---

## 8. Inference Pipeline (Week 4)

### Inference Steps
- Encode question with retriever
- Retrieve Top-K passages (K = 20)
- Re-rank Top-M passages (M = 5)
- Apply reader to candidate passages
- Select best answer span using confidence score
- Apply fallback if confidence < threshold

### Fallback Behavior
If no confident answer is found:
- Output: **“not found”**
- Display top source citations

This behavior is explicitly required by the assignment.

---

## 9. How to Run the System

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
URL        : https://www.sharesansar.com/..
```

## 10. Repository Structure


├── data/
│   ├── articles_*.json
│   ├── passages_*.json
│   ├── retriever_triplets_*.json
│   └── reader_*.json
│
├── models/
│   ├── retriever_bilstm.pt
│   ├── reader_bilstm_attn.pt
│   ├── reader_bilstm_attn_squad.pt
│   └── word2idx.json
│
├── week1/   # Data ingestion & passage building
├── week2/   # Retriever training & FAISS evaluation
├── week3/   # Reader training & SQuAD pretraining
├── week4/   # End-to-end inference
│
└── README.md
## 11. Known Limitations
Retriever trained on titles, not natural questions

Low recall due to static embeddings

Extractive-only answers (no synthesis)

Noise from distant supervision

CPU-only training constraints

## 12. Future Improvements
Train retriever on QA-style queries

Use contextual embeddings (e.g., BERT / DPR)

Hard negative mining

Improved confidence calibration

Use large-scale QA datasets (SQuAD, NewsQA, Natural Questions)

Add numeric reasoning and normalization