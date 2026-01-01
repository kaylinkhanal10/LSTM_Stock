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



---

## 2. System Architecture

