# 📰 Fake News Detection using Bidirectional LSTM

## 🚀 Run This Project in Google Colab
You can run this project directly in Google Colab with GPU support — no setup required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rithwik-sayini/Fake-News-Detection-Using-LSTM/blob/main/Fake_News_Detection_LSTM.ipynb)

---

## 📖 Overview
This project implements a **Fake News Detection System** using **Deep Learning (Bidirectional LSTM)** and **Natural Language Processing (NLP)**.  
It automatically classifies news articles as **True** or **Fake** based on their textual content.

By analyzing writing patterns, sentence structures, and semantics, the model helps identify misinformation — a step toward automated fact verification.

---

## 🎯 Problem Statement
Fake news spreads quickly through online platforms, misleading the public and affecting decision-making.  
Manual verification is slow and unscalable.  
This project aims to build an **AI-based text classifier** to detect fake news automatically.

---

## 🧩 Workflow

### 1️⃣ Data Loading
- Datasets used:
  - **True.csv** → Verified real news  
  - **Fake.csv** → Fabricated or misleading news  
- Labels:
  - `0` = True  
  - `1` = Fake  

### 2️⃣ Data Preprocessing
- Combine and shuffle datasets  
- Split into train and test sets (80/20)  
- Tokenize text using **Keras Tokenizer**  
- Pad sequences to a fixed length (300 tokens)

### 3️⃣ Model Architecture
