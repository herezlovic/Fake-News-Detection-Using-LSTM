## 🚀 Run This Project in Google Colab
You can run this project directly in Google Colab with GPU support — no setup required.

[![Open In Colab](https://colab.research.google.com/assets/colab-# 📰 Fake News Detection using Bidirectional LSTM

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
badge.svg)](https://colab.research.google.com/github/rithwik-sayini/Fake-News-Detection-Using-LSTM/blob/main/Fake_News_Detection_LSTM.ipynb)

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
Input → Embedding(20000, 128)
→ Bidirectional LSTM(64)
→ Dropout(0.5)
→ Dense(64, ReLU)
→ Dropout(0.3)
→ Dense(1, Sigmoid)

markdown
Copy code

| Layer | Purpose |
|--------|----------|
| **Embedding** | Converts words to numeric word vectors |
| **Bidirectional LSTM** | Learns context in both directions |
| **Dropout** | Prevents overfitting |
| **Dense Layers** | Perform classification |
| **Sigmoid Output** | Predicts True (0) or Fake (1) |

---

## ⚙️ Model Compilation & Training
- **Loss:** Binary Cross Entropy  
- **Optimizer:** Adam (`learning_rate = 1e-3`)  
- **Metric:** Accuracy  
- **Epochs:** 3  
- **Batch Size:** 128  
- **Validation Split:** 0.2  

Example:
```python
history = model.fit(
    X_train_pad, y_train,
    validation_split=0.2,
    epochs=3,
    batch_size=128,
    verbose=1
)
📊 Evaluation
Model tested on unseen test data (20%)

Typical accuracy: ~90%

Example output:

yaml
Copy code
✅ Test Accuracy: 0.9023
Visualization
python
Copy code
plt.scatter(range(len(y_test)), y_pred_probs, c=y_test, cmap='coolwarm')
plt.title('Predicted Probability of Being Fake')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Probability (Fake)')
plt.show()
🧾 User Input Testing
You can test any news article manually after training:

python
Copy code
user_input = input("Enter a news article: ")
seq = tokenizer.texts_to_sequences([user_input])
pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
prediction = model.predict(pad)[0][0]

if prediction < 0.5:
    print("📰 The news is likely TRUE.")
else:
    print("⚠️ The news is likely FAKE.")
Example:

vbnet
Copy code
Enter a news article: Government approves new renewable energy plan
📰 The news is likely TRUE.
📂 Dataset Access
The dataset is stored in Google Drive for easy Colab integration.

To use it:

Mount your Google Drive in Colab:

python
Copy code
from google.colab import drive
drive.mount('/content/drive')
Ensure your dataset is located at:

swift
Copy code
/MyDrive/NewsDataset (1)/
Files required:

True.csv

Fake.csv

If your folder name differs, update:

python
Copy code
true_path = '/content/drive/MyDrive/<your-folder>/True.csv'
fake_path = '/content/drive/MyDrive/<your-folder>/Fake.csv'
🧠 Technologies Used
Library	Purpose
Python	Core programming language
TensorFlow / Keras	Building and training the LSTM model
Pandas / NumPy	Data preprocessing
Scikit-learn	Train-test splitting and metrics
Matplotlib	Visualization

📈 Results
Achieved ~90% accuracy

LSTM effectively captures word order and context

Outperforms classical ML models (Naive Bayes, Logistic Regression)

🚀 Future Enhancements
Integrate pre-trained embeddings (GloVe / Word2Vec)

Add Attention Mechanism or Transformer models (BERT)

Build a web app (Streamlit / Flask) for real-time testing

Add Explainable AI (LIME / SHAP) for interpretability

🧑‍💻 Author
Rithwik Rohan Sayini
B.Tech – Computer Science and Engineering
📧 rithwiksayini2004@gmail.com
🔗 LinkedIn
