# RNN-based Sentiment Analysis on IMDB Movie Reviews

A PyTorch implementation of a Recurrent Neural Network (RNN) for binary sentiment classification (positive/negative) on the IMDB movie reviews dataset, using classic NLP preprocessing and TF-IDF feature vectorization.

## 📌 Overview

This project builds an end-to-end sentiment classification pipeline:
1. Clean and preprocess 50,000 IMDB movie reviews
2. Convert text to numerical features using TF-IDF
3. Train a Recurrent Neural Network (RNN) classifier in PyTorch
4. Evaluate model performance on a held-out test set

**Test Accuracy: ~85.1%**

## 🗂️ Dataset

- **Source:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 reviews (49,582 after removing duplicates)
- **Labels:** `positive` / `negative`

> Note: `IMDB Dataset.csv` is not included in this repo due to size. Download it from the link above and place it in the project root before running the notebook.

## ⚙️ Tech Stack

- Python, Pandas, NumPy
- NLTK (tokenization, stopword removal, stemming)
- Scikit-learn (`TfidfVectorizer`, `LabelEncoder`, `train_test_split`)
- PyTorch (`nn.RNN`, `DataLoader`, `TensorDataset`)

## 🔄 Pipeline

### 1. Data Cleaning
- Removed duplicate reviews
- Lowercased text
- Removed URLs, HTML tags, and punctuation using regex

### 2. Text Preprocessing
- Tokenization and stopword removal (NLTK)
- Stemming using `PorterStemmer` (e.g., `running` → `run`)

### 3. Feature Engineering
- Label encoding for sentiment (`positive` → 1, `negative` → 0)
- TF-IDF vectorization (`max_features=5000`) to convert text into numerical feature vectors

### 4. Model Architecture
```
Input (5000-dim TF-IDF vector)
      ↓
   nn.RNN (hidden_size=128, num_layers=1)
      ↓
 Fully Connected Layer (128 → 1)
      ↓
      Sigmoid → Probability (positive/negative)
```

### 5. Training
- Loss: Binary Cross-Entropy (`BCELoss`)
- Optimizer: Adam
- Epochs: 10
- Batch size: 64

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | ~85.15% |

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy nltk scikit-learn torch
```

### Run
1. Download `IMDB Dataset.csv` and place it in the project root
2. Open and run `RNN_for_sentimentanalysis.ipynb` cell by cell (or export/run as a script)

```bash
jupyter notebook RNN_for_sentimentanalysis.ipynb
```

## 🔧 Future Improvements

- Replace TF-IDF with word embeddings (Word2Vec/GloVe) and feed actual token sequences into the RNN instead of a single-timestep TF-IDF vector, to properly leverage sequential/contextual information
- Experiment with LSTM/GRU/Bidirectional RNN architectures for better long-range dependency handling
- Add dropout and learning rate scheduling to reduce overfitting
- Hyperparameter tuning (hidden size, number of layers, batch size)
- Deploy as a simple web app (Streamlit/Flask) for live predictions

## 📁 Project Structure

```
├── RNN_for_sentimentanalysis.ipynb   # Main notebook (preprocessing, model, training, evaluation)
├── IMDB Dataset.csv                  # Dataset (not included, download separately)
└── README.md
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
