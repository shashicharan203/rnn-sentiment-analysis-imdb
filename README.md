# 🎬 RNN-Based Sentiment Analysis on IMDB Movie Reviews

A Deep Learning-based sentiment classification system that predicts whether an IMDB movie review expresses a **Positive** or **Negative** sentiment.

The project implements a complete NLP pipeline including text preprocessing, TF-IDF feature extraction, PyTorch DataLoaders, and a custom Recurrent Neural Network (RNN) for binary sentiment classification.

**✅ Test Accuracy: ~85.1%**

---

## 🚀 Project Overview

Sentiment Analysis is a Natural Language Processing (NLP) task used to identify the emotional tone of textual data.

In this project, an RNN model is trained on the **IMDB Movie Reviews Dataset** to classify movie reviews into two categories:

* 😊 Positive
* 😞 Negative

---

## 📊 Dataset

The project uses the **IMDB Dataset of 50K Movie Reviews** available on Kaggle.

**Dataset:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

The dataset contains:

| Feature | Description |
|---|---|
| Dataset Size | 50,000 Reviews |
| Positive Reviews | 25,000 |
| Negative Reviews | 25,000 |
| Input Feature | Movie Review |
| Target | Sentiment |

After removing duplicate reviews, **49,582 unique samples** are used in the preprocessing and training pipeline.

> **Note:** The dataset is not included in this repository. Download it from Kaggle and place `IMDB Dataset.csv` in the project directory before running the notebook.

---

## 🧹 Text Preprocessing

The raw movie reviews are cleaned using multiple Natural Language Processing techniques:

* Converting text to lowercase
* Removing URLs
* Removing punctuation and special characters
* Removing HTML tags
* Tokenizing text using NLTK
* Removing English stopwords
* Applying Porter Stemming
* Encoding sentiment labels using `LabelEncoder`

---

## 🔢 Feature Extraction

The processed movie reviews are converted into numerical features using **TF-IDF Vectorization**.

```python
TfidfVectorizer(max_features=5000)
```

The model uses the **5,000 most relevant textual features** extracted from the movie reviews.

---

## 🧠 RNN Model Architecture

A custom Recurrent Neural Network is implemented using PyTorch.

```text
TF-IDF Input Features
        ↓
Vanilla RNN Layer
Hidden Size = 128
        ↓
Fully Connected Layer
        ↓
Sigmoid Activation
        ↓
Positive / Negative Sentiment
```

### Model Configuration

| Parameter | Value |
|---|---|
| Model | Vanilla RNN |
| Input Features | 5000 |
| Hidden Size | 128 |
| Number of RNN Layers | 1 |
| Output Size | 1 |
| Activation | Sigmoid |
| Loss Function | Binary Cross Entropy |
| Optimizer | Adam |

---

## ⚙️ Training Pipeline

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

PyTorch `TensorDataset` and `DataLoader` are used for efficient batch processing.

### Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Batch Size | 64 |
| Optimizer | Adam |
| Loss Function | BCELoss |
| Decision Threshold | 0.5 |

During training, the model performs:

1. Forward propagation
2. Sigmoid probability calculation
3. Binary Cross Entropy loss computation
4. Backpropagation
5. Model parameter updates using Adam

---

## 📈 Model Evaluation

The trained model is evaluated on unseen test data.

Predictions are generated using a probability threshold of `0.5`.

```python
predicted = (
    torch.sigmoid(outputs.squeeze()) > 0.5
).float()
```

The final model performance is measured using **classification accuracy**.

### Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **~85.15%** |

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Natural Language Processing
* Deep Learning

---

## 📁 Project Structure

```text
RNN-Sentiment-Analysis/
│
├── RNN_for_sentimentanalysis.ipynb
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd RNN-Sentiment-Analysis
```

### 3. Install Dependencies

```bash
pip install pandas numpy scikit-learn nltk torch jupyter
```

### 4. Download the Dataset

Download the **IMDB Dataset of 50K Movie Reviews** from Kaggle.

Place the downloaded file in the project directory:

```text
RNN-Sentiment-Analysis/
│
├── RNN_for_sentimentanalysis.ipynb
├── IMDB Dataset.csv
└── README.md
```

### 5. Start Jupyter Notebook

```bash
jupyter notebook
```

Open and run:

```text
RNN_for_sentimentanalysis.ipynb
```

---

## 🔮 Future Improvements

* Replace Vanilla RNN with LSTM or GRU
* Use Word2Vec or GloVe word embeddings
* Implement PyTorch Embedding layers
* Add Precision, Recall and F1-Score metrics
* Add confusion matrix visualization
* Build a real-time sentiment prediction interface
* Deploy the model using Flask or FastAPI

---

## 👤 Author

**Kunku Shashi Charan**
Machine Learning | Deep Learning | NLP | PyTorch | Python | Data Structures & Algorithms | Aspiring AI/ML Engineer
