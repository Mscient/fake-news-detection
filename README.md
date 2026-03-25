# 📰 Fake News Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-green)

A machine learning project that detects whether a news article is **Real or Fake** using Natural Language Processing and classical ML algorithms. Built as a minor project during my internship.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [ML Pipeline](#ml-pipeline)
- [Model Results](#model-results)
- [How to Run](#how-to-run)
- [Web App](#web-app)
- [Author](#author)

---

## 🔍 Overview

Fake news is a growing problem in today's digital world. This project builds a text classification system that can automatically identify whether a given news article is real or fake using TF-IDF vectorization and multiple machine learning classifiers.

---

## 📂 Dataset

**ISOT Fake News Dataset**
- Source: University of Victoria, Canada
- Total articles: ~44,000
- Real news: ~21,417 articles
- Fake news: ~23,481 articles
- Download: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## 🗂️ Project Structure

```
fake-news-detection/
├── data/
│   ├── raw/                  # Original True.csv and Fake.csv
│   └── processed_data.csv    # Cleaned and merged dataset
├── notebooks/
│   └── 01_fake_news_detection.ipynb   # Full EDA + ML pipeline
├── src/
│   ├── preprocess.py         # Text cleaning functions
│   ├── train.py              # Model training script
│   └── predict.py            # Prediction script
├── app/
│   └── streamlit_app.py      # Web application
├── models/
│   ├── model.pkl             # Trained ML model
│   └── tfidf.pkl             # TF-IDF vectorizer
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Category        | Tools                              |
|-----------------|------------------------------------|
| Language        | Python 3.8+                        |
| Data Handling   | Pandas, NumPy                      |
| NLP             | NLTK, re, string                   |
| ML Models       | Scikit-learn                       |
| Visualization   | Matplotlib, Seaborn                |
| Web App         | Streamlit                          |
| Model Saving    | Joblib                             |
| Version Control | Git, GitHub                        |

---

## ⚙️ ML Pipeline

```
Raw Data → Text Cleaning → TF-IDF Vectorization → Model Training → Evaluation → Deployment
```

### Text Preprocessing steps:
- Lowercasing
- Removing URLs, HTML tags, punctuation
- Removing stopwords
- Lemmatization using NLTK WordNetLemmatizer

### Feature Engineering:
- TF-IDF Vectorizer with 50,000 features
- Bigrams (ngram_range = 1,2)
- Combined title + article text

---

## 📊 Model Results

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | ~98.5%   |
| Linear SVM          | ~98.3%   |
| Random Forest       | ~99.0%   |
| Naive Bayes         | ~95.2%   |

**Best Model: Logistic Regression**
- Accuracy : 98.5%
- ROC-AUC  : 0.99
- F1 Score : 0.98

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Mscient/Fake-news-Detection.git
cd Fake-news-Detection
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download `True.csv` and `Fake.csv` from Kaggle and place them inside `data/raw/`

### 5. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/01_fake_news_detection.ipynb
```

### 6. Run the Web App
```bash
cd app
streamlit run streamlit_app.py
```

---

## 🌐 Web App

The Streamlit web app allows users to paste any news article and get an instant prediction of whether it is **Real** or **Fake**.

**Features:**
- Clean and simple UI
- Instant prediction
- Works on any news article text

---

## 👨‍💻 Author

**Prash** (Mscient)
- GitHub: https://github.com/Mscient
- Project: Internship Minor Project

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
