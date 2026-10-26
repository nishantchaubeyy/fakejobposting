# 🛡️ TruthGuard AI: Fake Product Review & Fraudulent Job Detection System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Mini Project Title:** Fake Product Review Detection Using Natural Language Processing and Machine Learning  
> **Repository:** [https://github.com/nishantchaubeyy/fakejobposting.git](https://github.com/nishantchaubeyy/fakejobposting.git)

---

## 📌 1. Project Overview & Problem Statement

Online e-commerce and employment platforms rely heavily on user-generated feedback and postings. However, deceptive reviews and fraudulent listings distort buyer perception, manipulate algorithmic search rankings, and damage user trust.

- **Fake Product Reviews:** Shills and automated bots publish deceptive 5-star ratings or competitor smear campaigns to artificially inflate or deflate product sales.
- **Fraudulent Job Postings:** Scammers post work-from-home or data-entry scams that ask for registration fees or wire transfers.

This project delivers a high-accuracy, production-ready machine learning system using Natural Language Processing (NLP) to classify reviews and job postings in real time as **Fake/Deceptive** or **Genuine/Truthful**.

---

## 🎯 2. Objectives

- **NLP Preprocessing Pipeline:** Tokenization, lowercase normalization, punctuation stripping, and stop-word filtering.
- **Feature Extraction:** High-dimensional **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization with unigrams and bigrams (`ngram_range=(1, 2)`).
- **Multi-Model Comparison:** Train and benchmark **Logistic Regression**, **Multinomial Naive Bayes**, and **Calibrated Linear Support Vector Machines (LinearSVC)**.
- **Evaluation:** Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC Curves.
- **Interactive Web Application:** Built with **Streamlit** featuring real-time inference, sample benchmark loaders, metadata impact analysis (e.g. unverified purchase signals), CSV batch processing, and Exploratory Data Analysis (EDA).

---

## 📂 3. Dataset Information

- **Dataset:** Amazon Product Reviews Dataset for Deception Detection
- **Source:** [GitHub - aayush210789/Deception-Detection-on-Amazon-reviews-dataset](https://github.com/aayush210789/Deception-Detection-on-Amazon-reviews-dataset)
- **Record Count:** 21,000 labeled reviews (10,500 Fake, 10,500 Genuine) across 30 product categories.
- **Target Variable:** `LABEL` (`__label1__` = Deceptive/Fake, `__label2__` = Genuine/Truthful).
- **Key Features:** `REVIEW_TEXT`, `REVIEW_TITLE`, `RATING`, `VERIFIED_PURCHASE`, `PRODUCT_CATEGORY`.

---

## 📊 4. Model Benchmark & Evaluation Results

Trained on **21,000 Amazon reviews** with an 80/20 stratified train-test split:

| Model Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **65.52%** | **65.54%** | **65.48%** | **65.51%** |
| **Multinomial Naive Bayes** | **64.57%** | **64.87%** | **63.57%** | **64.21%** |

> **Conclusion:** Logistic Regression outperforms Multinomial Naive Bayes on all four evaluation metrics on this balanced binary classification task.



## 🏗️ 5. System Architecture

```text
       ┌──────────────────────────────┐
       │   Customer Review Text /     │
       │   Job Posting Description    │
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │     Text Preprocessing       │
       │ (Lowercase, Regex, Cleaning) │
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │    TF-IDF Vectorization      │
       │  (1-2 N-grams, 10,000 feats) │
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │  Machine Learning Classifier │
       │  • Logistic Regression       │
       │  • Multinomial Naive Bayes   │
       │  • Linear SVM                │
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │ Streamlit Interactive UI     │
       │  • Deception Gauge & Badge   │
       │  • Highlighted Trigger Words │
       │  • Batch CSV Processing      │
       │  • Exploratory Visualizations│
       └──────────────────────────────┘
```

---

## 🚀 6. Installation & Local Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/nishantchaubeyy/fakejobposting.git
cd fakejobposting
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train Models (Optional, pre-generated artifact included)
```bash
python train_model.py
```

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 7. How to Deploy to Streamlit Cloud (Free & Instant)

1. Ensure the repository has been pushed to GitHub:
   ```bash
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **"New app"**.
4. Configure deployment settings:
   - **Repository:** `nishantchaubeyy/fakejobposting`
   - **Branch:** `main` (or `master`)
   - **Main file path:** `app.py`
5. Click **"Deploy!"** — Streamlit automatically installs dependencies from `requirements.txt` and hosts the live application.

---

## 📂 Repository Structure

```text
fakejobposting/
├── app.py                   # Main Streamlit web application
├── train_model.py           # NLP training & evaluation pipeline
├── model_pipeline.pkl       # Trained review model artifact & metrics
├── job_model_pipeline.pkl   # Trained job scam model artifact
├── requirements.txt         # Production Python dependencies
├── .gitignore               # Ignored files (caches, virtualenvs)
└── README.md                # Academic & technical documentation
```

---

## 👨‍💻 Author & Maintainer
- **Nishant Kumar Chaubey**
- GitHub: [@nishantchaubeyy](https://github.com/nishantchaubeyy)
- Email: `nishantchaubey.kr@gmail.com`
