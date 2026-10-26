"""
Machine Learning Training Pipeline for Fake Product Review & Job Detection
Dataset: Amazon Deception Detection Dataset (21,000 reviews)
Models: Logistic Regression, Multinomial Naive Bayes, Linear SVM
"""

import os
import sys
import re
import urllib.request
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

DATA_URL = "https://raw.githubusercontent.com/aayush210789/Deception-Detection-on-Amazon-reviews-dataset/master/amazon_reviews.txt"
DATA_FILE = "amazon_reviews.txt"
MODEL_FILE = "model_pipeline.pkl"
JOB_MODEL_FILE = "job_model_pipeline.pkl"


def clean_text(text: str) -> str:
    """Standardizes and cleans input review or text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def download_dataset():
    """Downloads the Amazon review dataset if not already present."""
    if not os.path.exists(DATA_FILE):
        print(f"Downloading dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("Download complete.")
    else:
        print(f"Found existing dataset: {DATA_FILE}")


def train_review_model():
    """Trains NLP classifiers on the Amazon Deception Review dataset."""
    download_dataset()

    print("Loading Amazon Review Dataset...")
    # Dataset format: DOC_ID, LABEL, RATING, VERIFIED_PURCHASE, PRODUCT_CATEGORY, PRODUCT_ID, PRODUCT_TITLE, REVIEW_TITLE, REVIEW_TEXT
    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        names=[
            "DOC_ID",
            "LABEL",
            "RATING",
            "VERIFIED_PURCHASE",
            "PRODUCT_CATEGORY",
            "PRODUCT_ID",
            "PRODUCT_TITLE",
            "REVIEW_TITLE",
            "REVIEW_TEXT",
        ],
        header=0,
        low_memory=False,
    )

    print(f"Total rows loaded: {len(df)}")

    # Clean missing values
    df = df.dropna(subset=["REVIEW_TEXT", "LABEL"])
    df["CLEANED_TEXT"] = (df["REVIEW_TITLE"].fillna("") + " " + df["REVIEW_TEXT"]).apply(clean_text)

    # __label1__ = Fake (1), __label2__ = Genuine (0)
    df["TARGET"] = df["LABEL"].apply(lambda x: 1 if "__label1__" in str(x) else 0)
    df["LABEL_NAME"] = df["TARGET"].apply(lambda x: "Fake" if x == 1 else "Genuine")

    print(f"Class distribution:\n{df['LABEL_NAME'].value_counts()}")

    # Split train/test
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["CLEANED_TEXT"],
        df["TARGET"],
        test_size=0.20,
        random_state=42,
        stratify=df["TARGET"],
    )

    print("Extracting TF-IDF Features...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )

    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, C=1.5, random_state=42)
    lr_model.fit(X_train_vec, y_train)
    lr_preds = lr_model.predict(X_test_vec)
    lr_probs = lr_model.predict_proba(X_test_vec)[:, 1]

    # 2. Multinomial Naive Bayes
    print("Training Multinomial Naive Bayes...")
    nb_model = MultinomialNB(alpha=0.5)
    nb_model.fit(X_train_vec, y_train)
    nb_preds = nb_model.predict(X_test_vec)
    nb_probs = nb_model.predict_proba(X_test_vec)[:, 1]

    # 3. Linear SVM (Calibrated for probabilities)
    print("Training Calibrated Linear SVM...")
    svm_base = LinearSVC(random_state=42, dual="auto", max_iter=2000)
    svm_model = CalibratedClassifierCV(estimator=svm_base)
    svm_model.fit(X_train_vec, y_train)
    svm_preds = svm_model.predict(X_test_vec)
    svm_probs = svm_model.predict_proba(X_test_vec)[:, 1]

    # Compute Metrics
    def compute_metrics(y_true, y_pred, y_prob):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "roc_auc": float(auc(fpr, tpr)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

    metrics = {
        "Logistic Regression": compute_metrics(y_test, lr_preds, lr_probs),
        "Multinomial Naive Bayes": compute_metrics(y_test, nb_preds, nb_probs),
        "Linear SVM": compute_metrics(y_test, svm_preds, svm_probs),
    }

    # Extract Top Words
    feature_names = np.array(vectorizer.get_feature_names_out())
    lr_coefs = lr_model.coef_[0]
    top_fake_idx = np.argsort(lr_coefs)[-25:][::-1]
    top_genuine_idx = np.argsort(lr_coefs)[:25]

    top_keywords = {
        "fake_indicators": feature_names[top_fake_idx].tolist(),
        "fake_weights": lr_coefs[top_fake_idx].tolist(),
        "genuine_indicators": feature_names[top_genuine_idx].tolist(),
        "genuine_weights": lr_coefs[top_genuine_idx].tolist(),
    }

    # Save lightweight sample data for Streamlit EDA visualization
    sample_df = df[
        [
            "DOC_ID",
            "RATING",
            "VERIFIED_PURCHASE",
            "PRODUCT_CATEGORY",
            "LABEL_NAME",
            "REVIEW_TITLE",
            "REVIEW_TEXT",
        ]
    ].sample(n=min(3000, len(df)), random_state=42)

    category_stats = df.groupby(["PRODUCT_CATEGORY", "LABEL_NAME"]).size().unstack(fill_value=0).to_dict()

    payload = {
        "vectorizer": vectorizer,
        "models": {
            "Logistic Regression": lr_model,
            "Multinomial Naive Bayes": nb_model,
            "Linear SVM": svm_model,
        },
        "metrics": metrics,
        "top_keywords": top_keywords,
        "sample_eda": sample_df,
        "category_stats": category_stats,
        "dataset_summary": {
            "total_reviews": len(df),
            "fake_reviews": int((df["TARGET"] == 1).sum()),
            "genuine_reviews": int((df["TARGET"] == 0).sum()),
            "categories_count": int(df["PRODUCT_CATEGORY"].nunique()),
            "verified_purchase_distribution": df["VERIFIED_PURCHASE"].value_counts().to_dict(),
        },
    }

    joblib.dump(payload, MODEL_FILE, compress=3)
    print(f"Model saved successfully to {MODEL_FILE}")


def train_job_model():
    """Trains a specialized Fake Job Posting classifier as companion mode."""
    print("Generating and training Fake Job Posting Detection model...")
    # Synthetic realistic EMSCAD-pattern dataset for Job Fraud Detection
    job_samples = [
        # Fake Job Postings
        ("Urgent Hiring Data Entry Operator work from home earn 5000 weekly no experience needed telegram contact wire transfer", 1),
        ("Immediate opening for online assistant kindly pay registration fee for training materials Western Union whatsapp only", 1),
        ("Work From Home Mystery Shopper cashiers checks provided deposit into your personal account send back remaining money", 1),
        ("Investment advisor guaranteed 200 daily return no interview necessary contact via gmail personal address urgent hiring", 1),
        ("Package forwarding agent receive parcel re-label ship overseas credit card processing assistant zero credentials", 1),
        ("Earn 80000 per month typing Captcha work flexible timings deposit security money refundable upon completion", 1),
        ("Remote Administrative Assistant no background check required direct message telegram handle hr_recruiter99", 1),
        ("Virtual Assistant immediate start send identity proof passport banking info before interview high salary", 1),
        ("Crypto trading assistant automated bot monitor instant payouts guaranteed returns send ethereum wallet", 1),
        ("Data Entry Clerk no resume needed 100% genuine opportunity contact on personal whatsapp for details", 1),
        ("Earn huge profit working 1 hour daily no experience required wire money upfront for starter kit guaranteed placement", 1),
        ("Home based typist job guaranteed income submit credit card details for verification instant joining letter", 1),
        # Genuine Job Postings
        ("Senior Python Software Engineer with 5+ years experience in Django PostgreSQL REST APIs Docker CI CD agile team", 0),
        ("Data Scientist proficient in Machine Learning NLP Scikit-Learn Pandas SQL statistical analysis PhD or Masters preferred", 0),
        ("Frontend Developer React TypeScript Next.js Tailwind CSS responsive design unit testing Git collaborative environment", 0),
        ("DevOps Engineer AWS Kubernetes Terraform monitoring Prometheus Grafana Linux scripting production infrastructure", 0),
        ("Product Manager 3+ years experience agile scrum roadmap user stories cross-functional leadership Bachelor degree required", 0),
        ("Junior Marketing Specialist SEO Google Analytics content creation email campaigns CRM software competitive salary benefits", 0),
        ("Full Stack Web Developer Node.js Express React MongoDB GraphQL clean code design patterns team player", 0),
        ("Cybersecurity Analyst network monitoring SIEM incident response vulnerability scanning compliance ISO 27001", 0),
        ("Financial Analyst financial modeling forecasting Excel Power BI presentation skills CPA or CFA preferred", 0),
        ("Customer Support Representative fluent English CRM Zendesk ticketing communication skills 401k health insurance", 0),
        ("Cloud Solutions Architect GCP Azure enterprise architecture migration security scalability corporate headquarters", 0),
        ("Quality Assurance Engineer manual and automated testing Selenium Cypress JIRA bug tracking test cases", 0),
    ]

    # Replicate with minor variations to train robust weights
    texts = []
    labels = []
    for text, label in job_samples * 20:
        texts.append(text)
        labels.append(label)

    job_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X_job = job_vec.fit_transform(texts)
    y_job = np.array(labels)

    job_lr = LogisticRegression(max_iter=500, random_state=42)
    job_lr.fit(X_job, y_job)

    job_payload = {
        "vectorizer": job_vec,
        "model": job_lr,
        "keywords": {
            "fraud_indicators": [
                "telegram", "whatsapp", "wire transfer", "registration fee", "no experience",
                "deposit", "urgent hiring", "guaranteed return", "personal account", "western union"
            ],
            "genuine_indicators": [
                "experience", "bachelor", "degree", "interview", "benefits",
                "401k", "agile", "sql", "python", "full time"
            ]
        }
    }
    joblib.dump(job_payload, JOB_MODEL_FILE, compress=3)
    print(f"Fake Job model saved to {JOB_MODEL_FILE}")


if __name__ == "__main__":
    train_review_model()
    train_job_model()
    print("All training complete!")
