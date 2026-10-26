"""
Fake Product Review & Job Posting Detection System
Mini Project Title: Fake Product Review Detection Using Natural Language Processing and Machine Learning
Streamlit Web Application
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration & CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="TruthGuard AI - Fake Review & Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Main Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Top Banner Styling */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        padding: 2.2rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        border: 1px solid rgba(165, 180, 252, 0.3);
        margin-bottom: 0.75rem;
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        max-width: 850px;
        line-height: 1.5;
    }
    
    /* Card Component */
    .metric-card {
        background: #1e293b;
        padding: 1.4rem;
        border-radius: 14px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
    }
    
    .metric-val {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.3rem;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.6px;
        font-weight: 600;
    }
    
    /* Prediction Pill Badges */
    .badge-fake {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        padding: 1.2rem 1.6rem;
        border-radius: 14px;
        text-align: center;
        font-weight: 700;
        font-size: 1.4rem;
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .badge-genuine {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        color: white;
        padding: 1.2rem 1.6rem;
        border-radius: 14px;
        text-align: center;
        font-weight: 700;
        font-size: 1.4rem;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Keyword Tags */
    .keyword-tag-fake {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.85rem;
        margin: 3px;
        font-weight: 600;
    }
    .keyword-tag-genuine {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.85rem;
        margin: 3px;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions & Model Loading
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    """Preprocess text with lowercase, regex cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_review_pipeline():
    """Loads trained NLP pipeline or creates quick fallback if file missing."""
    model_path = "model_pipeline.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    
    # If not yet trained, return None so user can trigger training or fallback
    return None

@st.cache_resource
def load_job_pipeline():
    """Loads Fake Job Posting model pipeline."""
    job_path = "job_model_pipeline.pkl"
    if os.path.exists(job_path):
        return joblib.load(job_path)
    return None

review_data = load_review_pipeline()
job_data = load_job_pipeline()

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">NLP & MACHINE LEARNING SYSTEM</div>
        <div class="hero-title">🛡️ TruthGuard AI: Deception Detection Platform</div>
        <div class="hero-subtitle">
            Advanced NLP & Machine Learning system for classifying <b>Fake vs Genuine Product Reviews</b> 
            (Amazon Deception Dataset, 21,000 records) and detecting <b>Fraudulent Job Postings</b> using 
            TF-IDF vectorization, Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/fluency/96/shield.png",
    width=64,
)
st.sidebar.title("Navigation & Settings")

app_mode = st.sidebar.radio(
    "Select Operating Mode:",
    [
        "🛒 Fake Product Review Detection",
        "💼 Fake Job Posting Detection",
        "📊 Model Comparison & Metrics",
        "📈 Amazon Dataset EDA & Insights",
        "📖 Project Documentation & Architecture",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Model Settings")
selected_classifier = st.sidebar.selectbox(
    "Classification Algorithm:",
    ["Logistic Regression", "Multinomial Naive Bayes", "Linear SVM"],
    index=0,
    help="Select the ML model to use for inference. Logistic Regression and Naive Bayes are compared side-by-side in this project.",
)

sensitivity_threshold = st.sidebar.slider(
    "Deception Sensitivity Threshold",
    min_value=0.30,
    max_value=0.70,
    value=0.50,
    step=0.05,
    help="Default is 0.50. Lower values make the classifier flag borderline reviews as potentially fake more aggressively.",
)

st.sidebar.divider()
st.sidebar.info(
    "**Project Repository:**\n`github.com/nishantchaubeyy/fakejobposting`\n\n"
    "**Core Technologies:**\nPython • Streamlit • Scikit-Learn • TF-IDF • NLTK/Regex • Plotly"
)


# If model is not trained yet, provide 1-click training in the UI
if review_data is None:
    st.warning("⚠️ Pre-trained model artifact `model_pipeline.pkl` was not detected. Please train the model to continue.")
    col_tr1, col_tr2 = st.columns([1, 3])
    with col_tr1:
        if st.button("🚀 Train Model Now on Amazon Dataset", type="primary"):
            with st.spinner("Downloading Amazon Deception dataset & training Logistic Regression + Naive Bayes..."):
                import subprocess
                res = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
                st.write(res.stdout)
                st.cache_resource.clear()
                st.success("Training complete! Please reload the page.")
                st.rerun()
    st.stop()


# ---------------------------------------------------------
# TAB 1: FAKE PRODUCT REVIEW DETECTION
# ---------------------------------------------------------
if app_mode == "🛒 Fake Product Review Detection":
    st.subheader("🛒 Real-Time Product Review Deception Analyzer")
    st.markdown(
        "Enter any product review below or load one of the test benchmarks. "
        "The model analyzes lexical, stylistic, and sentiment signals to classify whether the review is **Genuine** or **Deceptive / Fake**."
    )

    # Preset sample reviews for quick demonstration
    col_sample1, col_sample2, col_sample3 = st.columns(3)
    preset_title = ""
    preset_text = ""
    preset_rating = 5
    preset_verified = "No (Unverified)"
    preset_cat = "Electronics"

    if col_sample1.button("📌 Sample 1: Deceptive 5-Star Review"):
        preset_title = "BEST PRODUCT EVER INCREDIBLE MUST BUY!!"
        preset_text = (
            "This is absolutely the greatest product I have ever purchased in my entire life! "
            "It works like a miracle and solved all my problems instantly! Amazing quality, flawless, "
            "recommend 1000% to everyone, buy it right now without thinking!!"
        )
        preset_rating = 5
        preset_verified = "No (Unverified)"
        preset_cat = "Beauty"

    if col_sample2.button("📌 Sample 2: Genuine Balanced Review"):
        preset_title = "Solid battery life, slight heating after 2 hours"
        preset_text = (
            "I bought this wireless keyboard 3 weeks ago for my home office setup. The keys have "
            "good tactile feedback and typing is quiet. Battery lasts around 14 days on a single charge. "
            "However, the Bluetooth connection occasionally drops if you are more than 10 feet away. "
            "Overall decent value for $25."
        )
        preset_rating = 4
        preset_verified = "Yes (Verified Purchase)"
        preset_cat = "PC"

    if col_sample3.button("📌 Sample 3: Fake Competitor Smear"):
        preset_title = "Complete trash waste of money terrible"
        preset_text = (
            "Worst item ever manufactured. Broke in five seconds. Total scam and fraud company! "
            "Do not buy from them, go buy the other brand instead. Horrible customer service!"
        )
        preset_rating = 1
        preset_verified = "No (Unverified)"
        preset_cat = "Wireless"

    with st.form("review_analyzer_form"):
        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            review_title_input = st.text_input(
                "Review Headline / Title:",
                value=preset_title if preset_title else "Great wireless charger for desk",
                placeholder="e.g., Amazing product, highly recommended!",
            )
            review_text_input = st.text_area(
                "Full Review Text Body:",
                value=preset_text if preset_text else (
                    "I have been using this charger on my nightstand for two months. "
                    "It charges my phone reliably overnight without overheating. Build quality is sturdy "
                    "and the LED indicator is dim enough not to disturb sleep."
                ),
                height=140,
                placeholder="Paste the full customer review here...",
            )

        with col_f2:
            st.markdown("##### Metadata Features")
            rating_input = st.selectbox(
                "Star Rating:",
                [5, 4, 3, 2, 1],
                index=[5, 4, 3, 2, 1].index(preset_rating),
            )
            verified_input = st.selectbox(
                "Verified Purchase Status:",
                ["Yes (Verified Purchase)", "No (Unverified)"],
                index=0 if preset_verified.startswith("Yes") else 1,
            )
            category_input = st.selectbox(
                "Product Category:",
                [
                    "PC", "Wireless", "Electronics", "Beauty", "Home",
                    "Health & Personal Care", "Automotive", "Sports", "Books", "Other"
                ],
                index=0,
            )

        submit_review = st.form_submit_button("🔍 Analyze Review Deception", type="primary", use_container_width=True)

    if review_text_input:
        full_raw_text = f"{review_title_input} {review_text_input}"
        cleaned = clean_text(full_raw_text)

        # Vectorize
        vectorizer = review_data["vectorizer"]
        model = review_data["models"].get(selected_classifier, review_data["models"]["Logistic Regression"])
        vec_input = vectorizer.transform([cleaned])

        # Inference
        fake_prob = float(model.predict_proba(vec_input)[0][1])
        genuine_prob = 1.0 - fake_prob

        # Apply metadata empirical heuristic adjustment
        # Amazon empirical study: unverified purchase + extreme 5-star or 1-star has higher deception likelihood
        meta_shift = 0.0
        if verified_input.startswith("No"):
            meta_shift += 0.08
        if rating_input in [1, 5] and len(review_text_input.split()) < 25:
            meta_shift += 0.07

        adjusted_fake_prob = min(0.99, max(0.01, fake_prob + meta_shift))
        adjusted_genuine_prob = 1.0 - adjusted_fake_prob
        is_fake = adjusted_fake_prob >= sensitivity_threshold

        st.divider()
        st.markdown("### 📊 Classification Output & Confidence Analysis")

        col_res1, col_res2 = st.columns([1, 1])

        with col_res1:
            if is_fake:
                st.markdown(
                    f"""
                    <div class="badge-fake">
                        🚨 DECEPTIVE / FAKE REVIEW
                        <div style="font-size: 0.95rem; font-weight: 500; margin-top: 6px; opacity: 0.95;">
                            Probability of Deception: <b>{adjusted_fake_prob * 100:.1f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="badge-genuine">
                        ✅ GENUINE / TRUTHFUL REVIEW
                        <div style="font-size: 0.95rem; font-weight: 500; margin-top: 6px; opacity: 0.95;">
                            Probability of Truthfulness: <b>{adjusted_genuine_prob * 100:.1f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Linguistic feature breakdown
            words = review_text_input.split()
            word_count = len(words)
            exclamation_count = review_text_input.count("!")
            uppercase_words = sum(1 for w in words if w.isupper() and len(w) > 1)
            
            st.markdown("<br>", unsafe_allow_html=True)
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Word Count</div>
                        <div class="metric-val">{word_count}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Exclamations</div>
                        <div class="metric-val">{exclamation_count}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">All-Caps Words</div>
                        <div class="metric-val">{uppercase_words}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_res2:
            # Gauge chart
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=adjusted_fake_prob * 100,
                    title={"text": "Deception Risk Index (%)", "font": {"size": 18, "color": "#cbd5e1"}},
                    number={"suffix": "%", "font": {"color": "#ef4444" if is_fake else "#10b981", "size": 34}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                        "bar": {"color": "#ef4444" if is_fake else "#10b981"},
                        "steps": [
                            {"range": [0, 40], "color": "rgba(16, 185, 129, 0.2)"},
                            {"range": [40, 65], "color": "rgba(245, 158, 11, 0.2)"},
                            {"range": [65, 100], "color": "rgba(239, 68, 68, 0.2)"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 3},
                            "thickness": 0.75,
                            "value": sensitivity_threshold * 100,
                        },
                    },
                )
            )
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=240,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Lexical Attribution & Key Tokens
        st.markdown("#### 🔍 Lexical Cues & Detected Trigger Words")
        st.markdown(
            "Below are words from your review that influenced the classification based on TF-IDF model weights:"
        )

        tokens = set(cleaned.split())
        fake_keywords = set(review_data["top_keywords"]["fake_indicators"])
        genuine_keywords = set(review_data["top_keywords"]["genuine_indicators"])

        matched_fake = tokens.intersection(fake_keywords)
        matched_genuine = tokens.intersection(genuine_keywords)

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown("**Deceptive / Exaggerated Patterns Detected:**")
            if matched_fake:
                for w in matched_fake:
                    st.markdown(f'<span class="keyword-tag-fake">🚨 {w}</span>', unsafe_allow_html=True)
            else:
                st.caption("No strong deceptive trigger keywords detected.")

        with col_k2:
            st.markdown("**Genuine / Specific Evaluative Words Detected:**")
            if matched_genuine:
                for w in matched_genuine:
                    st.markdown(f'<span class="keyword-tag-genuine">✅ {w}</span>', unsafe_allow_html=True)
            else:
                st.caption("No strong genuine trigger keywords detected.")

    # -----------------------------------------------------
    # Batch Processing Sub-Section
    # -----------------------------------------------------
    st.divider()
    st.markdown("### 📁 Batch Review Classification (CSV Upload)")
    st.markdown(
        "Upload a `.csv` file containing multiple reviews to classify them in bulk. "
        "The CSV should contain a column named `review_text` (or `review`, `text`)."
    )

    uploaded_file = st.file_uploader("Upload Reviews CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df)}** reviews from uploaded file.")
            
            # Find candidate review column
            possible_cols = [c for c in batch_df.columns if "review" in c.lower() or "text" in c.lower()]
            target_col = possible_cols[0] if possible_cols else batch_df.columns[0]
            
            col_target = st.selectbox("Select Review Text Column:", batch_df.columns, index=batch_df.columns.get_loc(target_col))

            if st.button("⚡ Run Batch Classification", type="primary"):
                with st.spinner("Processing reviews with NLP pipeline..."):
                    cleaned_texts = batch_df[col_target].astype(str).apply(clean_text)
                    vecs = review_data["vectorizer"].transform(cleaned_texts)
                    model = review_data["models"][selected_classifier]
                    probs = model.predict_proba(vecs)[:, 1]
                    
                    batch_df["Fake_Probability"] = np.round(probs, 4)
                    batch_df["Prediction"] = np.where(probs >= sensitivity_threshold, "Fake", "Genuine")

                    c1, c2 = st.columns([1, 1])
                    with c1:
                        fig_batch = px.pie(
                            batch_df,
                            names="Prediction",
                            title="Batch Prediction Breakdown",
                            color="Prediction",
                            color_discrete_map={"Fake": "#ef4444", "Genuine": "#10b981"},
                            hole=0.45,
                        )
                        fig_batch.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig_batch, use_container_width=True)

                    with c2:
                        fake_count = (batch_df["Prediction"] == "Fake").sum()
                        gen_count = (batch_df["Prediction"] == "Genuine").sum()
                        st.markdown(f"""
                        <div class="metric-card" style="margin-bottom: 10px;">
                            <div class="metric-label">Detected Fake Reviews</div>
                            <div class="metric-val" style="color: #ef4444;">{fake_count} ({fake_count/len(batch_df)*100:.1f}%)</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Detected Genuine Reviews</div>
                            <div class="metric-val" style="color: #10b981;">{gen_count} ({gen_count/len(batch_df)*100:.1f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.dataframe(batch_df.head(20), use_container_width=True)

                    csv_data = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Classified Results CSV",
                        data=csv_data,
                        file_name="classified_reviews_output.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")


# ---------------------------------------------------------
# TAB 2: FAKE JOB POSTING DETECTION (COMPANION MODE)
# ---------------------------------------------------------
elif app_mode == "💼 Fake Job Posting Detection":
    st.subheader("💼 Fake Job Posting & Employment Scam Detector")
    st.markdown(
        "Detect scam job postings, work-from-home fraud, and wire-transfer recruitment scams "
        "using Natural Language Processing."
    )

    col_job1, col_job2 = st.columns(2)
    sample_job_title = ""
    sample_job_desc = ""

    if col_job1.button("🚨 Sample Scam Job: Telegram Data Entry"):
        sample_job_title = "Work From Home Data Entry Operator - $4,500/week"
        sample_job_desc = (
            "Urgent hiring! Earn $4,500 weekly with no experience needed. Flexible hours. "
            "Kindly connect with our HR recruiter on Telegram @hr_direct_recruitment. "
            "A small refundable registration fee of $50 is required for verification and training starter kit."
        )

    if col_job2.button("✅ Sample Legitimate Job: Software Engineer"):
        sample_job_title = "Senior Python Backend Engineer"
        sample_job_desc = (
            "We are seeking a Senior Python Engineer with 4+ years experience in FastAPI, PostgreSQL, "
            "and Docker. Competitive base salary with 401(k) matching, comprehensive health insurance, "
            "and flexible hybrid work policy. Bachelor's in CS or equivalent experience required."
        )

    with st.form("job_detector_form"):
        job_title_input = st.text_input(
            "Job Title:",
            value=sample_job_title if sample_job_title else "Customer Support Associate",
        )
        job_desc_input = st.text_area(
            "Job Description & Requirements:",
            value=sample_job_desc if sample_job_desc else (
                "Seeking customer support representative to handle inbound email queries. "
                "Proficiency in written English, 1+ year customer service experience, healthcare benefits included."
            ),
            height=140,
        )
        submit_job = st.form_submit_button("🔍 Screen Job Posting for Fraud", type="primary", use_container_width=True)

    if job_desc_input and job_data:
        full_job = f"{job_title_input} {job_desc_input}"
        c_job = clean_text(full_job)
        j_vec = job_data["vectorizer"].transform([c_job])
        j_prob = float(job_data["model"].predict_proba(j_vec)[0][1])

        st.divider()
        st.markdown("### 🛡️ Job Fraud Assessment Result")
        c_out1, c_out2 = st.columns([1, 1])

        with c_out1:
            if j_prob >= 0.50:
                st.markdown(
                    f"""
                    <div class="badge-fake">
                        🚨 SUSPICIOUS / FRAUDULENT JOB POSTING
                        <div style="font-size: 0.95rem; font-weight: 500; margin-top: 6px; opacity: 0.95;">
                            Scam Risk Probability: <b>{j_prob * 100:.1f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="badge-genuine">
                        ✅ LEGITIMATE JOB POSTING
                        <div style="font-size: 0.95rem; font-weight: 500; margin-top: 6px; opacity: 0.95;">
                            Legitimacy Confidence: <b>{(1 - j_prob) * 100:.1f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with c_out2:
            st.markdown("#### High-Risk Red Flags Monitored")
            st.markdown("""
            - 🚩 Mentions of private messaging (Telegram / WhatsApp)
            - 🚩 Unrealistic compensation with zero experience required
            - 🚩 Request for upfront fee / registration / equipment deposit
            - 🚩 Payment via cashier's checks, crypto, or wire transfer
            """)


# ---------------------------------------------------------
# TAB 3: MODEL COMPARISON & METRICS
# ---------------------------------------------------------
elif app_mode == "📊 Model Comparison & Metrics":
    st.subheader("📊 Comparative Model Evaluation (Logistic Regression vs Naive Bayes vs SVM)")
    st.markdown(
        "As outlined in the Project Objectives, multiple classification algorithms are trained on the "
        "**Amazon Review Deception Dataset (21,000 samples)** using TF-IDF feature extraction. "
        "Their performance is benchmarked on an independent 20% holdout test set."
    )

    metrics = review_data["metrics"]

    # Comparative Metric Cards
    m_cols = st.columns(3)
    for idx, (m_name, m_vals) in enumerate(metrics.items()):
        with m_cols[idx]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{m_name}</div>
                    <div class="metric-val">{m_vals['accuracy']*100:.2f}%</div>
                    <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 8px;">
                        • Precision: <b>{m_vals['precision']*100:.2f}%</b><br>
                        • Recall: <b>{m_vals['recall']*100:.2f}%</b><br>
                        • F1-Score: <b>{m_vals['f1']*100:.2f}%</b><br>
                        • ROC-AUC: <b>{m_vals['roc_auc']:.4f}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison Bar Chart
    comp_df = pd.DataFrame([
        {"Model": k, "Metric": "Accuracy", "Value": v["accuracy"] * 100}
        for k, v in metrics.items()
    ] + [
        {"Model": k, "Metric": "Precision", "Value": v["precision"] * 100}
        for k, v in metrics.items()
    ] + [
        {"Model": k, "Metric": "Recall", "Value": v["recall"] * 100}
        for k, v in metrics.items()
    ] + [
        {"Model": k, "Metric": "F1-Score", "Value": v["f1"] * 100}
        for k, v in metrics.items()
    ])

    fig_comp = px.bar(
        comp_df,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        title="Benchmark Metric Comparison Across Classification Models (%)",
        text_auto=".1f",
        color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b"],
    )
    fig_comp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[70, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Confusion Matrix & ROC Curves
    st.markdown("### 🔲 Confusion Matrix & ROC Curves")
    col_cm, col_roc = st.columns(2)

    with col_cm:
        cm_model_choice = st.selectbox(
            "Select Model for Confusion Matrix:",
            list(metrics.keys()),
            index=0,
        )
        cm = np.array(metrics[cm_model_choice]["confusion_matrix"])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            x=["Genuine", "Fake"],
            y=["Genuine", "Fake"],
            color_continuous_scale="Purples",
            title=f"Confusion Matrix: {cm_model_choice}",
        )
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        fig_roc = go.Figure()
        for m_name, m_vals in metrics.items():
            fig_roc.add_trace(
                go.Scatter(
                    x=m_vals["fpr"],
                    y=m_vals["tpr"],
                    mode="lines",
                    name=f"{m_name} (AUC={m_vals['roc_auc']:.3f})",
                    line=dict(width=2),
                )
            )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Chance",
                line=dict(dash="dash", color="gray"),
            )
        )
        fig_roc.update_layout(
            title="Receiver Operating Characteristic (ROC) Curves",
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (TPR)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Top Informative Words
    st.markdown("### 🔤 Most Informative Words Learned by Model")
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown("#### 🚨 Top Words Predictive of Fake Reviews")
        fake_kw_df = pd.DataFrame({
            "Token": review_data["top_keywords"]["fake_indicators"][:15],
            "Weight": review_data["top_keywords"]["fake_weights"][:15],
        })
        fig_fake_kw = px.bar(
            fake_kw_df[::-1],
            x="Weight",
            y="Token",
            orientation="h",
            color="Weight",
            color_continuous_scale="Reds",
        )
        fig_fake_kw.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_fake_kw, use_container_width=True)

    with col_w2:
        st.markdown("#### ✅ Top Words Predictive of Genuine Reviews")
        gen_kw_df = pd.DataFrame({
            "Token": review_data["top_keywords"]["genuine_indicators"][:15],
            "Weight": np.abs(review_data["top_keywords"]["genuine_weights"][:15]),
        })
        fig_gen_kw = px.bar(
            gen_kw_df[::-1],
            x="Weight",
            y="Token",
            orientation="h",
            color="Weight",
            color_continuous_scale="Greens",
        )
        fig_gen_kw.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_gen_kw, use_container_width=True)


# ---------------------------------------------------------
# TAB 4: DATASET EDA & INSIGHTS
# ---------------------------------------------------------
elif app_mode == "📈 Amazon Dataset EDA & Insights":
    st.subheader("📈 Exploratory Data Analysis: Amazon Deception Corpus (21,000 Reviews)")
    st.markdown(
        "Exploration of linguistic and stylistic patterns across the 30 product categories "
        "in the curated Amazon deception benchmark dataset."
    )

    summary = review_data["dataset_summary"]
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Corpus Reviews</div>
                <div class="metric-val">{summary['total_reviews']:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Deceptive Reviews</div>
                <div class="metric-val" style="color: #ef4444;">{summary['fake_reviews']:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Genuine Reviews</div>
                <div class="metric-val" style="color: #10b981;">{summary['genuine_reviews']:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Product Categories</div>
                <div class="metric-val">{summary['categories_count']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    sample_eda = review_data["sample_eda"].copy()
    sample_eda["Review_Length"] = sample_eda["REVIEW_TEXT"].astype(str).apply(lambda x: len(x.split()))

    col_eda1, col_eda2 = st.columns(2)

    with col_eda1:
        # Star Rating Distribution: Fake vs Genuine
        fig_rating = px.histogram(
            sample_eda,
            x="RATING",
            color="LABEL_NAME",
            barmode="group",
            title="Star Rating Distribution (Fake vs Genuine)",
            color_discrete_map={"Fake": "#ef4444", "Genuine": "#10b981"},
            category_orders={"RATING": [1, 2, 3, 4, 5]},
        )
        fig_rating.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_rating, use_container_width=True)

    with col_eda2:
        # Verified Purchase Impact
        fig_vp = px.histogram(
            sample_eda,
            x="VERIFIED_PURCHASE",
            color="LABEL_NAME",
            barmode="group",
            title="Verified Purchase Status vs Review Authenticity",
            color_discrete_map={"Fake": "#ef4444", "Genuine": "#10b981"},
        )
        fig_vp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_vp, use_container_width=True)

    # Review Length Distribution
    fig_len = px.box(
        sample_eda[sample_eda["Review_Length"] < 300],
        x="LABEL_NAME",
        y="Review_Length",
        color="LABEL_NAME",
        title="Review Length (Word Count) Distribution by Class",
        color_discrete_map={"Fake": "#ef4444", "Genuine": "#10b981"},
    )
    fig_len.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_len, use_container_width=True)


# ---------------------------------------------------------
# TAB 5: PROJECT DOCUMENTATION & ARCHITECTURE
# ---------------------------------------------------------
elif app_mode == "📖 Project Documentation & Architecture":
    st.subheader("📖 Mini Project Specifications & Academic Documentation")
    
    st.markdown(
        """
        ### 1. Mini Project Title
        **Fake Product Review Detection Using Natural Language Processing and Machine Learning**

        ---
        ### 2. Problem Statement
        Online shopping platforms rely heavily on customer reviews to help consumers make informed purchasing decisions. 
        However, deceptive or fake reviews can mislead customers, manipulate product ratings, and erode trust in online marketplaces. 
        Manually identifying deceptive reviews is impractical due to the massive volume of reviews published daily. 
        This project develops a machine learning system leveraging Natural Language Processing (NLP) to inspect review text, 
        extract contextual patterns, and classify reviews as fake or genuine.

        ---
        ### 3. Objective of the Project
        - Preprocess raw review text via tokenization, lowercase normalization, and regex filtering.
        - Transform unstructured text into high-dimensional numerical feature vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)** with unigrams and bigrams.
        - Train and benchmark supervised binary classification algorithms: **Logistic Regression**, **Multinomial Naive Bayes**, and **Linear Support Vector Machines (LinearSVC)**.
        - Evaluate model performance using Accuracy, Precision, Recall, F1-Score, and ROC-AUC curves.
        - Deliver an interactive, deployable Streamlit web application supporting single review inference, batch file classification, and exploratory data visualization.

        ---
        ### 4. Dataset Details
        - **Dataset Name:** Amazon Product Reviews Dataset for Fake Review Detection.
        - **Source:** [GitHub: aayush210789/Deception-Detection-on-Amazon-reviews-dataset](https://github.com/aayush210789/Deception-Detection-on-Amazon-reviews-dataset)
        - **Row Count:** ~21,000 labeled reviews across 30 product categories.
        - **Target Variable:** `LABEL` (`__label1__` = Deceptive/Fake, `__label2__` = Genuine/Truthful).

        ---
        ### 5. Model Performance Results
        Trained on the Amazon Deception Corpus (80% train / 20% test split, stratified):

        | Model | Accuracy | Precision | Recall | F1-Score |
        |:---|:---:|:---:|:---:|:---:|
        | **Logistic Regression** | **65.52%** | **65.54%** | **65.48%** | **65.51%** |
        | **Multinomial Naive Bayes** | **64.57%** | **64.87%** | **63.57%** | **64.21%** |

        > **Note:** Logistic Regression outperforms Multinomial Naive Bayes on all four evaluation metrics on this balanced binary classification task.

        ---
        ### 5. System Architecture
        ```
        [ Raw Text / CSV Upload ]
                   │
                   ▼
        [ Text Preprocessing (Cleaning, Lowercase, Regex) ]
                   │
                   ▼
        [ Feature Extraction: TF-IDF (1-2 N-grams, 10,000 features) ]
                   │
                   ▼
        [ Machine Learning Models: Logistic Regression | Naive Bayes | Linear SVM ]
                   │
                   ▼
        [ Deception Probability & Confidence Calibration ]
                   │
                   ▼
        [ Streamlit Interactive Dashboard (Gauge, Attribution, Batch Export) ]
        ```

        ---
        ### 6. Deployment on Streamlit Cloud & GitHub
        This project is structured for 1-click cloud deployment:
        1. Repository: [https://github.com/nishantchaubeyy/fakejobposting.git](https://github.com/nishantchaubeyy/fakejobposting.git)
        2. Streamlit Cloud: Log in to [share.streamlit.io](https://share.streamlit.io), connect your GitHub account, pick `nishantchaubeyy/fakejobposting`, set main file as `app.py`, and click Deploy!
        """
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center style='color: #64748b; font-size: 0.85rem;'>"
    "TruthGuard AI • Fake Review & Fraud Detection System • Deployed with Streamlit & Scikit-Learn"
    "</center>",
    unsafe_allow_html=True,
)
