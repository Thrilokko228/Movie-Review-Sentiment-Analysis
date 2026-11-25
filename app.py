import sys
from pathlib import Path

import joblib
import streamlit as st

# Project paths
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"

sys.path.insert(0, str(SRC_DIR))

from preprocessing import clean_text

# Application settings
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

THRESHOLD = 0.60

# Load trained model and vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "sentiment_model.pkl")
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer


# Custom styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #172554);
    color: white;
}
.main .block-container {
    max-width: 850px;
    padding-top: 3rem;
}
.hero {
    text-align: center;
    padding: 25px 10px;
}
.hero h1 {
    color: #60a5fa;
    font-size: 38px;
    font-weight: 800;
}
.hero p {
    color: #cbd5e1;
    font-size: 17px;
}
.result-card {
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
}
.positive {
    background: #123c32;
    border: 1px solid #22c55e;
}
.negative {
    background: #451f2a;
    border: 1px solid #ef4444;
}
.uncertain {
    background: #493719;
    border: 1px solid #f59e0b;
}
.footer {
    text-align: center;
    color: #94a3b8;
    padding: 30px 0;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# Header
st.markdown("""
<div class="hero">
    <h1>🎬 Movie Review Analyzer</h1>
    <p>Discover the sentiment behind every movie review using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)


# Load model
try:
    model, vectorizer = load_artifacts()
except Exception as error:
    st.error("Unable to load the trained model or TF-IDF vectorizer.")
    st.code(str(error))
    st.stop()


# Information
col1, col2 = st.columns(2)

with col1:
    st.metric("Model", "Logistic Regression")

with col2:
    st.metric("Classification Threshold", "60%")


st.divider()

st.subheader("✍️ Enter Your Movie Review")

review = st.text_area(
    "Write or paste your review below:",
    height=180,
    placeholder="Example: This movie was amazing! The acting and story were excellent..."
)

analyze = st.button(
    "🔍 Analyze Review",
    use_container_width=True,
    type="primary"
)


# Prediction
if analyze:

    if not review.strip():
        st.warning("Please enter a movie review first.")

    else:
        cleaned_review = clean_text(review)

        if not cleaned_review:
            st.warning("The review does not contain usable text. Please enter another review.")

        else:
            features = vectorizer.transform([cleaned_review])

            probabilities = model.predict_proba(features)[0]

            negative_probability = float(probabilities[0])
            positive_probability = float(probabilities[1])

            # Apply the same 60% threshold as predict.py
            if positive_probability >= THRESHOLD:
                sentiment = "POSITIVE"
                result_class = "positive"
                result_icon = "😊"

            else:
                sentiment = "NEGATIVE"
                result_class = "negative"
                result_icon = "😞"

            st.markdown(
                f"""
                <div class="result-card {result_class}">
                    <h2>{result_icon} {sentiment} REVIEW</h2>
                    <p>Predicted sentiment based on the 60% threshold</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("📊 Prediction Probabilities")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Positive Probability",
                    f"{positive_probability * 100:.2f}%"
                )
                st.progress(positive_probability)

            with col2:
                st.metric(
                    "Negative Probability",
                    f"{negative_probability * 100:.2f}%"
                )
                st.progress(negative_probability)

            st.caption(
                "Note: The 60% threshold means a review is classified as "
                "positive when its positive probability is at least 60%. "
                "Otherwise, it is classified as negative."
            )

            with st.expander("How does this prediction work?"):
                st.write("""
                1. Your review is cleaned using text preprocessing.
                2. TF-IDF converts the review into numerical features.
                3. Logistic Regression calculates positive and negative probabilities.
                4. The application applies the 60% positive classification threshold.
                """)

                st.write("Original review:")
                st.write(review)

                st.write("Cleaned review:")
                st.write(cleaned_review)


# Footer
st.markdown("""
<div class="footer">
    Built with Python, Scikit-learn and Streamlit<br>
    Movie Review Sentiment Analysis | Machine Learning Project
</div>
""", unsafe_allow_html=True)
