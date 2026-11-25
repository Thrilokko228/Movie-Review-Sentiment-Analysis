---
title: Movie Review Sentiment Analyzer
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🎬 Movie Review Sentiment Analyzer

An interactive Machine Learning application that predicts whether a movie review expresses positive or negative sentiment.

## Features

- Text preprocessing
- TF-IDF feature extraction
- Logistic Regression classification
- Positive and negative probability scores
- 60% positive classification threshold
- Interactive Streamlit interface

## Machine Learning Details

| Component | Technology |
|---|---|
| Dataset | IMDb Movie Reviews |
| Text Features | TF-IDF |
| Classification | Logistic Regression |
| Training Accuracy | 94.88% |
| Test Accuracy | 89.91% |
| Interface | Streamlit |
| Deployment | Docker / Hugging Face Spaces |

## How It Works

1. Enter a movie review.
2. The application cleans the text.
3. TF-IDF converts text into numerical features.
4. Logistic Regression predicts sentiment probabilities.
5. The application classifies the review using a 60% positive threshold.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Project Structure

```text
Movie_Review_Sentiment_Analysis/
├── dataset/
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Dataset

IMDb movie reviews dataset from Stanford NLP:
https://huggingface.co/datasets/stanfordnlp/imdb

## Model Performance

Test accuracy: 89.91%

This is an educational Machine Learning project. Predictions are estimates and may not always reflect the actual meaning of a review.
