# 🎬 CineSense AI — Movie Sentiment Analyser

> A high-performance Natural Language Processing (NLP) pipeline and machine learning application engineered to decode the emotional tone of movie reviews. Powered by a **Multinomial Naive Bayes** classifier trained on the robust **IMDB Movie Reviews Dataset**, the system uses text normalization and **TF-IDF vectorization** to instantly determine whether an underlying review vibe is positive or negative.

🚀 **Live Web Application:** [Explore CineSense AI Live](https://movie-sentiment-analyser-efuov6rkjkf9d9oveappn7.streamlit.app/)

---

## 📋 Core Architectural Pipeline

CineSense AI processes unstructured text inputs through a structured, low-latency data engineering and inference pipeline:

* 🧼 **Text Preprocessing & Stemming:** Cleans raw user input by isolating text tokens and leveraging the **NLTK** engine to apply efficient stemming routines, shrinking variant words to their base linguistic roots.
* 📐 **Feature Extraction Matrix:** Transforms normalized word sequences into mathematical features utilizing **TF-IDF (Term Frequency-Inverse Document Frequency)** and Bag-of-Words (BoW) vector spaces.
* 🤖 **Probabilistic Inference Engine:** Passes high-dimensional vector inputs directly to a serialized **Multinomial Naive Bayes** classification framework to instantly score text polarity.

---

## ✨ Features

* **Real-Time Vibe Assessment:** Paste any raw movie review text directly into the console to instantly decode the underlying emotional perspective.
* **Clean Context-Driven Layout:** Built with a custom dual-panel responsive interface, complete with a step-by-step runtime breakdown and system overview sidebar.
* **Serialized Pipeline Caching:** Keeps runtime deployment ultra-fast by separating model training scripts from application initialization via optimized `.pkl` checkpoint files.

---

## 🛠️ Tech Stack

* **Frontend Framework:** Streamlit
* **NLP Pipeline Engine:** NLTK (Tokenization & Stemming)
* **Vectorization Matrices:** TF-IDF / Bag of Words (BoW)
* **Classification Algorithm:** Multinomial Naive Bayes
* **Core Core Dataset:** IMDB Movie Reviews Dataset

---

## 📂 Project Directory Structure

```text
Movie-Sentiment-Analyser/
│
├── app.py             # Main Streamlit web application and layout setup
├── model.pkl          # Trained and serialized Multinomial Naive Bayes model checkpoint
├── vectorizer.pkl     # Pre-calculated TF-IDF token vectorizer matrix
├── requirements.txt   # Environment dependency declarations
└── README.md          # Technical documentation and repository profile page
