import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):

    text = re.sub('<.*?>','',text)
    text = text.lower()

    x = ''
    for i in text:
        if i.isalnum():
            x += i
        else:
            x += ' '

    words = []
    for i in x.split():
        if i not in stop_words:
            words.append(ps.stem(i))

    return " ".join(words)


model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("🎬 Movie Review Sentiment Analyzer")

review = st.text_area("Enter your movie review (minimum 10 words)")

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    elif len(review.split()) < 10:
        st.warning("Please enter at least 10 words for better prediction.")

    else:
        
        processed_review = preprocess(review)
        review_vector = vectorizer.transform([processed_review])
        prediction = model.predict(review_vector)

        if prediction[0] == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😡 Negative Review")
