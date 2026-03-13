import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time


nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = re.sub('<.*?>', '', text)
    text = text.lower()

    x = ""
    for i in text:
        if i.isalnum():
            x += i
        else:
            x += " "

    words = []
    for i in x.split():
        if i not in stop_words:
            words.append(ps.stem(i))

    return " ".join(words)


@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

try:
    model, vectorizer = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error("Could not load model.pkl or vectorizer.pkl. Please ensure they are in the same directory.")


st.set_page_config(
    page_title="CineSense | Movie Sentiment Analyzer",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        /* Global Background and Fonts */
        .stApp {
            background-color: #0b0c10;
            color: #c5c6c7;
            font-family: 'Helvetica Neue', sans-serif;
        }
        
        /* Hero Title */
        .title {
            font-size: 60px;
            font-weight: 900;
            background: -webkit-linear-gradient(#f1c40f, #e67e22);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
            padding-bottom: 0px;
            letter-spacing: 1px;
        }
        
        /* Subtitle */
        .subtitle {
            color: #a0a3a8;
            font-size: 22px;
            font-weight: 300;
            margin-top: 5px;
            margin-bottom: 40px;
        }
        
        /* Style the Text Area */
        .stTextArea textarea {
            background-color: #1f2833 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            border: 1px solid #45a29e !important;
            font-size: 16px !important;
            padding: 15px !important;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.3);
        }
        .stTextArea textarea:focus {
            border: 1px solid #66fcf1 !important;
            box-shadow: 0px 0px 10px #66fcf1;
        }

        /* Style the Button */
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #e67e22, #f1c40f);
            color: #0b0c10 !important;
            font-size: 20px;
            padding: 12px 0;
            border-radius: 8px;
            border: none;
            font-weight: 900;
            transition: all 0.3s ease;
            box-shadow: 0px 4px 15px rgba(241, 196, 15, 0.4);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0px 6px 20px rgba(241, 196, 15, 0.6);
            background: linear-gradient(90deg, #f1c40f, #e67e22);
        }

        /* Result Cards */
        .result-positive {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: #ffffff;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            font-size: 28px;
            font-weight: 800;
            box-shadow: 0px 10px 30px rgba(56, 239, 125, 0.3);
            animation: fadeIn 0.5s ease-in-out;
        }
        .result-negative {
            background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
            color: #ffffff;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            font-size: 28px;
            font-weight: 800;
            box-shadow: 0px 10px 30px rgba(239, 71, 58, 0.3);
            animation: fadeIn 0.5s ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* --- SIDEBAR STYLING --- */
        [data-testid="stSidebar"] {
            background-color: #121418;
            border-right: 1px solid #1f2833;
        }
        .sidebar-title {
            font-size: 26px;
            font-weight: bold;
            background: -webkit-linear-gradient(#66fcf1, #45a29e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            margin-top: 10px;
        }
        .sidebar-text {
            font-size: 15px;
            color: #c5c6c7;
            line-height: 1.5;
        }
        .sidebar-box {
            background-color: #1f2833;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #f1c40f;
            margin-top: 25px;
            margin-bottom: 25px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    # Centered Logo
    st.markdown(
        """
        <div style='text-align: center; padding-top: 20px;'>
            <img src='https://cdn-icons-png.flaticon.com/512/3171/3171927.png' width='90' style='filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.5));'>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("<div class='sidebar-title'>About CineSense</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sidebar-text'>CineSense uses a powerful Natural Language Processing (NLP) model to read your movie review and instantly detect whether the underlying emotional tone is positive or negative.</div>", 
        unsafe_allow_html=True
    )
    
    # Custom Styled Instructions Box
    st.markdown(
        """
        <div class='sidebar-box'>
            <h4 style='color: #f1c40f; margin-top: 0px; margin-bottom: 10px; font-weight: bold;'>💡 How it works:</h4>
            <ol style='color: #c5c6c7; margin-bottom: 0px; padding-left: 20px; font-size: 14px; line-height: 1.8;'>
                <li>Watch a movie 🍿</li>
                <li>Write your honest thoughts</li>
                <li>Click <b>Analyze Sentiment</b></li>
                <li>Let the AI detect your vibe!</li>
            </ol>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Tech Stack Section
    st.markdown("### 🛠️ Tech Stack")
    st.markdown(
        """
        <div style='font-size: 14px; color: #a0a3a8; line-height: 1.6;'>
        • <b>Frontend:</b> Streamlit<br>
        • <b>NLP Engine:</b> NLTK (Stemming)<br>
        • <b>Vectorization:</b> TF-IDF / BoW<br>
        • <b>Model:</b> Machine Learning
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("<br><hr style='border: 1px solid #1f2833;'><br>", unsafe_allow_html=True)
    st.caption("🎬 Made with ❤️ for Movie Lovers")


col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("<h1 class='title'>CineSense AI</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Instantly decode the sentiment of any movie review. Type your thoughts and let the AI do the rest.</p>",
        unsafe_allow_html=True,
    )
    
  
    review = st.text_area("Drop your review below:", placeholder="e.g., The cinematography was absolutely breathtaking, but the plot fell flat in the second half...", height=200)

with col2:
    st.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=2070&auto=format&fit=crop", caption="What did you think of the film?", use_container_width=True)


st.markdown("<br>", unsafe_allow_html=True) 
col_btn, col_empty = st.columns([1, 2]) 

with col_btn:
    analyze_button = st.button("🎬 Analyze Sentiment")

if analyze_button and models_loaded:
    if review.strip() == "":
        st.error("🚨 Please enter a review before analyzing.")
    elif len(review.split()) < 10:
        st.warning("⚠️ Please enter at least 10 words for a more accurate prediction.")
    else:
        with st.spinner('Analyzing the cinematic vibes...'):
            time.sleep(1) 
            
            processed_review = preprocess(review)
            review_vector = vectorizer.transform([processed_review])
            prediction = model.predict(review_vector)

        st.markdown("<br>", unsafe_allow_html=True)
        if prediction[0] == 1:
            st.markdown(
                "<div class='result-positive'>✨ Fresh & Positive</div>",
                unsafe_allow_html=True,
            )
            st.balloons() 
        else:
            st.markdown(
                "<div class='result-negative'>🍅 Rotten & Negative</div>",
                unsafe_allow_html=True,
            )
