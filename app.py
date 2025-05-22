import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
import nltk
for pkg in ['punkt','stopwords','wordnet','averaged_perceptron_tagger']:
        nltk.download(pkg)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud

# -- Utility functions ------------------------------------------------------

def load_model(path='models/model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_vectorizer(path='models/tfidf_vectorizer.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def preprocess_text(text):
    # lowercase
    text = text.lower()
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # tokenize sentences then words
    words = []
    for sent in sent_tokenize(text):
        words.extend(word_tokenize(sent))
    # remove stopwords & punctuation & non-alpha
    stop = set(stopwords.words('english'))
    cleaned = [w for w in words if w.isalpha() and w not in stop]
    # stemming + lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    pos_map = { 'N': wordnet.NOUN, 'V': wordnet.VERB, 'J': wordnet.ADJ, 'R': wordnet.ADV }
    processed = []
    for w in cleaned:
        w_stem = stemmer.stem(w)
        p = pos_tag([w_stem])[0][1][0]
        wn = pos_map.get(p, wordnet.NOUN)
        processed.append(lemmatizer.lemmatize(w_stem, wn))
    return ' '.join(processed)

# -- Main app ---------------------------------------------------------------

def main():
    st.title("Topic Modeling Explorer")
    st.sidebar.header("Settings")
    menu = st.sidebar.selectbox("Choose action:", ["Analyze CSV", "Predict Single Text"])

    # Load models once
    vectorizer = load_vectorizer()
    model = load_model()
    feature_names = vectorizer.get_feature_names_out()

    if menu == "Analyze CSV":
        uploaded = st.file_uploader("Upload articles CSV", type=['csv'])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Raw data:", df.head())
            # preprocess content column
            df['processed'] = df['content'].fillna('').apply(preprocess_text)
            X = vectorizer.transform(df['processed'])
            clusters = model.predict(X)
            df['cluster'] = clusters
            st.write("Cluster assignments:" , df[['title','cluster']].head())
            # show distribution
            st.bar_chart(df['cluster'].value_counts().sort_index())
            # top words per cluster
            topn = st.slider("Top words per cluster", min_value=5, max_value=20, value=10)
            centers = model.cluster_centers_
            top_words = {}
            for i, center in enumerate(centers):
                inds = center.argsort()[-topn:][::-1]
                top_words[i] = [feature_names[j] for j in inds]
            for cid, words in top_words.items():
                st.write(f"Cluster {cid}:", ", ".join(words))
                # word cloud
                wc = WordCloud(width=400, height=200).generate(' '.join(words))
                st.image(wc.to_array(), caption=f"WordCloud: Cluster {cid}")

    else:
        text = st.text_area("Enter text to classify")
        if st.button("Predict topic") and text:
            proc = preprocess_text(text)
            vec = vectorizer.transform([proc])
            pred = model.predict(vec)[0]
            st.write(f"Predicted cluster: {pred}")
            # show top words in cluster
            center = model.cluster_centers_[pred]
            topn = st.number_input("Words to show", min_value=5, max_value=20, value=10)
            inds = center.argsort()[-topn:][::-1]
            words = [feature_names[j] for j in inds]
            st.write("Topic words:", ", ".join(words))

if __name__ == "__main__":
    main()
