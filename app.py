import streamlit as st
import pandas as pd
import re
import pickle
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud

# Make sure the required NLTK resources are actually present
RESOURCES = {
    'punkt':      'tokenizers/punkt/english.pickle',
    'stopwords':  'corpora/stopwords.zip',
    'wordnet':    'corpora/wordnet.zip',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger.zip',
}

for pkg, path in RESOURCES.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

# Initialize heavy objects once
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()
POS_MAP = {'N': wordnet.NOUN, 'V': wordnet.VERB, 'J': wordnet.ADJ, 'R': wordnet.ADV}

# Cache model & vectorizer loads
@st.cache_resource
def load_vectorizer(path: str = 'models/tfidf_vectorizer.pkl') -> TfidfVectorizer:
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_model(path: str = 'models/model.pkl') -> KMeans:
    with open(path, 'rb') as f:
        return pickle.load(f)

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    tokens = []
    for sent in sent_tokenize(text):
        tokens.extend(word_tokenize(sent))
    filtered = [w for w in tokens if w.isalpha() and w not in STOPWORDS]
    processed = []
    for w in filtered:
        stem = STEMMER.stem(w)
        pos = pos_tag([stem])[0][1][0]
        wn_pos = POS_MAP.get(pos, wordnet.NOUN)
        processed.append(LEMMATIZER.lemmatize(stem, wn_pos))
    return ' '.join(processed)

def main():
    st.title("Topic Modeling Explorer")
    st.sidebar.header("Settings")
    action = st.sidebar.selectbox("Action:", ["Analyze CSV", "Predict Single Text"])

    vectorizer = load_vectorizer()
    model = load_model()
    feature_names = vectorizer.get_feature_names_out()

    if action == "Analyze CSV":
        uploaded = st.file_uploader("Upload CSV file", type='csv')
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Raw data", df.head())
            df['processed'] = df['content'].fillna('').apply(preprocess_text)
            X = vectorizer.transform(df['processed'])
            df['cluster'] = model.predict(X)
            st.write("Cluster assignments", df[['title', 'cluster']].head())
            st.bar_chart(df['cluster'].value_counts().sort_index())

            topn = st.slider("Top words per cluster", 5, 20, 10)
            for idx, center in enumerate(model.cluster_centers_):
                inds = center.argsort()[-topn:][::-1]
                words = [feature_names[i] for i in inds]
                st.write(f"Cluster {idx}:", ', '.join(words))
                wc = WordCloud(width=400, height=200).generate(' '.join(words))
                st.image(wc.to_array(), caption=f"WordCloud Cluster {idx}")

    else:
        text = st.text_area("Enter text to classify")
        if st.button("Predict topic") and text.strip():
            proc = preprocess_text(text)
            vec = vectorizer.transform([proc])
            pred = model.predict(vec)[0]
            st.write(f"Predicted cluster: {pred}")

            topn = st.number_input("Words to show", 5, 20, 10)
            center = model.cluster_centers_[pred]
            inds = center.argsort()[-topn:][::-1]
            words = [feature_names[i] for i in inds]
            st.write("Topic words:", ', '.join(words))

if __name__ == "__main__":
    main()