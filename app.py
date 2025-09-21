from vectorizer import TF_IDF
from vectorizer import BagofWords as bow
from vectorizer import WordEmbedding as we
import streamlit as st
import numpy as np
import pandas as pd

# Dummy embedding model for simulation
class DummyEmbeddingModel(dict):
    def __getitem__(self, key):
        # Simulate embedding lookup by returning a random vector
        return np.random.rand(300)

# Streamlit UI
st.title("Text Vectorizer Demo")

# Input text area
user_input = st.text_area("Enter some text (separate documents with newlines):")

# Select vectorizer
vectorizer_choice = st.selectbox(
    "Choose a vectorizer:",
    ("TF-IDF", "Bag of Words", "Word Embedding")
)

if st.button("Transform"):
    if not user_input.strip():
        st.warning("Please enter some text first!")
    else:
        # Split user input into documents
        documents = [doc.split() for doc in user_input.strip().split("\n")]

        if vectorizer_choice == "TF-IDF":
            vect = TF_IDF(max_features=10)
            X = vect.fit_transform(documents)
            st.write("**Feature Names:**", vect.get_feature_names())
            st.write("**TF-IDF Matrix (sparse shown as dense):**")
            df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
            st.dataframe(df)
            
        elif vectorizer_choice == "Bag of Words":
            vect = bow(max_features=10)
            X = vect.fit_transform(documents)
            st.write("**Feature Names:**", vect.get_feature_names())
            st.write("**BoW Matrix (sparse shown as dense):**")
            df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
            st.dataframe(df)

        elif vectorizer_choice == "Word Embedding":
            dummy_model = DummyEmbeddingModel()
            vect = we(dummy_model)
            X = vect.transform(documents)
            st.write("**Embedding Shape:**", X.shape)
            st.write("**Embeddings (first 5 vectors):**")
            st.dataframe(X[:5])