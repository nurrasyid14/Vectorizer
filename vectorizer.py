# vectorizer.py

import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer as SklearnCountVectorizer
import pandas as pd
import re
import string
import math

class TF_IDF:
    def __init__(self, max_features=None, ngram_range=(1, 1), stop_words='english'):
        self.vectorizer = SklearnTfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words
        )
        self.feature_names = None

    def fit(self, X, y=None):
        # Join tokenized words back into strings for sklearn vectorizer
        X_str = [' '.join(words) for words in X]
        self.vectorizer.fit(X_str)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self

    def transform(self, X):
        X_str = [' '.join(words) for words in X]
        return self.vectorizer.transform(X_str)

    def fit_transform(self, X, y=None):
        X_str = [' '.join(words) for words in X]
        X_transformed = self.vectorizer.fit_transform(X_str)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return X_transformed

    def get_feature_names(self):
        return self.feature_names

    def get_params(self, deep=True):
        return {
            "max_features": self.vectorizer.max_features,
            "ngram_range": self.vectorizer.ngram_range,
            "stop_words": self.vectorizer.stop_words
        }


class BagofWords:
    def __init__(self, max_features=None, ngram_range=(1, 1), stop_words='english'):
        self.vectorizer = SklearnCountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words
        )
        self.feature_names = None

    def fit(self, X, y=None):
        X_str = [' '.join(words) for words in X]
        self.vectorizer.fit(X_str)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self

    def transform(self, X):
        X_str = [' '.join(words) for words in X]
        return self.vectorizer.transform(X_str)

    def fit_transform(self, X, y=None):
        X_str = [' '.join(words) for words in X]
        X_transformed = self.vectorizer.fit_transform(X_str)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return X_transformed

    def get_feature_names(self):
        return self.feature_names

    def get_params(self, deep=True):
        return {
            "max_features": self.vectorizer.max_features,
            "ngram_range": self.vectorizer.ngram_range,
            "stop_words": self.vectorizer.stop_words
        }


class WordEmbedding:
    def __init__(self, model):
        self.model = model
        self.dim = 300

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        return self.transform(X)
        
    def transform(self, X):
        return np.array([
            np.mean([self.model[w] for w in words if w in self.model]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
    def get_feature_names(self):
        return self.model.keys()

    def get_params(self, deep=True):
        return {"model": self.model}
