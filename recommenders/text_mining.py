import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class LSA:

    def __init__(self, descriptions, k=100):
        self.tfidf = TfidfVectorizer(stop_words='english', use_idf=True)
        self.svd = TruncatedSVD(n_components=k)
        self.nlp = Pipeline([
            ('tfidf', self.tfidf),
            ('svd', self.svd)
        ])

        self.LSA_space = self.nlp.fit_transform(descriptions)

    def make_query(self, descriptions, games, index, owned_games, n=10):
        query = self.nlp.transform(descriptions)
        similarities = cosine_similarity(self.LSA_space, query)

        df = pd.DataFrame.from_records(similarities)
        df = df.apply(np.argsort).iloc[::-1]
        df = df.transpose()
        df = df.apply(
            lambda idx: games.loc[index[idx].values]['name'].values, axis=1)
        return df.apply(lambda games: games[np.isin(games, owned_games, invert=True)][:10])
