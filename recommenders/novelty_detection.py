import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import OneClassSVM

from recommenders.scoring import playtime_ratio_scorer


def fit_novelty_detector(user_data):
    """
    Returns a novelty detector fit to the given user data.
    The user data needs to be a Pandas `DataFrame` with at least the columns `desc` containing short descriptions for each game.
    and `playtime_forever` containing the total playtime the user has for each game.
    """

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('svd', TruncatedSVD(random_state=752)),
        ('novelty', OneClassSVM())
    ])

    # Tweak these!
    params = {
        'tfidf__ngram_range': list((1, n) for n in range(1, 4)),
        'tfidf__use_idf': [True, False],
        'svd__n_components': list(range(2, user_data.size//10)),
        'novelty__nu': list(n/10 for n in range(1, 10)),
        'novelty__gamma': list(n/10 for n in range(1, 10))
    }

    # The trick for the cv parameter here is to avoid cross validation
    clf = GridSearchCV(pipeline, params, n_jobs=-1,
                       scoring=playtime_ratio_scorer, cv=[(slice(None), slice(None))])
    clf.fit(user_data['desc'], user_data['playtime_forever'])
    return clf
