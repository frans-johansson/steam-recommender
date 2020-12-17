import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer


def make_bags(data, bags):
    df = pd.DataFrame(columns=bags.keys(), index=data.index).fillna('')
    for key, cols in bags.items():
        df[key] = df[key].str.cat(data[cols], sep=',')

    return df


def calculate_similarities(bags, game_idxs, game_names):
    def _calculate_content_similarities(data, game_idx):
        return linear_kernel(data[game_idx, :], data)

    tf = CountVectorizer(stop_words='english')
    content_data = bags.apply(tf.fit_transform)
    content_data = content_data.apply(
        _calculate_content_similarities, game_idx=game_idxs)

    return pd.DataFrame.from_records(content_data, index=bags.columns, columns=game_names)


def get_recommendations(similarities, games, index, owned_idx, n=10):
    def _sort_no_owned(data):
        sorted = np.argsort(data)[::-1]
        not_owned = sorted[np.isin(sorted, owned_idx, invert=True)][:n]
        return games.loc[index[not_owned].values]['name'].values

    return similarities.applymap(_sort_no_owned)
