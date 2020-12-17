import pandas as pd


def playtime_ratio_scorer(estimator, X, y):
    """
    For use with the novelty detector in this module.
    Will attempt to partition the novelty predictions such that the playtime `y` in the inliers is consistently high and consistently low in the outliers
    """
    pred = estimator.fit_predict(X)
    inliers = y[pred > 0]
    outliers = y[pred < 0]
    return inliers.mean() / (inliers.std() * outliers.mean())


def get_ratio(test_data, users_games, rec_games):
    all_games = test_data.columns
    tot_recommends = list(set(all_games).intersection(rec_games))
    user_games_in_test = list(set(all_games).intersection(users_games))
    
    ratio = len(tot_recommends)/len(user_games_in_test)

    return ratio