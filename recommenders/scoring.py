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
