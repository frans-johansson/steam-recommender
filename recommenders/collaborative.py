import numpy as np
import pandas as pd

from steam_api.api_requests import get_friends, get_owned_games

def make_user_item_data(id):
    users = get_friends(id) + [id]
    users_dict = {user: get_owned_games(user) for user in users}

    user_item_df = pd.DataFrame(users_dict).transpose()
    user_item_df.fillna(0.0, inplace=True)
    user_item_df = user_item_df[(user_item_df.T != 0.0).any()]
    return user_item_df


def SVD_gradient_descent(ui_data, k=5):
    # SVD with gradient descent to "fill in the blanks" in our user-item interactions
    leading_components = k

    # Setting matricies
    Y = ui_data.copy()
    I = Y.copy()
    for col in I.columns:
        I[col] = I[col].apply(lambda x: 1 if x > 0 else 0)
        
    U = np.random.normal(0, 0.01, [I.shape[0], leading_components])
    V = np.random.normal(0, 0.01, [I.shape[1], leading_components])

    #Squared error
    def dfu(U):
        return np.dot((2*I.values*(np.dot(U, V.T)-Y.values)), V)
    def dfv(V):
        return np.dot((2*I.values*(np.dot(U, V.T)-Y.values)).T, U)

    #Gradient descent
    N = 500
    alpha = 0.001
    pred = np.round(np.dot(U, V.T), decimals=2)

    for _ in range(N):
        U = U - alpha*dfu(U)
        V = V - alpha*dfv(V)
        pred = np.round(np.dot(U, V.T), decimals=2)

    return pred


def get_EM_recommendations(predicted_data, user, owned, n=10):
    ranked_data = predicted_data.rank(axis=1, pct=True)
    return ranked_data.loc[user].drop(owned).sort_values(ascending=False)[:n].index