import random
import implicit
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

def make_train_test(data, sample_size):
    test_set = data.copy() # Test set is a copy of the original set
    test_set[test_set !=0] = 1 # Store test set as a binary preference matrix
    training_set = data.copy() # Training set also a copy that you can alter
    nonzero_inds = training_set.nonzero() # Fins indicies on the data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0],nonzero_inds[1])) #zippar ihop pare av nonzero
    random.seed(0) # Set the random seed to 0 for reproducibility
    num_samples = int(np.ceil(sample_size*len(nonzero_pairs))) # Round the amount of samples to nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user_item pair without replacements
    user_inds = [index[0] for index in samples] # Get user row indicies 
    games_inds = [index[1] for index in samples] # Get the item column indicies
    
    training_set[user_inds, games_inds] = 0 # Assign all of the random choosen user- & item pairs to 0
    training_set.eliminate_zeros() # Get rid of 0 in the sparse array storage after update to save space

    return training_set, test_set, list(set(user_inds)) # Output the unique list of game rows that were altered + places


def fit_ALS_model(iu_train, alpha=15):
    # id = user_id
    # users = [id] + get_friends(id)
    # owned = get_owned_games(id).index

    # users_dict = {user: get_owned_games(user) for user in users}
    # print(users_dict)
    # # Make data, transpose and a sparse matrix
    # ui_data = pd.DataFrame(users_dict).transpose().fillna(0.0)
    # all_games = ui_data.columns

    # #rec_data  = [get_game_description(str(game_id)) for game_id in rec_gameids]
    # ui_matrix = sparse.csr_matrix(ui_data.values) # Speltid axises are delete

    # [ui_train, iu_test, item_inds] = makeDataset(ui_matrix, 0.2)

    # iu_matrix = ui_train.T

    # ALS function thats already implemented in Python
    model = implicit.als.AlternatingLeastSquares(factors = 20,regularization= 0.1, iterations=50)
    model.fit((alpha*iu_train).astype('double'))

    return model

def get_ALS_recommendations(model, user_id, ui_train, n=10):
    # Recommender
    recs = model.recommend(userid=user_id, user_items=ui_train, N=n)
    recs_idxs = [list(r) for r in zip(*recs)][0]
    # rec_gameids = ui_data.columns[recs_idk]

    # rec_data  = [get_game_description(str(game_id)) for game_id in rec_gameids]


    # resultat = [data['name'] for data in rec_data if data is not None ] # Get the result recommended
    
    #Code for getting the masked games
    #games = [ui_data.columns[games] for games in item_inds]
    #rec_data2 = [get_game_description(str(i)) for i in games] # Get the games that were masked
   # resultat2 = [data['name'] for data in rec_data2 if data is not None ] 
    
    # ratio = getRatio(ui_data,owned,rec_gameids)
  
    return recs_idxs


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