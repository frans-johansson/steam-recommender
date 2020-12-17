import pandas as pd
import json 
import requests
import implicit
import pathlib
import scipy
import sklearn 
import random
import numpy as np
from scipy import sparse
from steam_api.api_requests import get_owned_games, get_friends, get_game_description

def make_training(data, sample_size):
    #vi ska ersätta 20% av värderna som  ej är noll med noll och spara dom i test
    test_set = data.copy() # Test set is a copy of the original set
    test_set[test_set !=0] = 1 # Store test set as a binary preference matrix
    training_set = data.copy() # Training set also a copy that you can alter
    nonzero_inds = training_set.nonzero() # Fins indicies on the data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0],nonzero_inds[1])) #zippar ihop pare av nonzero
    random.seed(0) # Set the random seed to 0 for reproducibility
    num_samples = int(np.ceil(sample_size*len(nonzero_pairs))) # Round the amount of samples to nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user_item pair without replacements
    #print("samples")
    #print(samples)
    user_inds = [index[0] for index in samples] # Get user row indicies 
    games_inds = [index[1] for index in samples] # Get the item column indicies
    #print("user inds")
    
    training_set[user_inds, games_inds] = 0 # Assign all of the random choosen user- & item pairs to 0
    training_set.eliminate_zeros() # Get rid of 0 in the sparse array storage after update to save space

    return training_set, test_set, list(set(games_inds),list(set(games_inds))) # Output the unique list of game rows that were altered + places

def get_ratio(test_data,users_games,rec_games):
    all_games = test_data.columns
    tot_recommends = list(set(all_games).intersection(rec_games))
    user_games_in_test = list(set(all_games).intersection(users_games))
    
    ratio = len(tot_recommends)/len(user_games_in_test)

    return ratio

    
def als_make_model(user_id):
    id = user_id
    users = [id] + get_friends(id)
    owned = get_owned_games(id).index

    users_dict = {user: get_owned_games(user) for user in users}
    print(users_dict)
    # Make data, transpose and a sparse matrix
    ui_data = pd.DataFrame(users_dict).transpose().fillna(0.0)
    all_games = ui_data.columns

    #rec_data  = [get_game_description(str(game_id)) for game_id in rec_gameids]
    ui_matrix = sparse.csr_matrix(ui_data.values) # Speltid axises are delete

    [ui_train,iu_test, item_inds] = makeDataset(ui_matrix,0.2)

    iu_matrix = ui_train.T

    # ALS function thats already implemented in Python
    alpha = 15
    model = implicit.als.AlternatingLeastSquares(factors = 20,regularization= 0.1, iterations=50)
    model.fit((alpha*iu_matrix).astype('double'))
    alpha = 15 

    return (model,ui_train,ui_data,owned)

     
def als_recommend(model,user_id,ui_train,ui_data):
    
    # Recommender
    recs = model.recommend(userid=user_id, user_items = ui_train, N=10)
    recs_idk = [list(r) for r in zip(*recs)][0]
    rec_gameids = ui_data.columns[recs_idk]

    rec_data  = [get_game_description(str(game_id)) for game_id in rec_gameids]


    resultat = [data['name'] for data in rec_data if data is not None ] # Get the result recommended
    
    #Code for getting the masked games
    #games = [ui_data.columns[games] for games in item_inds]
    #rec_data2 = [get_game_description(str(i)) for i in games] # Get the games that were masked
   # resultat2 = [data['name'] for data in rec_data2 if data is not None ] 
    
    ratio = getRatio(ui_data,owned,rec_gameids)
  
    return (resultat, rec_gameids, ratio)

id = '76561198003578578'
(res,rec_gameids,ratio) = als_recommend(id)
print(res)
print(ratio)