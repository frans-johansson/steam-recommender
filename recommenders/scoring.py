import pandas as pd

def get_ratio(test_data, users_games, rec_games):
    all_games = test_data.columns
    tot_recommends = list(set(users_games).intersection(rec_games))
    user_games_in_test = list(set(all_games).intersection(users_games))
    
    ratio = len(tot_recommends)/len(user_games_in_test)

    return ratio