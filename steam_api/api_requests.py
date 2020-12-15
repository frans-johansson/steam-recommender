import pandas as pd
import requests


def get_owned_games(user_id):
    """
    Returns a Pandas `Series` with all games owned by the given user. Index is the application ID's.
    May return empty data if an API error occurs.
    """
    url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key=8D0F87EEE7053D55B0A5ED8CD94D3202&steamid={user_id}&format=json"
    response = requests.request("GET", url)

    try:
        data = response.json()['response']['games']
        df = pd.DataFrame(data)
        df.set_index('appid', inplace=True)
        df = df['playtime_forever']
    except KeyError:
        data = []
        df = pd.Series(data)

    return df


def get_friends(user_id):
    """
    Returns a Pandas `Series` with all the friends' Steam ID's of the given user.
    """
    url =  f"http://api.steampowered.com/ISteamUser/GetFriendList/v0001/?key=8D0F87EEE7053D55B0A5ED8CD94D3202&steamid={user_id}&relationship=friend"
    response = requests.request("GET", url)
    
    friend_ids = []
    try:
        json_data = response.json()['friendslist']['friends']
        friend_ids = [friend['steamid'] for friend in json_data]
    except:
        pass

    return friend_ids


def get_game_description(game_id):
    """
    Returns the short description from the Steam store page for a given application ID.
    May return `None` if an API error occurs.
    """
    url = f"https://store.steampowered.com/api/appdetails/?appids={game_id}"
    response = requests.request("GET", url)

    try:
        # return response.json()[game_id]['data']['short_description']
        return response.json()[game_id]['data']
    except:
        return None