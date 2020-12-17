import pandas as pd
from steam_api.utils import id_from_url

# Make a DataFrame of the games available
games = None
index = None
try:
    games = pd.read_csv('data/steam_games.csv')
    games = games[['name', 'game_description', 'url', 'developer',
                   'publisher', 'popular_tags', 'game_details', 'genre', 'types']]
    games = games[games['types'] == 'app']
    games['id'] = games['url'].apply(lambda url: id_from_url(url))
    games = games.dropna(subset=['id'])
    games.sort_values('id', inplace=True)
    games.set_index('id', inplace=True)
    games.drop(['url', 'types'], axis=1, inplace=True)
    games.fillna('', inplace=True)

    index = pd.Series(games.index)
except:
    print('Failed to collect game data')

# Make a DataFrame of cleaned game descriptions available
try:
    descriptions = games['game_description']
    descriptions = descriptions.apply(
        lambda desc: ' '.join(str(desc).split()[3:]))
    descriptions = descriptions.apply(lambda desc: ' '.join(
        [word for word in desc.split() if word.isalpha()]))
except:
    print('Failed to create descriptions')
