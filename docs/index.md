![header](/assets/header.png)

## Introduction

Hello! This tutorial blog will walk you through the steps to implement your very own recommender system. In our case, the recommender system has been designed to work with the [Steam gaming platform][steam], but the ideas could really be applied to any platform you can imagine!

First and foremost, we believe some introductions are in order. We are three Master's degree students in Media Technology at Linköping University, and we are the ones taking you on this journey down recommender system lane:

- Frans Johansson, [@frans-johansson](https://github.com/frans-johansson/)
- Jonas Bertilsson, [@femtiolapp](https://github.com/femtiolapp/)
- Moa Gutenwik, [@gutenwik](https://github.com/gutenwik/)

This project was part of the course TNM108, Machine Learning for Social Media, at Linköping University during the fall of 2020.

### So what is a recommender system?

A recommender system is a sub-class of information filtering system that seeks to predict the rating or preference a user would give an item. Most big companies you know of (including Valve, the company behind the Steam platform) already use them to great effect in order to provide their user-base with a continous feed of relevant items, e.g. games, movies, songs, to consume. With that said, the purpose of this project was not to innovate, but rather to explore the problem of constructing a recommender system from a data scientific point of view. And hopefully, by reading this, you will be equipped with the knowledge and ideas to go out there and make something neat of your own.

![The general gist of recommendation systems](/assets/recomm_illustration.png)

The types of recommender systems we will examine can be categorized as either *content-based* methods or *collaborative filtering* methods. The former considers various descriptors of the items the users might be interested in, e.g. genre, developer or description, while the latter considers item-user interactions with the main idea being "similar users like similar items". These ideas will be elaborated further in their corresponding sections of this blog.

With all this in mind, we are ready to delve into the meat of the problem: what do we need to start constructing our recommender systems?

## Libraries and other prerequisites

This project was implemented in the Python programming language. In order for you to follow along, or test our code which you can find publically available in the [repository][repository], we strongly recommend setting up a virtual Python environment ([venv][python venv]). The dependencies you need are all recorder in the *requirements.txt* file, and you can easily install them by running `pip install -r requirements.txt` from a terminal in the project directory.

The libraries *Pandas* and *NumPy* will be used throughout most of the examples in the remainder of this blog. Naturally, these have been imported as

```python
import numpy as np
import pandas as pd
```

but for sake of brevity we will not append these lines of code to every example. Just know that they are always there!

## The data

*Data from the Steam CSV file*

```python
def id_from_url(url):
    found = re.search('app\/(\d+)', url)
    if found:
        return found.group(1)
    return None


steam_games = None
steam_index = None
try:
    steam_games = pd.read_csv('data/steam_games.csv')
    steam_games = steam_games[['name', 'game_description', 'url', 'developer',
                   'publisher', 'popular_tags', 'game_details', 'genre', 'types']]
    steam_games = steam_games[steam_games['types'] == 'app']
    steam_games['id'] = steam_games['url'].apply(lambda url: id_from_url(url))
    steam_games = steam_games.dropna(subset=['id'])
    steam_games.sort_values('id', inplace=True)
    steam_games.set_index('id', inplace=True)
    steam_games.drop(['url', 'types'], axis=1, inplace=True)
    steam_games.fillna('', inplace=True)

    steam_index = pd.Series(steam_games.index)
except:
    print('Failed to collect game data')

steam_descriptions = None
try:
    steam_descriptions = steam_games['game_description']
    steam_descriptions = steam_descriptions.apply(
        lambda desc: ' '.join(str(desc).split()[3:]))
    steam_descriptions = steam_descriptions.apply(lambda desc: ' '.join(
        [word for word in desc.split() if word.isalpha()]))
except:
    print('Failed to create descriptions')

```

*Methods for interfacing with the API*

```python
def get_owned_games(user_id):
    url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key=8D0F87EEE7053D55B0A5ED8CD94D3202&steamid={user_id}&format=json"
    response = requests.request("GET", url)

    try:
        data = response.json()['response']['games']
        df = pd.DataFrame(data)
        df['appid'] = df['appid'].astype('string')
        df.set_index('appid', inplace=True)
        df = df['playtime_forever']
    except KeyError as e:
        data = []
        df = pd.Series(data)

    return df
```

```python
def get_friends(user_id):
    url = f"http://api.steampowered.com/ISteamUser/GetFriendList/v0001/?key=8D0F87EEE7053D55B0A5ED8CD94D3202&steamid={user_id}&relationship=friend"
    response = requests.request("GET", url)

    friend_ids = []
    try:
        json_data = response.json()['friendslist']['friends']
        friend_ids = [friend['steamid'] for friend in json_data]
    except:
        print(f'Failed to get friends for user {user_id}')
        pass

    return friend_ids
```

```python
def get_game_data(game_id):
    url = f"https://store.steampowered.com/api/appdetails/?appids={game_id}"
    response = requests.request("GET", url)

    try:
        return response.json()[game_id]['data']
    except:
        print(f'Failed to get game data for {game_id}')
        return None
```

*User ID of some random individual found on the API docummentation site*

```python
id = '76561197960434622'
```

## Content-based recommendations

The main idea of content-based recommender  systems is that the items, in this case the games, are considered in terms of their features in the data. As one might suspect, this is quite the umbrella-term as what these exact features are depends heavily on what is available in whatever data set you are using.  For example, a music track could be thought of in terms of anything from genre and artist to the actual textual contents of the song’s lyrics. For our content-based recommender system we will focus our attention on two main sets of features for the games: META information about the game such as genre, developer and user-added tags; and the detailed game description shown on its Steam store page.

For both of these methods, we will need some data for a given user. The [previous section](#the-data) explains the methods and sources we use to collect data, and the code snippet below shows how these methods are used to gather the data we need in these methods:

```python
user_library = get_owned_games(id)
owned_in_store = user_library.index.isin(steam_games.index)
user_library = user_library[owned_in_store].sort_values(ascending=False)
owned_idx = [steam_games.index.get_loc(idx) for idx in user_library.index]
```

### META information

For the META information about each game, we will limit ourselves to the genre, developer, publisher, user-added tags and game details. Here we thought it would be interesting to see if any particular combination of these data would yield better or worse results -- so to test this, we constructed a few collections, or bags, of different combinations of the data.

```python
def make_bags(data, bags):
    df = pd.DataFrame(columns=bags.keys(), index=data.index).fillna('')
    for key, cols in bags.items():
        df[key] = df[key].str.cat(data[cols], sep=',')

    return df


bags = make_bags(steam_games, {
    'dev_tags':             ['developer', 'popular_tags'],
    'pub_tags':             ['publisher', 'popular_tags'],
    'tags_details_genre':   ['popular_tags', 'game_details', 'genre'],
    'all':                  ['developer', 'publisher', 'popular_tags', 'game_details', 'genre']
})
```

Next we need some way of representing this data in a way that allows us to compare the games to each other. Since we are working with textual data, a good approach is to vectorize the data using something like the `CountVectorizer` from sklearn.

```python
from sklearn.feature_extraction.text import CountVectorizer
tf = CountVectorizer(stop_words='english')
content_data = bags.apply(tf.fit_transform)
```

This leaves us with each game in the Steam store data set being represented by a vector where each dimension corresponds to the count of some genre, developer, user-added tag etc. for the given game. Now vectors are things we easily compare with each other! One fairly common way of comparing vectors in high-dimensional spaces such as these, is to use the [cosine similarity][cosine similarity wikipedia] which measures the angle between two vectors using their inner product and normalizing this by the vector norms. We will do something similar here and only compute the inner product as suggested by [this handy article][content-based article]. This can be efficiently performed by the `linear_kernel` transformation from sklearn. To generate suggestions to a user we will now first select a game they have already played and enjoyed. Since this kind of data is not directly available to us, we will settle on selecting one of their games with highest total playtime; in reality, a company would likely implement more sophisticated algorithms for selecting which games to genereate the recommendations from, we will simply select the user’s top played games for this example. This chosen game is then compared to every other game in the data set using the `linear_kernel` transformation, and the five games with highest similarity that are not already owned by the user are selected as recommendations.

```python
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer

def calculate_similarities(bags, game_idxs, game_names):
    def _calculate_content_similarities(data, game_idx):
        return linear_kernel(data[game_idx, :], data)

    tf = CountVectorizer(stop_words='english')
    content_data = bags.apply(tf.fit_transform)
    content_data = content_data.apply(
        _calculate_content_similarities, game_idx=game_idxs)

    return pd.DataFrame.from_records(content_data, index=bags.columns, columns=game_names)


def get_recommendations(similarities, games, index, owned_idx, n=5):
    def _sort_no_owned(data):
        sorted = np.argsort(data)[::-1]
        not_owned = sorted[np.isin(sorted, owned_idx, invert=True)][:n]
        return games.loc[index[not_owned].values]['name'].values

    return similarities.applymap(_sort_no_owned)


most_played_idx = owned_idx[:3]
most_played_names = steam_games.iloc[most_played_idx]['name']
similarities = calculate_similarities(bags, most_played_idx, most_played_names)

get_recommendations(similarities, steam_games, steam_index, owned_idx)
```

|Bags              |Factorio                                                                   |Clicker Heroes                                                                                                                                                                            |The Witcher® 3: Wild Hunt                                                                                                                                                                                                   |
|------------------|---------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|dev_tags          |['Imagine Earth' 'The Universim' 'After the Collapse' 'Avorion' 'Embark']  |['Talisman: Digital Edition' 'Mitos.is: The Game' 'Voodoo Garden'  'Click Legends' 'Business Tour - Board Game with Online Multiplayer']                                                  |['The Witcher 3: Wild Hunt - Blood and Wine'  'The Witcher 2: Assassins of Kings Enhanced Edition' 'Gothic 1'  'Fable - The Lost Chapters' 'The Witcher 3: Wild Hunt - Expansion Pass']                                     |
|pub_tags          |['Imagine Earth' 'The Universim' 'After the Collapse' 'Avorion' 'Embark']  |['Mitos.is: The Game' 'Talisman: Digital Edition' 'Click Legends'  'Voodoo Garden' 'Kingdom Rush']                                                                                        |['The Witcher 3: Wild Hunt - Blood and Wine'  'The Witcher 2: Assassins of Kings Enhanced Edition'  'The Witcher 3: Wild Hunt - Expansion Pass'  'The Witcher 3: Wild Hunt - Hearts of Stone' 'Fable - The Lost Chapters']  |
|tags_details_genre|['Barotrauma' 'CounterAttack' 'Zaccaria Pinball' 'StarMade' 'BrainBread 2']|['scram' 'Business Tour - Board Game with Online Multiplayer'  'Insanity Clicker' 'God Awe-full Clicker' 'Learn to Fly 3']                                                                |['The Witcher 3: Wild Hunt - Blood and Wine'  'The Witcher 2: Assassins of Kings Enhanced Edition'  "Dragon's Dogma: Dark Arisen" 'Darksiders II Deathinitive Edition'  'Lords Of The Fallen™']                             |
|all               |['Barotrauma' 'CounterAttack' 'Zaccaria Pinball' 'StarMade' 'BrainBread 2']|['scram' 'Clicker Heroes: Boxy & Bloop Auto Clicker'  'Clicker Heroes: Zombie Auto Clicker'  'Clicker Heroes: Unicorn Auto Clicker'  'Business Tour - Board Game with Online Multiplayer']|['The Witcher 2: Assassins of Kings Enhanced Edition'  'The Witcher 3: Wild Hunt - Blood and Wine'  'The Witcher 3: Wild Hunt - Hearts of Stone'  'The Witcher 3: Wild Hunt - Expansion Pass' "Dragon's Dogma: Dark Arisen"]|

### Game descriptions

The second content-based approach for recommendations will use the detailed game description from the Steam store page as the features which describe the contents of a given game. The detailed description is extracted for each game in the data set, and this time we vectorize it using sklearn’s `TfIdfVectorizer`. This differs from the `CountVectorizer` in the last example in that [TF-IDF weighting][tf-idf wikipedia] is applied to the elements of the vectors, which will help highlight more significant terms in distinguishing between the contents of all game descriptions. Since there are many more terms when looking at game descriptions as opposed to simple labels like genre and developer, there will likely be much more noise in the description vectors. To combat this issue, we decided to apply a common dimensionality reduction method called [Latent Semantic Analysis][lsa towardsdatascience] (LSA), which in concept is quite similar to something like Principal Component Analysis (PCA), since the aim is to find hidden (latent) patterns across all game descriptions. LSA can be performed quite easily with the `TruncatedSVD` from sklearn on data which has been preprocessed with the `TfIdfVectorizer`. As such, a `Pipeline` of the two models was created and fit to the corpus of all game descriptions in the Steam store data set.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class LSA:

    def __init__(self, descriptions, k=100):
        self.tfidf = TfidfVectorizer(stop_words='english', use_idf=True)
        self.svd = TruncatedSVD(n_components=k)
        self.nlp = Pipeline([
            ('tfidf', self.tfidf),
            ('svd', self.svd)
        ])

        self.LSA_space = self.nlp.fit_transform(descriptions)

    def make_query(self, descriptions, games, index, owned_games, n=5):
        query = self.nlp.transform(descriptions)
        similarities = cosine_similarity(self.LSA_space, query)

        df = pd.DataFrame.from_records(similarities)
        df = df.apply(np.argsort).iloc[::-1]
        df = df.transpose()
        df = df.apply(
            lambda idx: games.loc[index[idx].values]['name'].values, axis=1)
        return df.apply(lambda games: games[np.isin(games, owned_games, invert=True)][:n])
```

With the game descriptions vectorized we are now able to generate recommendations! The gist is very similar to the previous example, however we decided on using the `cosine_similarity` from sklearn for computing similarities instead of the `linear_kernel`. Again, a few games are selected from the user’s most played games and their vectorized descriptions are compared with every other game’s description. The five most similar games not already in the user’s library are then selected as recommendations.

```python
lsa = LSA(steam_descriptions)
most_played_desc = steam_descriptions.iloc[most_played_idx]
owned_names = steam_games.loc[user_library.index]['name']

lsa.make_query(most_played_desc, steam_games, steam_index, owned_names).to_frame().set_index(most_played_names)
```

|Game              |Recommendations                                                            |
|------------------|---------------------------------------------------------------------------|
|Factorio          |['Spiritlands' 'GameGuru' 'Blockscape' 'A列車で行こう8' 'SubmarineCraft']        |
|Clicker Heroes    |['Fantasy Hero Manager' 'Hero Go' 'Flamebound' 'Golden Axe III'  'Super Dungeon Tactics']|
|The Witcher® 3: Wild Hunt|['Dinosaurs Prehistoric Survivors' 'Far Cry® 4' 'Balrum' 'Vulpine'  'Vietcong']|

## Collaborative filtering

As we mentioned in the introduction, the second general class of recommender system we will take a look at are known as *collaborative filtering methods*. These are set apart from the previous class of content-based methods in that they consider interactions between a set of users and items, and try to find recommendations to one user based on what items other similar users have already interacted with. A simple way to view this is how two people, after discovering they share a similar taste in music, might recommend artists or albums the other person hasn't listened to yet.

![A simple example of how collaborative filtering works](/assets/collaborative_example.png)

A problem we quickly run into though, is how we should quantify how much a given user has enjoyed a game they played. In other scenarios, it may be possible to extract explicit feedback from a user, e.g. ratings on a movie or reviews of a restaurant, which can let us score the user's preferences. Indeed, the Steam platform allows users to review and rate games they have played; however, this information was not easilly accsessible on a per-user basis via the web API. What we do have available though, is the user's total playtime of a game. This allows us to consider a different type of feedback, namely *implicit feedback*. While this will not be as accurate of a reflection of the user's actual preferences, it will work better with the data from the API. Naturally, using a predefined data set with as many user-item interactions as possible would work best here, but since we really wanted the added element of interactivity brought by using live data from the API, we settled on the user-item interactions for one user and their friends.

[This article][gentle introduction] provides a nice introduction to using implicit feedback data in collaborative filtering systems, based on research papers on the subject; and [this blog][game recommendation system], which we have taken a great deal of inspiration from, provides some additional implememtation and algorithm ideas. The main problem here, it turns out, is approximating a sparse user-item matrix through a full matrix decomposition such that predicted user-item interactions can be obtained via the dot product of the two matrix factors.

![Approximate decomposition of a user-item matrix](/assets/matrix_decomp.png)

This type of prediction model (commonly referred to as a *matrix factorization model*) has proven to be quite effective when working with implicit feedback. This was demonstrated in the [Netflix Prize competition][netflix prize], as discussed in [this article][recommender systems netflix].

In less mathematical terms, what we want to accomplish is finding an approximation of the original user-item data where the missing values, i.e. the items that some user has not interacted with, are predicted based on all other known values. The two aforementioned resources suggest two main algorithms for accomplishing this: the Alternating Least Squares (ALS) algorithm and the SVD through Gradient Descent (GD) algorithm. Both of which we will take a look at here!

### The ALS algorithm

The ALS algorithm is available in Python through the [`implicit` library](https://github.com/benfred/implicit), which we will install with `pip install implicit`, making sure that we are sourced into our *venv* (this is all in the *requirements.txt*, so as anyone following along thus far should be fine). This simplifies our work significantly, as the majority of code we need to write ourselves has to do with data formatting.

We will first collect the necessary user-item data using our API helper functions `get_friends(id)` which returns the the friends of some user as a list of user IDs, and `get_owned_games(id)` which gets us total playtime data for the games owned by a given user. The resulting `DataFrame` will contain a large bit of *NaN* values which are imputed with the value `0.0`. Any columns (corresponding to items) that have no non-zero data are also removed (this might happen due to users owning a game without having any hours played, which in the context of implicit feedback just amounts to superfluous data).

```python
def make_user_item_data(id):
    users = get_friends(id) + [id]
    users_dict = {user: get_owned_games(user) for user in users}

    user_item_df = pd.DataFrame(users_dict).transpose()
    user_item_df.fillna(0.0, inplace=True)
    user_item_df = user_item_df[(user_item_df.T != 0.0).any()]
    return user_item_df


ui_data = make_user_item_data(id)
ui_data.iloc[:5, :5]
```

|User             |10   |100 |10000|1000010|1000030|
|-----------------|-----|----|-----|-------|-------|
|76561197960265754|1.0  |0.0 |0.0  |0.0    |0.0    |
|76561197960381818|432.0|18.0|0.0  |0.0    |0.0    |
|76561197960794555|281.0|0.0 |0.0  |0.0    |0.0    |
|76561197962146232|0.0  |0.0 |0.0  |0.0    |0.0    |
|76561197963135603|1.0  |0.0 |0.0  |0.0    |0.0    |

Next we would prefer to move from a Pandas `DataFrame` representation of the data to a sparse matrix representation which is what `implicit` needs. The library SciPy has a `sparse` module for dealing with this. Lastly, before training the ALS model, we will mask some of the known data out in order to create training and testing sets. This idea, and the function `make_train_test`, is explained in more detail under [Evaluating results](#evaluating-results).

```python
from scipy.sparse import csr_matrix
ui_mat = csr_matrix(ui_data)
ui_train, ui_test, u_to_test = make_train_test(ui_mat, 0.1)
```

Now we are ready to use the `implicit` library to train an ALS model, which we will then be able to use to generate recommendations for any of the users in the `ui_data` `DataFrame`. We have wrapped this functionality in some helper functions to make the final scripts a bit neater. The parameter `alpha` in the `fit_ALS_model` acts as a linear scaling factor for the implicit rating data, and is used to obtain *confidence values* used for training.

```python
def fit_ALS_model(iu_train, alpha=15):
    model = implicit.als.AlternatingLeastSquares(factors = 20,regularization= 0.1, iterations=50)
    model.fit((alpha*iu_train).astype('double'))

    return model


def get_ALS_recommendations(model, user_id, ui_train, n=10):
    recs = model.recommend(userid=user_id, user_items=ui_train, N=n)
    recs_idxs = [list(r) for r in zip(*recs)][0]
  
    return recs_idxs


als_model = fit_ALS_model(ui_train.T)
# Here the user_id is somewhat confusingly the index as opposed to a Steam user ID
# Since the ID of the user was appended to the end of the list, they should have the last index
rec_idxs = get_ALS_recommendations(als_model, ui_data.shape[0]-1, ui_train, n=10)
# Index data is translated to game ID data
rec_ids = ui_data.columns[rec_idxs]

# Collects complete game data from the API
rec_data = [get_game_data(rec_id) for rec_id in rec_ids]
# Only try to show names of successfully collected games
[game['name'] for game in rec_data if game is not None]
```

Which gives us the following output:

```text
['Hyper Light Drifter',
 "Assassin's Creed® Revelations",
 'Tropico 5',
 'Crypt of the NecroDancer',
 'Faerie Solitaire Remastered',
 'DOOM',
 'Far Cry 3 - Blood Dragon',
 'Shadow Warrior',
 'Eufloria',
 'Star Wars: Battlefront 2 (Classic, 2005)']
```

In order to evaluate this model, the results are computed for all users specified in `u_to_test` which we obtained when creating the training and testing data.

### The SVD with GD algorithm

We implemented the gradient descent algorithm by following along with the aforementioned [blog about recommender systems][game recommendation system]. There is a fairly well-known Python library called [Scikit Surprise][surprise] which implements many useful algorithms for recommender systems; however, Surprise does not support implicit or content-based data, which happens to be exactly what this project has explored. Following along with the implementation of a gradient-descent-powered SVD approximator, we ended up with the follwing function:

```python
def SVD_gradient_descent(ui_data, k=5):
    leading_components = k

    # Setting matricies
    Y = ui_data.copy()
    I = Y.copy()
    for col in I.columns:
        I[col] = I[col].apply(lambda x: 1 if x > 0 else 0)
        
    U = np.random.normal(0, 0.01, [I.shape[0], leading_components])
    V = np.random.normal(0, 0.01, [I.shape[1], leading_components])

    #Squared error functions
    def dfu(U):
        return np.dot((2*I.values*(np.dot(U, V.T)-Y.values)), V)
    def dfv(V):
        return np.dot((2*I.values*(np.dot(U, V.T)-Y.values)).T, U)

    #Gradient descent algorithm
    N = 500
    alpha = 0.001
    pred = np.round(np.dot(U, V.T), decimals=2)

    for _ in range(N):
        U = U - alpha*dfu(U)
        V = V - alpha*dfv(V)
        pred = np.round(np.dot(U, V.T), decimals=2)

    return pred
```

This function also takes user-item training data, but unlike the ALS model this needs to be a `DataFrame`. Thankfully, Pandas has got us covered here, as converting our sparse `ui_train` matrix to a `DataFrame` is really simple!

```python
ui_train_data = pd.DataFrame.sparse.from_spmatrix(ui_train)
```

Next, we implemented a function to genereate recommendations to a user given the predicted implicit feedback from `SVD_gradient_descent`.

```python
def get_EM_recommendations(predicted_data, user, owned, n=10):
    ranked_data = predicted_data.rank(axis=1, pct=True)
    return ranked_data.loc[user].drop(owned).sort_values(ascending=False)[:n].index
```

The idea here, according to the [source blog][game recommendation system], is that playtime for a given game across a large collection of users can be observed as roughly distributed according to a combination some number of normal distributions. For example, they have considered five such normal distributions which in a sense will correspond to a typical five-star rating system. Naturally, this idea plays out way better when you have a substantial amount of user-item data, which admittedly just a single user and their friends might not be able to provide.

![Example from the source blog of how playtime ratings can be inferred from multiple normal distributions](/assets/inferred_ratings.png)

Keeping this idea in mind, `get_EM_recommendations` will emulate this line of thinking without actually implementing any normal distributions. Instead, the predicted implicit rating for a user is considered in relation with all other users in terms of its *percentile rank*. The idea being that higher percentile rank corresponds to that user being in a higher inferred rating distribution (i.e. likely to give a five-star rating in the figure above). Our recommendations are then simply the games with highest percentile rank that the user does not already own.

```python
ui_pred = SVD_gradient_descent(ui_train_data)
ui_pred_data = pd.DataFrame(ui_pred, columns=ui_data.columns, index=ui_data.index)
# Recommendations are already Steam game IDs 
rec_ids = get_EM_recommendations(ui_pred_data, id, user_library.index)

# Collects complete game data from the API
rec_data = [api.get_game_data(rec_id) for rec_id in rec_ids]
# Only try to show names of successfully collected games
[game['name'] for game in rec_data if game is not None]
```

Which gives us the following output, where we can see that the API fails to get any results for one of the games (ah, the joys of live data):

```text
Failed to get game data for 10000
['Counter-Strike: Condition Zero',
 'Crown Trick',
 'Cook, Serve, Delicious! 3?!',
 'Zengeon',
 'WRATH: Aeon of Ruin',
 'Angry Birds VR: Isle of Pigs',
 'BoneCraft',
 'Tower Behind the Moon',
 'Chronicon Apocalyptica']
```

Similarly to the ALS method, we generate recommendations for all users in `u_to_test` in order to evaluate the model.

## Evaluating results

## Conclusion

[steam]: https://store.steampowered.com/ "Steam store page"
[repository]: https://github.com/frans-johansson/steam-recommender "Project repository with complete code base"
[python venv]: https://docs.python.org/3/library/venv.html "Python docummentaton for setting up a virtual environment"
[content-based article]: https://medium.com/analytics-vidhya/content-based-recommender-systems-in-python-2b330e01eb80 "Content-based recommender systems in Python"
[cosine similarity wikipedia]: https://en.wikipedia.org/wiki/Cosine_similarity "Wikipedia article about the cosine similarity"
[tf-idf wikipedia]: https://en.wikipedia.org/wiki/Tf%E2%80%93idf "Wikipedia article about TF-IDF weighting"
[lsa towardsdatascience]: https://towardsdatascience.com/latent-semantic-analysis-intuition-math-implementation-a194aff870f8 "Article describing LSA using the SVD in text-mining applications"
[gentle introduction]: https://jessesw.com/Rec-System/ "A Gentle Introduction to Recommender Systems with Implicit Feedback"
[game recommendation system]: https://audreygermain.github.io/Game-Recommendation-System/ "Recommendation System for Steam Game Store: An overview of recommender systems"
[surprise]: http://surpriselib.com/ "A Python scikit for recommender systems"
[netflix prize]: https://en.wikipedia.org/wiki/Netflix_Prize "Wikipedia article about the Netlix Prize competition"
[recommender systems netflix]: https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf "Matrix Factorization Techniques for Recommender Systems"
