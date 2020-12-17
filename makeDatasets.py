from pandas import read_csv
import pathlib

# Get data from csv
location = pathlib.Path('steam-recommender/data/clean_Data.csv')
dataU = read_csv(location)

# Get 20% of random elements (comb. user-game) for test dataset
testU = dataU.sample(frac=0.2, replace= False)

# Get remaining data for training set
trainU = dataU[~dataU.isin(testU)].dropna()

# Output csv
testU.to_csv(pathlib.Path(r'steam-recommender/data/testU.csv'), index=False)
trainU.to_csv(pathlib.Path(r'steam-recommender/data/trainU.csv'), index =False)
