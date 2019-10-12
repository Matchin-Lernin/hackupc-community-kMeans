import pandas as pd
import numpy as np

#final = pd.read_csv('datasets/final/final.csv')
#metadata = pd.read_csv('datasets/final/metadata.csv')

# New Example
songsDb = pd.read_csv('datasets/final/newSongDb.csv')
metadata = pd.read_csv('datasets/final/metadataDB.csv')
input_user = pd.read_csv('datasets/final/input.csv')
#### Model Selection - K Means Algorithm
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

songsDb = shuffle(songsDb)
X = songsDb.loc[[i for i in range(0, 6000)]]
Y = input_user
X = shuffle(X)
Y = shuffle(Y)

# metadata = metadata.set_index('track_id')

print(X.head())
print(Y.head())
kmeans = KMeans(n_clusters=6)

def fit(df, algo, flag=0):
    if flag:
        algo.fit(df)
    else:
        algo.partial_fit(df)          
    df['label'] = algo.labels_
    return (df, algo)

def predict(t, Y):
    y_pred = t[1].predict(Y)
    mode = pd.Series(y_pred).mode()
    return t[0][t[0]['label'] == mode.loc[0]]

def recommend(recommendations, meta, Y):
    dat = []
    for i in Y['track_id']:
        dat.append(i)
    genre_mode = meta.loc[dat]['Name'].mode()
    return genre_mode

t = fit(X, kmeans, 1)
print("T1")
print(t[1])
print("T0")
print(t[0])
recommendations = predict(t, Y)
output = recommend(recommendations, metadata, Y)
genre_recommend = output
# Genre wise recommendations
print(genre_recommend.head())
# Artist wise recommendations
print(recommendations.head())


# #### Testing
testing = Y.iloc[6:12]['track_id']
testing
ids = testing.loc[testing.index]
songs = metadata.loc[testing.loc[list(testing.index)]]
songs
re = predict(t, Y.iloc[6:12])
output = recommend(re, metadata, Y.iloc[6:12])
ge_re = output
print("GE_RE")
print(ge_re.head())