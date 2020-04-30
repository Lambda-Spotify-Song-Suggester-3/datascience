# Spotify-BW
# Data Science
Machine Learning Engineers:
- Objective: Build a model to recommend songs based on similarity to user input; create visualizations using song data
- Dataset: original dataset from Kaggle (https://www.kaggle.com/tomigelo/spotify-audio-features)

We experimented with several models before utilizing K-Nearest Neighbors to find which songs would best match with the user’s preference based on their selected input song. 

**Model 1 : A Neural Network Approach**  
*file : model.h5*

If you load our pretrained model, you'll be able to use a NN. This model was trained on the embeddings of the song and artist features. The weights extracted from the model are used to find similar songs. Problems with this model: We were not able to capture all the features in the data, and only trained on 10,000 songs. 

```
from keras.models import load_model
model = load_model('model.h5')
model.summary()
```


**Model 2 : Using Cosine Similarity**  
*file : cosine_similarity_recommender.py*

With the sklearn cosine_similarity() function we were able to visually see the relationship between each song in a matrix. By isolating one song (row) in the "Y" parameter, we were able to reduce RAM allowing us the ability to use the entire dataset of over 130k songs.

```
matrix = cosine_similarity(df, df[404:405)])
matrix = pd.DataFrame(matrix)
top = matrix[0].sort_values(ascending=False)[:10]
```

**Model 3 : Implementing K-Nearest Neighbors**  
*file : nearest_neighbors_recommender.py*

With over a hundred thousand songs to choose from, we decided to use the extremely fast KNN (K Nearest Neighbors) architecture to find the most similar songs. The song features literally decide which other songs are closest to it.

```
nn = NearestNeighbors(n_neighbors=10, algorithm='kd_tree')
nn.fit(df)
neighbor_predictions = nn.kneighbors([df[404]])
```

The file above contains two helper functions used to calculate a playlist of ten similar songs ```epic_predictor(input_track_key)``` and return the average of eight features from the  generated playlist ```feature_average(input_track_key)```.
