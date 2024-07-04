import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
from joblib import load
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

label_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model', 'cluster_labels.pkl'))
cluster_label = load(label_path)

cluster_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model', 'kmeans_model.pkl'))
kmeans_model = load(cluster_path)

    
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'df_to_cluster.csv'))
data = pd.read_csv(data_path)

data_path_original = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'df_cleaned.csv'))
data_ori = pd.read_csv(data_path_original)


def find_most_relatable_movies(movie_indices, movie_features, kmeans_model, n_recommendations=3):
    cluster_labels = kmeans_model.predict(movie_features)
    
    relatable_movies_pre = []
    relatable_movies = []
    
    for movie_index in movie_indices:

        cluster_label = cluster_labels[movie_index]
        
        cluster_indices = np.where(cluster_labels == cluster_label)[0]
        
        distances = euclidean_distances(movie_features[movie_index].reshape(1, -1), movie_features[cluster_indices])
        
        sorted_indices = np.argsort(distances[0])
        
        top_indices = cluster_indices[sorted_indices][1:n_recommendations+1]
        
        relatable_movies_pre.extend(top_indices)

    for i in range(3):
      relatable_movies.append(relatable_movies_pre[i])
    
    return relatable_movies


st.title('Movie Recommendation App')


st.sidebar.title('Input Movies')


input1 = st.sidebar.number_input('Movie Index 1', min_value=0, max_value=len(data)-1, value=0)
input2 = st.sidebar.number_input('Movie Index 2', min_value=0, max_value=len(data)-1, value=1)
input3 = st.sidebar.number_input('Movie Index 3', min_value=0, max_value=len(data)-1, value=2)

# Show selected movies
st.subheader('Selected Movies:')
selected_movies = [data_ori['title'][input1], data_ori['title'][input2], data_ori['title'][input3]]
st.write(selected_movies)

# Recommendation logic
if st.button('Get Recommendations'):
    input_movie_ids = [input1, input2, input3]
    recommendations = find_most_relatable_movies(input_movie_ids, data.values, kmeans_model)
    st.subheader('Recommended Movies:')
    st.write(recommendations)

