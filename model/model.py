import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
from joblib import load
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

label_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cluster_labels.pkl'))
cluster_label = load(label_path)

cluster_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'kmeans_model.pkl'))
kmeans_model = load(cluster_path)

    
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'df_to_cluster.csv'))
data = pd.read_csv(data_path)

data_path_original = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'df_cleaned.csv'))
data_ori = pd.read_csv(data_path_original)


def find_most_relatable_movies(movie_indices, movie_features, n_recommendations=3):
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