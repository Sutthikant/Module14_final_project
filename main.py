import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
from joblib import load
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from model.model import find_most_relatable_movies
    
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'df_to_cluster.csv'))
data = pd.read_csv(data_path)

data_path_original = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'df_cleaned.csv'))
data_ori = pd.read_csv(data_path_original)

st.title('Movie Recommendation App')


st.sidebar.title('Input Movies')


input1 = st.sidebar.text_input('Movie Index 1', "Toy Story")
input2 = st.sidebar.text_input('Movie Index 2', "Jumanji")
input3 = st.sidebar.text_input('Movie Index 3', "Grumpier Old Men")

input1 = data_ori.index[data_ori['title'] == input1]
input2 = data_ori.index[data_ori['title'] == input2]
input3 = data_ori.index[data_ori['title'] == input3]

# Show selected movies
st.subheader('Selected Movies:')
selected_movies = [[data_ori["title"][input1], data_ori["genres"][input1]], [data_ori["title"][input2], data_ori["genres"][input2]], [data_ori["title"][input3], data_ori["genres"][input3]]]
st.write(selected_movies)



# Recommendation logic
if st.button('Get Recommendations'):
    input_movie_ids = [input1, input2, input3]
    recommendations = find_most_relatable_movies(input_movie_ids, data.values)
    output1, output2, output3 = recommendations
    st.subheader('Recommended Movies:')
    st.write([[data_ori["title"][output1], data_ori["genres"][output1]], [data_ori["title"][output2], data_ori["genres"][output2]], [data_ori["title"][output3], data_ori["genres"][output3]]])

