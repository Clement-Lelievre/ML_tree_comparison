import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import *

st.set_page_config(
    page_title="ML models comparison", # => Quick reference - Streamlit
    page_icon="🚀",
    layout="centered", # wide
    initial_sidebar_state="auto") # collapsed

st.set_option('deprecation.showPyplotGlobalUse', False) #prevent an annoying banner related to an error when plotting

if st.sidebar.radio('Select model demo',('Tree','K-means')) == 'Tree':
    st.title('DecisionTreeRegressor vs Adaboost')
    
    n_est = st.sidebar.slider("n_est", min_value=1, max_value=5_000, step=1, value=1_000)

    y1, y2 = make_predictions(n_est=n_est)

    if st.sidebar.checkbox("Toggle ScatterChart"):
        plt.scatter(x, y, alpha=0.1)
    plt.plot(x, y1, label="Simple decision tree")
    plt.plot(x, y2, label=f"adaboost-{n_est}")
    plt.legend()
    st.pyplot()
    
else:
    st.title('K-means demo with make_blobs (2D)')
    st.sidebar.markdown('# Generate random points')
    nb_points = st.sidebar.slider('Number of points',min_value=0,max_value=1000,value=100, step = 10)
    std = st.sidebar.slider('Standard deviation',min_value=0.05, max_value=3.0, value=1.0, step=0.5)
    nb_clusters = st.sidebar.slider('Number of clusters',min_value=1, max_value=10, value=3, step=1, help='How many zones are the points divided into?')
    if st.sidebar.button('Generate points and run K-means'):
        points = make_blobs_app(nb_points,nb_clusters,std)[0]
        X = [item[0] for item in points]
        Y = [item[1] for item in points]
        plt.scatter(X,Y)
        plt.title('Your points')
        st.pyplot()
        km = KMeans(n_clusters=nb_clusters)
        prediction = km.fit_predict(points)
        plt.scatter(X,Y,c=prediction)
        plt.title('K-means labeled datapoints like so')
        st.pyplot()
