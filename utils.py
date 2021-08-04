from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_blobs
import streamlit as st
import numpy as np

n = 1000
np.random.seed(42)
x = np.linspace(0, 6, n)
X = np.linspace(0, 6, n)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + np.random.random(n) * 0.3

@st.cache
def make_predictions(n_est:int):
    'uses the trees to make predictions based on the estimator provided'
    mod1 = DecisionTreeRegressor(max_depth=4)
    y1 = mod1.fit(X,y).predict(X)
    y2 = AdaBoostRegressor(mod1, n_estimators=n_est).fit(X, y).predict(X)
    return y1, y2

@st.cache
def make_blobs_app(n_samples:int, centers:int, cluster_std:float):
    "generates random points defined by the params chosen by the user, using sklearn's make_blobs function"
    return make_blobs(n_samples=n_samples, n_features=2, centers=centers, cluster_std=cluster_std, return_centers=False)