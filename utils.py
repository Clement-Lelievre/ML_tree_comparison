from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import streamlit as st
import numpy as np

n = 1000
np.random.seed(42)
x = np.linspace(0, 6, n)
X = np.linspace(0, 6, n)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + np.random.random(n) * 0.3

@st.cache
def make_predictions(n_est):
    mod1 = DecisionTreeRegressor(max_depth=4)
    y1 = mod1.fit(X,y).predict(X)
    y2 = AdaBoostRegressor(mod1, n_estimators=n_est).fit(X, y).predict(X)
    return y1, y2