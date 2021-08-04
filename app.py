import numpy as np
import streamlit as st
import matplotlib.pylab as plt
from utils import *

st.set_page_config(
    page_title="ML models comparison", # => Quick reference - Streamlit
    page_icon="🚀",
    layout="centered", # wide
    initial_sidebar_state="auto") # collapsed

st.title('DecisionTreeRegressor vs Adaboost')

n_est = st.sidebar.slider("n_est", min_value=1, max_value=5_000, step=1)

y1, y2 = make_predictions(n_est=n_est)

if st.sidebar.checkbox("Toggle ScatterChart"):
    plt.scatter(x, y, alpha=0.1)
plt.plot(x, y1, label="just a tree")
plt.plot(x, y2, label=f"adaboost-{n_est}")
plt.legend()

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()