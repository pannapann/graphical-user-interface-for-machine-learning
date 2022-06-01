import streamlit as st
import numpy as np
import pandas as pd
#from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
import pickle
import datetime
import os
from io import StringIO
from pycaret.regression import *

#Todo
# create new feature (selectable ex time date if timeseries)



st.header('Upload file')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
     # Can be used wherever a "file-like" object is accepted:
     dataframe = pd.read_csv(uploaded_file)
     st.write(dataframe)

st.header('Select machine learning types')
target = ('Regression', 'Classification', 'Time series')
selected_target = st.selectbox('Select target for prediction', target)
st.write('selected target :', selected_target)


if selected_target == 'Regression' and uploaded_file is not None:
     st.header('Select column to predict')
     target = tuple(dataframe.columns)
     selected_target = st.selectbox('Select target for prediction', target)
     st.write('selected target :', selected_target)

     st.header('Select train : test ratio')
     traintest = st.slider('train:test:', min_value=0, max_value=100, step=5, value=80)
     train_ratio = traintest / 100
     st.write('train ratio :', train_ratio)
     test_ratio = (100 - traintest) / 100
     st.write('test ratio :', test_ratio)

     s = setup(dataframe, target = selected_target,silent =True, train_size = train_ratio)

     best = compare_models(sort='rmse', n_select=3)
     st.write(best[0])
     st.write(best[1])
     st.write(best[2])
     st.write(pull())
     predictions = predict_model(best[0], data=dataframe)

     st.write(predictions.head())
