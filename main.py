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
from PIL import Image

#Todo
# create new feature (selectable ex time date if timeseries)
# add plot feature (eda) or add plot scatter any graph



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
     st.header('Best model')
     st.write(pull())

     models = ('1', '2', '3')
     selected_model = st.selectbox('Select model (top 3)', models)
     st.write('selected model :', best[int(selected_model)-1])
     st.header('Predicted dataframe')
     predictions = predict_model(best[int(selected_model)-1], data=dataframe)
     st.write(predictions.head())

     st.header('Predicted result')
     dictt = {'actual': predictions[selected_target], 'predict': predictions['Label']}
     df_test = pd.DataFrame(dictt)
     st.write(df_test.head())

     st.header('Residuals')
     plot_model(best[int(selected_model)-1], plot='residuals',save=True)
     image = Image.open('Residuals.png')
     st.image(image, caption='Residuals')

     st.header('Feature importance')
     plot_model(best[int(selected_model) - 1], plot='feature', save=True)
     image = Image.open('Feature Importance.png')
     st.image(image, caption='Feature importance')