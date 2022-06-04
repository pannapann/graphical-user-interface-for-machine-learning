import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import pickle
import datetime
import os
from io import StringIO
from PIL import Image


#Todo
# create new feature (selectable ex time date if timeseries)
# add plot feature (eda) or add plot scatter any graph
# add add feature and try prediction (https://pycaret.gitbook.io/docs/learn-pycaret/official-blog/build-and-deploy-ml-app-with-pycaret-and-streamlit)
# add feature small batches (if dataset is too large we just sample it for efficiency)
# save model (button or automatically)
# create function at compare_model() so we can cache increase speed
# add selection number of fold
# explore setup pycaret
# add sort at compare_model (sort by rmse mae r2 etc)


classification_dict = {'Area Under the Curve':['auc','AUC'],
                       'Discrimination Threshold':['threshold','Threshold'],
                       'Precision Recall Curve':['pr', 'Precision Recall'],
                       'Confusion Matrix':['confusion_matrix', 'Confusion Matrix'],
                       'Class Prediction Error':['error','Prediction Error'],
                       'Classification Report':['class_report', 'Class Report'],
                       'Decision Boundary':['boundary', 'Decision Boundary'],
                       'Recursive Feature Selection':['rfe', 'Feature Selection'],
                       'Learning Curve':['learning', 'Learning Curve'],
                       'Manifold Learning':['manifold', 'Manifold Learning'],
                       'Calibration Curve':['calibration', 'Calibration Curve'],
                       'Validation Curve':['vc', 'Validation Curve'],
                       'Dimension Learning':['dimension', 'Dimensions'],
                       'Feature Importance (Top 10)':['feature','Feature Importance'],
                       'Feature Importance (all)':['feature_all', 'Feature Importance (All)'],
                       'Lift Curve':['lift','lift'],
                       'Gain Curve':['gain', 'gain'],
                       'KS Statistic Plot':['ks','ks']}

regression_dict = {'Residuals Plot':['residuals','Residuals'],
                    'Prediction Error Plot':['error', 'Prediction Error'],
                    'Cooks Distance Plot':['cooks', 'Cooks Distance'],
                    'Recursive Feature Selection':['rfe', 'Feature Selection'],
                    'Learning Curve':['learning', 'Learning Curve'],
                    'Validation Curve':['vc', 'Validation Curve'],
                    'Manifold Learning':['manifold', 'Manifold Learning'],
                    'Feature Importance (top 10)':['feature', 'Feature Importance'],
                    'Feature Importance (all)':['feature_all', 'Feature Importance (All)']}


def setup_pycaret():
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

    return train_ratio, selected_target


st.header('Upload file')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
     # Can be used wherever a "file-like" object is accepted:
     dataframe = pd.read_csv(uploaded_file)
     st.write(dataframe)

st.header('Select machine learning types')
target = ('Regression', 'Classification', 'Time series')
selected_target = st.selectbox('Select machine learning types', target)
st.write('selected machine learning types :', selected_target)


if selected_target == 'Regression' and uploaded_file is not None:
     from pycaret.regression import *
     train_ratio, selected_target = setup_pycaret()
     s = setup(dataframe, target = selected_target,silent =True, train_size = train_ratio)

     best = compare_models(sort='rmse',n_select=3)
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

     st.header('Select evaluation')
     options = st.multiselect(
          'Select evaluation',
          ['Residuals Plot',
           'Prediction Error Plot',
           'Cooks Distance Plot',
           'Recursive Feature Selection',
           'Learning Curve',
           'Validation Curve',
           'Manifold Learning',
           'Feature Importance (top 10)',
           'Feature Importance (all)']
     )

     for i in options:
          plot_model(best[int(selected_model) - 1], plot=regression_dict[i][0], save=True)
          image = Image.open(f'{regression_dict[i][1]}.png')
          st.header(i)
          st.image(image, caption=i)


elif selected_target == 'Classification' and uploaded_file is not None:
     from pycaret.classification import *
     train_ratio, selected_target = setup_pycaret()

     s = setup(dataframe, target = selected_target,silent =True, train_size = train_ratio)

     best = compare_models( n_select=3)
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

     st.header('Select evaluation')
     options = st.multiselect(
          'Select evaluation',
          ['Area Under the Curve',
           'Discrimination Threshold',
           'Precision Recall Curve',
           'Confusion Matrix',
           'Class Prediction Error',
           'Classification Report',
           'Decision Boundary',
           'Recursive Feature Selection',
           'Learning Curve',
           'Manifold Learning',
           'Calibration Curve',
           'Validation Curve',
           'Dimension Learning',
           'Feature Importance (Top 10)',
           'Feature Importance (all)',
           'Lift Curve',
           'Gain Curve',
           'KS Statistic Plot']
     )

     for i in options:
          plot_model(best[int(selected_model) - 1], plot=classification_dict[i][0], save=True)
          image = Image.open(f'{classification_dict[i][1]}.png')
          st.header(i)
          st.image(image, caption=i)

