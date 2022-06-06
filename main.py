import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import datetime
import plotly
import json
import re

#Todo
# create new feature (selectable ex time date if timeseries)
# (DONE) add plot feature (eda) or add plot scatter any graph
# add add feature and try prediction (https://pycaret.gitbook.io/docs/learn-pycaret/official-blog/build-and-deploy-ml-app-with-pycaret-and-streamlit)
# add feature small batches (if dataset is too large we just sample it for efficiency)
# (DONE) save model (button or automatically)
# (DONE) create function at compare_model() so we can cache increase speed
# add selection number of fold
# explore setup pycaret(data preparation)
# (OPTIONAL) add sort at compare_model (sort by rmse mae r2 etc)
# add config file (store dict)
# add tune model


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


clustering_dict = {'Cluster PCA Plot (2d)':['cluster','cluster'],
                    'Cluster TSnE (3d)':['tsne', 'tsne'],
                    'Elbow Plot':['elbow', 'Elbow'],
                    'Silhouette Plot':['silhouette', 'Silhouette'],
                    'Distance Plot':['distance', 'Distance'],
                    'Distribution Plot':['distribution', 'distribution'],}


anomaly_dict = {'t-SNE (3d) Dimension Plot':['tsne', 'tsne'],
                'UMAP Dimensionality Plot':['umap', 'umap']}


nlp_dict = {'Word Token Frequency':['frequency', 'Word Frequency.html'],
            'Word Distribution Plot':['distribution', 'Distribution.html'],
            'Bigram Frequency Plot':['bigram', 'Bigram.html'],
            'Trigram Frequency Plot':['trigram', 'Trigram.html'],
            'Sentiment Polarity Plot':['sentiment', 'Sentiments.html'],
            'Part of Speech Frequency': ['pos', 'POS.html'],
            't-SNE (3d) Dimension Plot':['tsne', 'TSNE.html'],
            'UMAP Dimensionality Plot':['umap', 'UMAP'],
            # 'Topic Model (pyLDAvis)':['topic_model', 'topic_model'],
            # 'Topic Infer Distribution':['topic_distribution', 'topic_distribution'],
            'Wordcloud':['wordcloud', 'wordcloud']}


def read_from_html(plot_json):
    with open(plot_json) as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html[-2**16:])[0]
    call_args = json.loads(f'[{call_arg_str}]')
    plotly_json = {'data': call_args[1], 'layout': call_args[2]}
    return plotly.io.from_json(json.dumps(plotly_json))


def init_pycaret():
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


@st.cache
def plot_scatter(feature_a,feature_b):
    fig = px.scatter(dataframe, x=feature_a, y=feature_b, trendline="ols")
    fig.update_xaxes(
        rangeslider_visible=True,
    )
    fig.update_layout(
        title='Scatter plot',
        autosize=True, )
    return fig


def initiate_dataframe():
    image = Image.open(f'assets/title.jpg')
    st.image(image)
    st.header('Upload csv file')
    uploaded_file = st.file_uploader("Upload csv file")
    if uploaded_file is None:
        st.stop()
    if uploaded_file is not None:
        try:
            dataframe = pd.read_csv(uploaded_file)
            st.write(dataframe)
        except:
            print('Upload csv file only!!')
    return dataframe, uploaded_file


@st.cache(suppress_st_warning=True,hash_funcs={'xgboost.sklearn.XGBRegressor': id},allow_output_mutation=True)
def setup_pycaret(train_ratio,selected_target):
    s = setup(dataframe, target=selected_target, silent=True, train_size=train_ratio)
    best = compare_models(n_select=3)
    compare_df = pd.DataFrame(pull())
    return best,compare_df


def eda_pycaret():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.header('EDA')
    st.pyplot(eda(display_format='svg'))


def select_ml():
    st.header('Select machine learning types')
    ml = ('Regression', 'Classification', 'Clustering', 'Anomaly Detection', 'Natural Language Processing')
    selected_ml = st.selectbox('Select machine learning types', ml)
    st.write('selected machine learning types :', selected_ml)
    return selected_ml


def evaluation_pycaret(selected_ml,selected_model,best):
    if selected_ml == 'Regression':
        dict = regression_dict
        model_ = best[int(selected_model) - 1]
    elif selected_ml == 'Classification':
        dict = classification_dict
        model_ = best[int(selected_model) - 1]
    elif selected_ml == 'Clustering':
        dict = clustering_dict
        model_ = best
    elif selected_ml == 'Anomaly Detection':
        dict = anomaly_dict
        model_ = best
    elif selected_ml == 'Natural Language Processing':
        dict = nlp_dict
        model_ = best

    st.header('Select evaluation')
    options = st.multiselect(
        'Select evaluation', options=dict.keys()
    )

    for i in options:
        plot_model(model_, plot=dict[i][0], save=True)
        try:
            image = Image.open(f'{dict[i][1]}.png')
            st.header(i)
            st.image(image, caption=i)
        except:
            st.header(i)
            st.plotly_chart(read_from_html(dict[i][1]))


def top_3_model(best,selected_target):
    st.header('top 3 model')
    models = ('1', '2', '3')
    selected_model = st.selectbox('Select model (top 3)', models)
    st.write('selected model :', best[int(selected_model) - 1])
    st.header('Predicted dataframe')
    predictions = predict_model(best[int(selected_model) - 1], data=dataframe)
    st.write(predictions.head())

    st.header('Predicted result')
    dictt = {'actual': predictions[selected_target], 'predict': predictions['Label']}
    df_test = pd.DataFrame(dictt)
    st.write(df_test.head())

    return selected_model


def select_scatter():
    st.header('Scatter plot between each feature')
    options = st.multiselect(
        'Select 2 feature',
        options=dataframe.columns)
    try:
        st.plotly_chart(plot_scatter(options[0], options[1]))
    except IndexError:
        st.write('Please select 2 numerical features')
    except ValueError:
        st.write('Please select 2 numerical features')


def save_model_pycaret(best,selected_model,selected_ml,compare_df = 'default'):
    if selected_ml == 'Regression':
        model_ = best[int(selected_model) - 1]
        selected_model = compare_df.index[int(selected_model)-1]
    elif selected_ml == 'Classification':
        model_ = best[int(selected_model) - 1]
        selected_model = compare_df.index[int(selected_model) - 1]
    elif selected_ml == 'Clustering':
        model_ = best
    elif selected_ml == 'Anomaly Detection':
        model_ = best
    elif selected_ml == 'Natural Language Processing':
        model_ = best
    st.header('Save model')

    if st.button('Save model'):
        save_model(model_, f'best_{selected_model}_{datetime.datetime.now()}')
        st.write('Model saved!!')
    else:
        pass


@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def cache_model(selected_ml):
    if selected_ml in ['Clustering', 'Anomaly Detection']:
        s = setup(dataframe, normalize=True, silent=True)

    elif selected_ml in ['Natural Language Processing']:
        st.header('Select column to predict')
        target = tuple(dataframe.columns)
        selected_target = st.selectbox('Select target for prediction', target)
        st.write('selected target :', selected_target)
        s = setup(dataframe, target=selected_target)


def unsupervised_pipeline(md):
    st.header(f'Select {selected_ml} model')
    selected_md = st.selectbox(f'Select {selected_ml} model', md)
    st.write(f'selected {selected_ml} model :', selected_md)
    model = unsupervised_model(selected_md)
    result = assign_model(model)
    st.header(f'{selected_md} model prediction')
    st.write(result.head())
    evaluation_pycaret(selected_ml, 1, model)
    save_model_pycaret(model, selected_md, selected_ml)


@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def unsupervised_model(selected_md):
    model = create_model(selected_md)
    st.write(model)
    return model


def pipeline_st(selected_ml):
    if selected_ml in ['Regression','Classification']:
        train_ratio, selected_target = init_pycaret()
        best, compare_df = setup_pycaret(train_ratio, selected_target)
        eda_pycaret()
        select_scatter()
        st.header('Compare model')
        st.write(compare_df)
        selected_model = top_3_model(best,selected_target)
        evaluation_pycaret(selected_ml, selected_model,best)
        save_model_pycaret(best,selected_model,selected_ml,compare_df)

    elif selected_ml in ['Clustering']:
        cm = ('kmeans', 'ap', 'meanshift', 'sc', 'hclust', 'dbscan', 'optics', 'birch', 'kmodes')
        cache_model(selected_ml)
        unsupervised_pipeline(cm)

    elif selected_ml in ['Anomaly Detection']:
        am = ('abod', 'cluster', 'cof', 'histogram', 'knn', 'lof', 'svm', 'pca', 'mcd', 'sod', 'sos')
        cache_model(selected_ml)
        unsupervised_pipeline(am)

    elif selected_ml in ['Natural Language Processing']:
        nlpm = ('lda', 'lsi', 'hdp', 'rp', 'nmf')
        cache_model(selected_ml)
        unsupervised_pipeline(nlpm)



# Start
dataframe,uploaded_file = initiate_dataframe()
selected_ml = select_ml()


if selected_ml == 'Regression' and uploaded_file is not None:
    from pycaret.regression import *
    pipeline_st(selected_ml)
elif selected_ml == 'Classification' and uploaded_file is not None:
    from pycaret.classification import *
    pipeline_st(selected_ml)
elif selected_ml == 'Clustering' and uploaded_file is not None:
    from pycaret.clustering import *
    pipeline_st(selected_ml)
elif selected_ml == 'Anomaly Detection' and uploaded_file is not None:
    from pycaret.anomaly import *
    pipeline_st(selected_ml)
elif selected_ml == 'Natural Language Processing' and uploaded_file is not None:
    from pycaret.nlp import *
    pipeline_st(selected_ml)



