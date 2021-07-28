# computational imports
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# plotting imports
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style("darkgrid")
# for reading files from urls
# import urllib.request
# display imports
from IPython.display import display, IFrame
# from IPython.core.display import HTML

# import notebook styling for tables and width etc.
# response = urllib.request.urlopen('https://raw.githubusercontent.com/DataScienceUWL/DS775v2/master/ds755.css')
# HTML(response.read().decode("utf-8"));
import os
import pycaret

# # check version
from pycaret.utils import version
print(version())

def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns

def stripID(x):
    output = None
    sample_name = x.str.split(" ").str.get(0)
    sample_name = sample_name.str[2:]
    output = sample_name
    return(output)

def enhanceAnnotation():
    #read in the data
    # df = pd.read_excel('./data/BUSL_Rubric_PublicAccess_rle2021_2021-04-28.xlsx', sheet_name="Sheet2")

    df = pd.read_csv('./data/BUSL_Rubric_PublicAccess_rle2021_2021-04-28.csv', header=0)

    #print the shape of the dataframe
    print(f"The shape is {df.shape}")

    #get the column info
    print(df.info())
    print(df.isna().sum())
    # df = df.dropna()

    print(f"The shape is {df.shape}")
    # df = df.drop(columns=["Peripheral ZoneACR", "Size", "Quality", "Lesion Type"])
    # index = df[df['Sample name'] == "PA3"].index
    # index =[idx for idx, row in df.iterrows() if len(row['Sample name'].split(" ")) > 1]
    # df.drop(index, inplace=True)
    # df = df.dropna()
    df = df.assign(id = lambda x: stripID(x['Sample name']))
    df["id"] = pd.to_numeric(df["id"])
    # df = df.drop(columns=["Sample name", "id"])
    df['Biopsy'] = df['BI-RADS'].apply(lambda x: 'No' if x in ['1', '2', '3'] else 'Yes')
    print(df.info())
    display(missing_values_table(df))
    print(f"The shape is {df.shape}")
    # df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    print(df.describe())
    display(df)

    df.to_csv('dataScored.csv')

# https://stackoverflow.com/questions/65012601/attributeerror-simpleimputer-object-has-no-attribute-validate-data-in-pyca
# https://github.com/pycaret/pycaret/issues/1107

from pycaret.classification import *
df_target = pd.read_csv('./data/dataScored.csv', header=0)
df_predictors = pd.read_csv('./data/statsGT.csv', header=0)
# df = df_target.set_index('id').join(df_predictors.set_index('id'))
df = df_target.join(df_predictors, on='id', rsuffix='r')
# target_list = ['Peripheral ZoneACR', 'Peripheral ZoneTS',
#        'Marginal ZoneACR', 'Boundary ZoneACR', 'MarginalBoundary ZoneTS',
#        'ShapeACR', 'ShapeTS', 'Orientation', 'Echo PatternACR',
#        'Echo PatternTS', 'Posterior FeaturesACR', 'Posterior FeaturesTS',
#        'VascularityACR']

target_list = ['Orientation']

predictor_list = ['aspect_ratio', 'area', 'leftx',
       'rightx', 'topy', 'bottomy', 'width', 'height', 'extent', 'solidity',
       'angle', 'min_val', 'max_val', 'min_loc', 'max_loc', 'mean_val', 'cx',
       'cy', 'perimeter']
print(df_predictors.columns)

for target in target_list:
    columns  = []
    columns.append(target)
    columns.extend(predictor_list)
    df_py = df[columns]
    print(df_py.columns)

# # clf1 = setup(df, target = 'BI-RADS', imputation_type='iterative', session_id=123, log_experiment=True, experiment_name='exp1',fix_imbalance=True,
# #  ignore_features=["Histology"], feature_selection=True)

    clf1 = setup(df_py, target = target, imputation_type='iterative', session_id=123, log_experiment=False, experiment_name='exp1',fix_imbalance=True,
    html=False, silent=True)

    # best_model = compare_models(include=['lr'])

    # display(best_model)


# def on_close(event):
#     print('Closed Figure!')

    model = create_model('lr')

    # tuned_model = tune_model(best_model)
    # f = plt.figure(figsize=(12,16))
    # f.add_subplot(3, 1, 1)
# # f.canvas.mpl_connect('close_event', on_close)
# #  https://pycaret.org/plot-model/
    plot_model(model, plot='auc')
# f.add_subplot(2, 1, 2)
    plot_model(model, plot='feature')
    plot_model(model, plot='confusion_matrix')
# plot_model(tuned_model, plot='threshold')

# #  ,display_format="streamlit"
# # grid.arrange(plt1, plt2)
# # plot_grid(c(plt1, plt2))
