import plotly.graph_objs as go
import plotly as plotly
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix

import pickle
from PIL import Image
import streamlit as st

import base64 #to code & encode into excel format
import time #for file-naming convention
timestr = time.strftime("%Y%m%d-%H%M%S")

#Sets the layout to full width
st.set_page_config(layout= "wide")
image = Image.open("logo.png")
st.image(image)


# title of the app
st.title("""
Predictive Maintainence powered by AI

Upload the data gathered by sensors. 

For smooth functioning, upload data in the specified format.

""")

#un-pickling the pickle file
model = pickle.load(open('model_v2.pkl','rb'))

#adding a sidebar
st.sidebar.header("Upload CSV File")

#setup file upload
uploaded_file = st.sidebar.file_uploader(label="Upload your file in CSV format",
                         type=['csv'])

#----------------------------------------------------------------------------------#

#Encoding Categorical Features

def cat_feat(df):

    df = df.drop(['DATE'], axis =1)
    df_dv = pd.get_dummies(df['REGION_CLUSTER'])

    df_dv = df_dv.rename(
        columns={"A": "CLUSTER_A", "B": "CLUSTER_B", "C": "CLUSTER_C", "D": "CLUSTER_D", "E": "CLUSTER_E",
                 "F": "CLUSTER_F", "G": "CLUSTER_G", "H": "CLUSTER_H"})

    df = pd.concat([df, df_dv], axis=1)

    df_dv = pd.get_dummies(df['MAINTENANCE_VENDOR'])

    df_dv = df_dv.rename(
        columns={"I": "MV_I", "J": "MV_J", "K": "MV_K", "L": "MV_L", "M": "MV_M", "N": "MV_N", "O": "MV_O",
                 "P": "MV_P"})

    df = pd.concat([df, df_dv], axis=1)

    df_dv = pd.get_dummies(df['MANUFACTURER'])

    df_dv = df_dv.rename(
        columns={"Q": "MN_Q", "R": "MN_R", "S": "MN_S", "T": "MN_T", "U": "MN_U", "V": "MN_V", "W": "MN_W", "X": "MN_X",
                 "Y": "MN_Y", "Z": "MN_Z"})

    df = pd.concat([df, df_dv], axis=1)

    df_dv = pd.get_dummies(df['WELL_GROUP'])

    df_dv = df_dv.rename(
        columns={1: "WG_1", 2: "WG_2", 3: "WG_3", 4: "WG_4", 5: "WG_5", 6: "WG_6", 7: "WG_7", 8: "WG_8"})

    final_df = pd.concat([df, df_dv], axis=1)

    return final_df

#-------------------------------------------------------------------#

#Prediction

def predict_results(final_df):
    y_test = final_df["EQUIPMENT_FAILURE"]


    X = final_df.drop(['REGION_CLUSTER', 'MAINTENANCE_VENDOR', 'MANUFACTURER', 'WELL_GROUP', 'EQUIPMENT_FAILURE', 'ID'],
                       axis=1)

    prediction = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
    prediction = pd.DataFrame(prediction, columns =["Possible_Defect"])
    result = pd.concat([final_df['ID'],prediction],axis=1)

    return [result,tp,fp]

#-----------------------------------------------------------------#

#Download predicted results

def filedownload(result):
    csvfile = result.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode() # strings <-> bytes conversions
    new_filename = "new_csv_file_{}_.csv".format(timestr)
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download CSV File with Predicted results.</a>'
    return href

#----------------------------------------------------------------------------------#

st.subheader("Result can be downloaded from here")

global df
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    final_df = cat_feat(df)
    pred,tp,fp = predict_results(final_df)
    st.markdown(filedownload(pred), unsafe_allow_html=True)
    st.write("Total machines that would require maintainence")
    st.info(tp)
    st.write("Assuming maintenance cost /machine to be $30,000. Total Probable Cost to be incurred will be:")
    cost = 30000*tp
    st.info("$" f"{cost:,d}")
    st.write("Act immediately to avoid any further deterioration.")
else:
    st.info("Awaiting for CSV file to be uploaded.")
