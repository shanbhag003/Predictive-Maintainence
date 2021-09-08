#!/usr/bin/env python
# coding: utf-8

# In[1]:


import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly as plotly
import types
import pandas as pd
import numpy as np
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from xgboost.sklearn import XGBClassifier


# In[2]:


df1 = pd.read_csv(r"C:\Users\kartik\Predictive Main\equipment_failure_data_1.csv")


# In[3]:


df2 = pd.read_csv(r"C:\Users\kartik\Predictive Main\equipment_failure_data_2.csv")


# In[4]:


df = pd.concat([df1, df2])


# In[5]:


df = df.drop(["Unnamed: 0"], axis = 1)


# In[6]:


df.head()


# In[7]:


xxxx = pd.DataFrame(df.groupby(['ID']).agg(['count']))
xxxx.shape


# In[8]:


df = df.drop(columns =['ID', 'DATE'])


# In[9]:


df.head()


# In[10]:


X = df.drop("EQUIPMENT_FAILURE", axis =1)
y = df["EQUIPMENT_FAILURE"]


# In[11]:


X.shape, y.shape


# In[12]:


from imblearn.over_sampling import SMOTENC
smx = SMOTENC(random_state=12, categorical_features=[0, 1, 2, 3])
X_res, y_res = smx.fit_resample(X, y)


# In[14]:


from collections import Counter
print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))


# In[13]:


X_res.shape, y_res.shape


# In[19]:


X_res.head()


# In[20]:


df_dv = pd.get_dummies(X_res['REGION_CLUSTER'])

df_dv=df_dv.rename(columns={"A": "CLUSTER_A","B":"CLUSTER_B","C":"CLUSTER_C","D":"CLUSTER_D","E":"CLUSTER_E","F":"CLUSTER_F","G":"CLUSTER_G","H":"CLUSTER_H"})

X_res= pd.concat([X_res, df_dv], axis=1)

df_dv = pd.get_dummies(X_res['MAINTENANCE_VENDOR'])

df_dv=df_dv.rename(columns={"I": "MV_I","J":"MV_J","K":"MV_K","L":"MV_L","M":"MV_M","N":"MV_N","O":"MV_O","P":"MV_P"})

X_res = pd.concat([X_res, df_dv], axis=1)

df_dv = pd.get_dummies(X_res['MANUFACTURER'])

df_dv=df_dv.rename(columns={"Q": "MN_Q","R":"MN_R","S":"MN_S","T":"MN_T","U":"MN_U","V":"MN_V","W":"MN_W","X":"MN_X","Y":"MN_Y","Z":"MN_Z"})

X_res = pd.concat([X_res, df_dv], axis=1)

df_dv = pd.get_dummies(X_res['WELL_GROUP'])

df_dv=df_dv.rename(columns={1: "WG_1",2:"WG_2",3:"WG_3",4:"WG_4",5:"WG_5",6:"WG_6",7:"WG_7",8:"WG_8"})

X_res = pd.concat([X_res, df_dv], axis=1)


# In[21]:


X_res.head()


# In[22]:


X_res.columns


# In[23]:


X_res.shape


# In[24]:


X_res = X_res.drop(columns =['REGION_CLUSTER', 'MAINTENANCE_VENDOR', 'MANUFACTURER',
       'WELL_GROUP'])


# In[25]:


X_res.shape


# In[26]:


X_res.head()


# In[27]:


y_res.value_counts()


# # Model Building

# In[31]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, recall_score, precision_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state = 71)


# In[34]:


print("X_train shape is {}".format(X_train.shape))
print("X_test shape is {}".format(X_test.shape))
print("y_train shape is {}".format(y_train.shape))
print("y_test shape is {}".format(y_test.shape))


# In[36]:


xgb = XGBClassifier(n_estimators=100)


# In[49]:


X_train.columns


# In[50]:


X_train = X_train.values
y_train = y_train.values


# In[51]:


final_model = xgb.fit(X_train, y_train)


# In[52]:


preds = xgb.predict(X_test)


# In[53]:


print('\nconfustion matrix') 
print(confusion_matrix(y_test, preds))


print('\n')
print("Accuracy:",metrics.accuracy_score(y_test, preds))
print("Precision:",metrics.precision_score(y_test, preds))
print("Recall:",metrics.recall_score(y_test, preds))


print('\nclassification report')
print(classification_report(y_test, preds))


# # MODEL PICKLE

# In[44]:


import pickle


# In[54]:


file = open('model_v2.pkl', 'wb')


# In[55]:


pickle.dump(final_model, open("model_v2.pkl", "wb"))


# In[56]:


X_train.columns


# In[57]:


X_train.shape


# In[ ]:




