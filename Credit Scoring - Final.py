# -*- coding: utf-8 -*-
"""
Created on Sat May  9 02:02:41 2020

@author: VIDHI
"""

#importing necessary libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
import seaborn as sns
from numpy import nan
from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import xgboost as xgb
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import DiscriminationThreshold

#%%
#importing data
dataset = pd.read_csv(r'C:\Users\VIDHI\Downloads\cs-training.csv', header=None)
#%%
#Preprocessing the data
# count the number of missing values for each column
num_missing = (dataset[[1,2,3,4,5,6,7,8,9,10,11]] == 'X').sum()
# report the results
print(num_missing)
#%%
# marking missing values with nan values
# replace 'x' values with 'nan'
dataset[[6,11]] = dataset[[6,11]].replace('X', nan)
# count the number of nan values in each column
print(dataset.isnull().sum())

dataset = dataset.drop(12, axis = 1)
dataset = dataset.drop(13, axis = 1)
dataset = dataset.drop(14, axis = 1)

print(dataset.shape)

# drop rows with missing values
dataset.dropna(inplace=True)
# summarize the shape of the data with missing rows removed
print(dataset.shape)

#changing objects to floats 64
dataset[1] = dataset[1].astype(str).astype(float)
dataset[2] = dataset[2].astype(str).astype(float)
dataset[3] = dataset[3].astype(str).astype(float)
dataset[4] = dataset[4].astype(str).astype(float)
dataset[5] = dataset[5].astype(str).astype(float)
dataset[6] = dataset[6].astype(str).astype(float)
dataset[7] = dataset[7].astype(str).astype(float)
dataset[8] = dataset[8].astype(str).astype(float)
dataset[9] = dataset[9].astype(str).astype(float)
dataset[10] = dataset[10].astype(str).astype(float)
dataset[11] = dataset[11].astype(str).astype(float)

#%%
#creating a dataframe
ds = pd.DataFrame(dataset)
#%%
#Identifying outliers
k = np.abs(stats.zscore(ds))
print(k)

#setting threshold
threshold = 4
print(np.where(k> 4))
#%%
#removing outliers
ds1 =ds[(k<4).all(axis=1)]
print(ds1.shape)

#%%
#EDA
ds1.info()
ds1.describe()
ds1.head()
print(ds1.shape)
#%%
#Visual EDA of input values
sns.pairplot(ds1)

#%%
#dropping unnecessary columns
ds1 = ds1.drop(0, axis =1)
#%%
ds1 = pd.DataFrame(ds1)
#%%
#Scaling the dataset
ss = StandardScaler()

ss_ts = ds1[[3,6,7,2,5]]

scaled = ss.fit_transform(ss_ts)
#%%
#checking variance of the scaled data
scaled_ds = pd.DataFrame(scaled)
scaled_ds.var()

#%%
ds2 = ds1
#%%
ds2[3] = scaled_ds[0].values
ds2[6] = scaled_ds[1].values
ds2[7] = scaled_ds[2].values
ds2[2] = scaled_ds[3].values
ds2[5] = scaled_ds[4].values
#%%
#defining target and feature varaibles
X = ds2.drop(1, axis = 1)
y = ds2[1]
#%%
#splitting the data to check performance later
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 21, test_size = 0.3)

#%%
#Fitting K nearest neighbor model
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, y_train)
y_pred1 = knn.predict(X_test)

#%%
#Measuign performance
train_score1= knn.score(X_train, y_train)
test_score1 = knn.score(X_test, y_test)
print(train_score1)
print(test_score1)
#%%
print(accuracy_score(y_test, y_pred1))
#%%
print(classification_report(y_test, y_pred1))
#%%
print(confusion_matrix(y_test, y_pred1))
#%%
xgbc = xgb.XGBClassifier(objective='binary:logistic', n_estimators=180, seed=123)

# Fit the classifier to the training set
xgbc.fit(X_train,y_train)

# Predict the labels of the test set: preds
preds = xgbc.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

#%%
#Measuign performance
train_score2= xgbc.score(X_train, y_train)
test_score2 = xgbc.score(X_test, y_test)
print(train_score1)
print(test_score1)
#%%
#Measuign performance
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))
print(accuracy_score(y_test, preds))

#%%
#ROC
# Instantiate the visualizer with the classification model
visualizer = ROCAUC(xgbc, classes=["will not default", "will default"])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()  

#%%
visualizer = DiscriminationThreshold(xgbc)

visualizer.fit(X, y)        
# Fit the data to the visualizer
visualizer.show()  