# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:54:22 2019

@author: nikhi
"""

import pandas as pd
import numpy as np
dataFile = pd.read_csv(r'E:\assignments\Semester3\Data Mining\homework\Homework1\melb_data.csv')
dataFile['Date'] = pd.to_datetime(dataFile['Date'])
dataFile['Date'] = dataFile['Date'].dt.year
dataFile['Date'] = dataFile['Date'].astype(int)
dataFile2 = pd.DataFrame(dataFile)

del dataFile['Address']

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(dataFile.iloc[:,[11,13,14]])
dataFile.iloc[:,[11,13,14]] = imputer.fit_transform(dataFile.iloc[:,[11,13,14]])


imputer2 = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputer2.fit( dataFile.iloc[:,[5,15]])
dataFile.iloc[:,[5,15]] = imputer2.fit_transform(dataFile.iloc[:,[5,15]])

dataFile2 = pd.get_dummies(data=dataFile,columns=['Suburb','Type','Method','SellerG','CouncilArea','Regionname'])

dataFile2.sort_values("Price",inplace=True)


# create target

target = pd.DataFrame(np.zeros((13580, 1)),columns=['house_value'])
target.house_value.iloc[0:2716] = 1
target.house_value.iloc[2716:5432] = 2
target.house_value.iloc[5432:8148] = 3
target.house_value.iloc[8148:10864] = 4
target.house_value.iloc[10864:13581] = 5

del dataFile2['Price']
from sklearn.model_selection import train_test_split


knn_neighbours = [5,6,7,8,9,10]

averageData = []
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

for k in knn_neighbours:
    knn = KNeighborsClassifier(n_neighbors=k)
    average = 0
    for random_value  in range(10):
        X_train, X_test, y_train, y_test = train_test_split(dataFile2,target, test_size=0.34,random_state=random_value)
        knn.fit(X_train, y_train.values.ravel())
        y_pred = knn.predict(X_test)
       
       # print("Accuracy for Knn:",random_value ,metrics.accuracy_score(y_test, y_pred))
       
        average = average + metrics.accuracy_score(y_test, y_pred)
    averageData.append(average/10)
    print("average accuracy for nearest neighbours",k,average/10)
    

from matplotlib import pyplot as plt
plt.bar(knn_neighbours,averageData, label='Bars1')
plt.plot(knn_neighbours,averageData)
from sklearn.ensemble import RandomForestClassifier

averageForestData = 0;
for random_value in range(10):
    rfc = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(dataFile2,target, test_size=0.34,random_state=random_value)
    rfc.fit(X_train,y_train.values.ravel())
    rfc_predict = rfc.predict(X_test)
#averageData.append(metrics.accuracy_score(y_test, rfc_predict))
    averageForestData = averageForestData + metrics.accuracy_score(y_test, rfc_predict)
    print( metrics.accuracy_score(y_test, rfc_predict))
print("Accuracy for random forest:",averageForestData/10)




#plt.plot(knn_neighbours,averageData)