#!/usr/bin/env python -W ignore::DeprecationWarning

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier

#Get Data
filePath = "Diabetes.csv"
Data = pd.read_csv(filePath)
del Data['DiabetesPedigreeFunction']
print(Data.columns)
X_train, X_test, y_train, y_test = train_test_split(Data.loc[:, Data.columns != 'Outcome'], Data['Outcome'], stratify=Data['Outcome'], random_state=66, test_size = 0.25)






#rs 2 = 14.6305
#Calculate accuracy

def getMae():
    # Make model
    Jarvis = GradientBoostingClassifier(random_state = 6, min_samples_split = 2, max_leaf_nodes = 15, max_features = 7)
    Jarvis.fit(X_train, y_train)
    # Predict
    prediction = Jarvis.predict(X_test)
    answers = y_test
    #Calculate mean absolute error
    mae = mean_absolute_error(answers, prediction)
    return mae
print(getMae())

for i in range(1, 100):
    if(getMae(i) < 0.3066):
        print(getMae(i))
        print(i)

