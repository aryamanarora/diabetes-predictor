#!/usr/bin/env python -W ignore::DeprecationWarning

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


#Get Data
filePath = "bigyeet.csv"
Data = pd.read_csv(filePath)
print(Data.columns)

target = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]

#Set target and predictors
X = Data.columns
Y = Data[target]
print("DESCRIBE DATA")
print(Y.describe())

#Split into training and test data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.25, random_state=2)

#rs 2 = 14.6305
#Calculate accuracy

def getMae(randomState, maxDepth, minSamplesSplit, minImpurityDecrease):
    # Make model
    Jarvis = RandomForestRegressor(random_state=120, max_depth=3, min_samples_split=7, min_impurity_decrease=40)
    Jarvis.fit(trainX, trainY)
    # Predict
    prediction = Jarvis.predict(testX)
    answers = testY
    #Calculate mean absolute error
    mae = mean_absolute_error(answers, prediction)
    return mae

print(getMae(0, 0, 0, 0))

