#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from dataprocess import preProcessTest, preProcessTrain
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor


dataTrain = pd.read_csv("../data/train.csv")
dataTest = pd.read_csv("../data/test.csv")


def buildModel(df):
    trainData = df.filter(
        regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    numpypData = trainData.as_matrix()
    y = numpypData[:, 0]
    X = numpypData[:, 1:]

    lrModel = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    # lrModel.fit(X, y)
    # return lrModel
    baggingLrModel = BaggingRegressor(lrModel, n_estimators=20, max_samples=0.8,
                                      max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1)
    baggingLrModel.fit(X, y)
    return baggingLrModel


def main():

    formattedDataTrain, ageModel = preProcessTrain(dataTrain)
    lrModel = buildModel(formattedDataTrain)
    dataTest.loc[(dataTest.Fare.isnull()), 'Fare'] = 0
    formattedDataTest = preProcessTest(dataTest, ageModel)
    testX = formattedDataTest.filter(
        regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictSurvived = lrModel.predict(testX)
    result = pd.DataFrame({'PassengerId': dataTest['PassengerId'].as_matrix(
    ), 'Survived': predictSurvived.astype(np.int32)})
    result.to_csv("../out/logistic_regression_predictions.csv", index=False)


if __name__ == '__main__':
    main()
