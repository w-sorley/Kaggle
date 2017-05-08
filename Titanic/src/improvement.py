#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt


def crossValidate(df, cvModel):
    useData = df.filter(
        regex='Survived|Age_.*|SibSp|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    X = useData.as_matrix()[:, 1:]
    y = useData.as_matrix()[:, 0]
    plotLearningCurve(cvModel, X, y)
    validateScore = cross_validation.cross_val_score(cvModel, X, y, cv=5)
    return validateScore


def getBadCase(df):
    spilt_train, split_cv = cross_validation.train_test_split(
        df, test_size=0.3, random_state=0)
    dataTrain = spilt_train.filter(
        regex='Survived|Age_.*|SibSp|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    dataCV = split_cv.filter(
        regex='Survived|Age_.*|SibSp|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    trainX = dataTrain.as_matrix()[:, 1:]
    trainy = dataTrain.as_matrix()[:, 0]
    lrModel = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    lrModel.fit(trainX, trainy)
    cvX = dataCV.as_matrix()[:, 1:]
    cvy = dataCV.as_matrix()[:, 0]
    predictins = lrModel.predict(cvX)
    badCase = df.loc[df['PassengerId'].isin(
        split_cv[predictins != cvy]['PassengerId'].values)]
    return badCase


def plotLearningCurve(trainModel, X, y):

    trainSizes, trainScores, testScores = learning_curve(
        trainModel, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(0.05, 1.0, 20), verbose=0)

    trainScoresMean = np.mean(trainScores, axis=1)
    trainScoresStd = np.std(trainScores, axis=1)
    testScoresMean = np.mean(testScores, axis=1)
    testScoresStd = np.std(testScores, axis=1)

    plt.figure()
    plt.title("模型学习曲线")
    plt.xlabel('学习样本数')
    plt.ylabel('得分')
    plt.gca().invert_yaxis()
    plt.grid()

    plt.fill_between(trainSizes, trainScoresMean - trainScoresStd,
                     trainScoresMean + trainScoresStd, alpha=0.1, color='b')
    plt.fill_between(trainSizes, testScoresMean - testScoresStd,
                     testScoresMean + testScoresStd, alpha=0.1, color='r')
    plt.plot(trainSizes, trainScoresMean, 'o-', color='b', label='训练集上的得分')
    plt.plot(trainSizes, testScoresMean, 'o-', color='r', label='交叉验证集的得分')
    plt.legend(loc='best')
    plt.draw()
    plt.show()
    plt.gca().invert_yaxis()

    midPoint = ((trainScoresMean[-1] + trainScoresStd[-1]) +
                (testScoresMean[-1] - testScoresStd[-1])) / 2
    diff = ((trainScoresMean[-1] + trainScoresStd[-1]) -
            (testScoresMean[-1] - testScoresStd[-1]))
    return midPoint, diff
