#!/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing

import pandas as pd


def setMissAge(df):

    numDF = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    knownDF = numDF[numDF.Age.notnull()].as_matrix()
    unknownDF = numDF[numDF.Age.isnull()].as_matrix()

    y = knownDF[:, 0]
    X = knownDF[:, 1:]
    model = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    model.fit(X, y)
    predictAge = model.predict(unknownDF[:, 1::])
    df.loc[(df.Age.isnull()), 'Age'] = predictAge

    return df, model


def setCabin(df):
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'No'
    return df


def feature2num(df):
    dummiesCabin = pd.get_dummies(df['Cabin'], prefix='Cabin')
    dummiesEmbarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    dummiesSex = pd.get_dummies(df['Sex'], prefix='Sex')
    dummiesPclass = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df = pd.concat([df, dummiesCabin, dummiesEmbarked,
                    dummiesPclass, dummiesSex], axis=1)
    df.drop(['Pclass', 'Name', 'Ticket', 'Embarked',
             'Sex', 'Cabin'], axis=1, inplace=True)

    return df


def scaleFeature(df):
    scale = preprocessing.StandardScaler()
    ageScalePara = scale.fit(df["Age"])
    df['AgeScaled'] = scale.fit_transform(df["Age"], ageScalePara)
    fareScalePara = scale.fit(df["Fare"])
    df['FareScaled'] = scale.fit_transform(df['Fare'], fareScalePara)
    return df


def preProcessTrain(rawData):
    (df, ageModel) = setMissAge(rawData)
    df = setCabin(df)
    df = feature2num(df)
    formattedData = scaleFeature(df)
    return formattedData, ageModel


def preProcessTest(rawData, ageModel):
    rawDataTemp = rawData[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    unknownAge = rawDataTemp[rawDataTemp.Age.isnull()].as_matrix()

    rawData.loc[(rawData.Age.isnull()), 'Age'] = ageModel.predict(
        unknownAge[:, 1:])

    formattedDataTest = setCabin(rawData)
    formattedDataTest = feature2num(formattedDataTest)
    formattedDataTest = scaleFeature(formattedDataTest)
    return formattedDataTest
