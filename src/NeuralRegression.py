#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:19:40 2017

@author: dhingratul
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from keras.layers.advanced_activations import LeakyReLU
from sklearn.svm import SVR


def baselineScores(X_train, y_train, X_test, y_test):
    # Optimized for top 20%
    opt1 = SVR(kernel='rbf', C=73.77654120502568, gamma=1.1539713541666757)
    opt2 = SVR(kernel='rbf', C=18.885695893079657, gamma=18.38411458333333)
    opt3 = SVR(kernel='rbf', C=25.384843416756983, gamma=1.170620957526686)
    # Linear Model
    lin = SVR(kernel='linear', C=10, epsilon=0.2)
    # Predictions
    clf1 = lin.fit(X_train, y_train)
    clf2 = opt1.fit(X_train, y_train)
    clf3 = opt2.fit(X_train, y_train)
    clf4 = opt3.fit(X_train, y_train)
    c1 = clf1.predict(X_test)
    c2 = clf2.predict(X_test)
    c3 = clf3.predict(X_test)
    c4 = clf4.predict(X_test)
    mse1 = mean_squared_error(y_test, c1)
    mse2 = mean_squared_error(y_test, c2)
    mse3 = mean_squared_error(y_test, c3)
    mse4 = mean_squared_error(y_test, c4)
    return mse1, mse2, mse3, mse4


def nonLinearModel():
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal',
                    activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Data
direc = '../data/'
file = direc + 'housing.csv'
df = pd.read_csv(file, delim_whitespace=True, header=None)
# split into X and y
X = df.iloc[:, 0:13].as_matrix()
y = df.iloc[:, 13].as_matrix()
num = X.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=0)

seed = 7
np.random.seed(seed)
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)
clf = KerasRegressor(build_fn=nonLinearModel, nb_epoch=10, batch_size=5,
                     verbose=1)
clf.fit(X_train, y_train)
res = clf.predict(X_test)
mse = mean_squared_error(y_test, res)
print("\n MSE DL:", mse)
mse1, mse2, mse3, mse4 = baselineScores(X_train, y_train, X_test, y_test)
print("\n MSE Lin:", mse1)
print("\n MSE OPT1:", mse2)
print("\n MSE OPT2:", mse3)
print("\n MSE OPT3:", mse4)
