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