#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:23:45 2017

@author: dhingratul
"""
import pandas as pd
import optunity
import optunity.metrics
import sklearn.svm
from sklearn.cross_validation import train_test_split


# Data
direc = '../data/'
file = direc + 'housing.csv'
df = pd.read_csv(file, delim_whitespace=True, header=None)
# split into X and y
X = df.iloc[:, 0:13].as_matrix()
y = df.iloc[:, 13].as_matrix()
num = X.shape[1]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=0)

# SVC
outer_cv = optunity.cross_validated(x=x_train, y=y_train, num_folds=3)
space = {'kernel': {'linear': {'C': [0, 100]},
                    'rbf': {'gamma': [0, 50], 'C': [1, 100]},
                    'poly': {'degree': [2, 5], 'C': [1000, 20000],
                             'coef0': [0, 1]}
                    }
         }


def compute_mse_all_tuned(x_train, y_train, x_test, y_test):
    """Computes MSE of an SVR with RBF kernel and optimized hyperparameters."""

    # define objective function for tuning
    @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
    def tune_cv(x_train, y_train, x_test, y_test, kernel, C, gamma, degree,
                coef0):
        if kernel == 'linear':
            model = sklearn.svm.SVR(kernel=kernel, C=C)
        elif kernel == 'poly':
            model = sklearn.svm.SVR(kernel=kernel, C=C, degree=degree,
                                    coef0=coef0)
        elif kernel == 'rbf':
            model = sklearn.svm.SVR(kernel=kernel, C=C, gamma=gamma)
        else:
            raise ArgumentError("Unknown kernel function: %s" % kernel)
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        return optunity.metrics.mse(y_test, predictions)

    # optimize parameters
    optimal_pars, _, _ = optunity.minimize_structured(tune_cv, num_evals=150,
                                                      search_space=space)

    # remove hyperparameters with None value from optimal pars
    for k, v in list(optimal_pars.items()):
        if v is None:
            del optimal_pars[k]
    print("optimal hyperparameters: " + str(optimal_pars))

    tuned_model = sklearn.svm.SVR(**optimal_pars).fit(x_train, y_train)
    predictions = tuned_model.predict(x_test)
    return optunity.metrics.mse(y_test, predictions)


# wrap with outer cross-validation
compute_mse_all_tuned = outer_cv(compute_mse_all_tuned)
print(compute_mse_all_tuned())
