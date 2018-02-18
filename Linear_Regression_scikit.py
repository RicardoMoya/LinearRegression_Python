# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

DATASET = "./dataSet/LR_Calories_Time.csv"


def dataset_to_list_samples(dir_data_set):
    csv = np.genfromtxt(dir_data_set, delimiter=',', skip_header=True)
    y = csv[:, 0]
    x = csv[:, 1]
    return x, y


def plot_results(x, y, a, b):
    # Plot samples
    plt.scatter(x, y)

    # Plot line regression
    result = [(a * i) + b for i in x]
    plt.plot(x, result, 'r-', linewidth=3)

    # Plot x, y labels
    plt.xlabel('Tiempo (minutos)')
    plt.ylabel('Calorias')
    plt.show()


def linear_regression(dataset):
    # Read data set
    x, y = dataset_to_list_samples(dataset)
    X = np.reshape(np.asarray(x), (len(x), 1))

    # Object LinearRegression
    reg = linear_model.LinearRegression()

    # Calculate Linear Regression: calories = time * a + b
    # X: numpy array or sparse matrix of shape [n_samples,[n_features]]
    # y: numpy array of shape [n_samples, n_targets]
    reg.fit(X, y)

    # Obtain estimated coefficients: 2D array of shape (n_targets, n_features)
    a = reg.coef_[0]

    # Obtain Independent term
    b = reg.intercept_

    # Result Linear Regression
    print "Regression Function: Calories = %f · Time + %f" % (a, b)

    # Calculate Mean Squared error
    mse = mean_squared_error(y, reg.predict(X))
    print "Mean Squared Error (MSE): %f" % mse

    # Calculate Coefficient of Determination R^2
    r2 = reg.score(X, y)
    print "Coefficient of Determination R^2: %f" % r2

    # Plot Result
    plot_results(x, y, a, b)


if __name__ == '__main__':
    linear_regression(DATASET)
