# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import random
import math
import pandas as pd
import matplotlib.pyplot as plt

DATASET = "./dataSet/RL_Calorias_Tiempo.csv"
ALPHA = 0.0005
MAX_ITERATIONS = 100
CONVERGENCE_TOLERANCE = 0.01


def print_results(a, b, error):
    print '\n----------------------------'
    print '\nFINAL RESULTS'
    print '\t Calorias(Tiempo) = %f * Tiempo + %f' % (a, b)
    print '\t\tTotal Error = %f' % error


def print_iteration_status(it_counter, a, b, error):
    print '\nITERATION %d' % it_counter
    print '\t Y = %f X + %f' % (a, b)
    print '\t\tError Iteration = %f' % error


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


def get_hipotesis(a, b, x):
    return (a * x) + b


def get_diff(hipotesis, y):
    return hipotesis - y


def get_error(samples, a, b):
    error = 0
    for i, r in samples.iterrows():
        error += math.pow((get_hipotesis(a, b, r['Tiempo']) - r['Calorias']), 2)
    return error


def is_convergence(a, b, a_old, b_old, convergence):
    return math.fabs(a - a_old) < convergence \
           and math.fabs(b - b_old) < convergence


def linear_regression(dataset, iterations, convergence):
    # Read data set
    samples = pd.read_csv(dataset)

    # Random parameters
    a, b = (random.randrange(-10, 10),) * 2
    a_old, b_old = (convergence,) * 2
    it_counter = 0

    while not is_convergence(a, b, a_old, b_old, convergence) \
            and it_counter < iterations:
        a_old = a
        b_old = b
        sum_a, sum_b = (0,) * 2
        for i, r in samples.iterrows():
            h = get_hipotesis(a_old, b_old, r['Tiempo'])
            diff = get_diff(h, r['Calorias'])
            sum_a += diff * r['Tiempo']
            sum_b += diff
        a = a_old - ((ALPHA / len(samples)) * sum_a)
        b = b_old - ((ALPHA / len(samples)) * sum_b)

        it_counter += 1
        print_iteration_status(it_counter, a, b, get_error(samples, a, b))

    print_results(a, b, get_error(samples, a, b))
    plot_results(samples['Tiempo'].tolist(), samples['Calorias'].tolist(), a, b)


if __name__ == '__main__':
    linear_regression(DATASET, MAX_ITERATIONS, CONVERGENCE_TOLERANCE)
