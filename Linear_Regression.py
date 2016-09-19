# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASET = "./dataSet/RL_Calorias_Tiempo.csv"
ITERATIONS = 1000


def print_results():
    pass


def plot_results(x, y):
    # Plot samples
    plt.scatter(x, y)

    # Plot line regression
    # TODO

    # Plot x, y labels
    plt.xlabel('Tiempo (minutos)')
    plt.ylabel('Calorias')
    plt.show()


def linear_regression(dataset, iterations):
    # Read data set
    samples = pd.read_table(dataset, engine='python', sep='::')
    for i, r in samples.iterrows():
        print str(r['Calorias']) + '::' + str(r['Tiempo'])

    plot_results(samples['Tiempo'].tolist(), samples['Calorias'].tolist())


if __name__ == '__main__':
    linear_regression(DATASET, ITERATIONS)
