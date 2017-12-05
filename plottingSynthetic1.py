# !/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


# Getting Average regret and Confidence interval
def accumulative_regret_error(regret):
    time_horizon = [0]
    samples = len(regret[0])
    runs = len(regret)
    batch = samples / 10

    # Time horizon
    t = 0
    while True:
        t += 1
        if time_horizon[-1] + batch > samples:
            if time_horizon[-1] != samples:
                time_horizon.append(time_horizon[-1] + samples % batch)
            break
        time_horizon.append(time_horizon[-1] + batch)

    # Mean batch regret of R runs
    avg_batched_regret = []
    for r in range(runs):
        count = 0
        accumulative_regret = 0
        batch_regret = [0]
        for s in range(samples):
            count += 1
            accumulative_regret += regret[r][s]
            if count == batch:
                batch_regret.append(accumulative_regret)
                count = 0

        if samples % batch != 0:
            batch_regret.append(accumulative_regret)
        avg_batched_regret.append(batch_regret)

    regret = np.mean(avg_batched_regret, axis=0)

    # Confidence interval
    conf_regret = []
    freedom_degree = runs - 2
    for r in range(len(avg_batched_regret[0])):
        conf_regret.append(ss.t.ppf(0.95, freedom_degree) * ss.sem(np.array(avg_batched_regret)[:, r]))
    return time_horizon, regret, conf_regret


# Regret Plotting
def regret_plotting(regret, classifier_count, data_points):
    colors = list("rgbcmyk")
    # shape = ['--H', '--d', '--X', '--^', '--v', '--*', '--+']
    shape = ['--^', '--d', '--v']

    # Scatter Error bar with scatter plot
    for a in range(len(classifier_count)):
        horizon, batched_regret, error = accumulative_regret_error(np.array(regret[a]))
        plt.errorbar(horizon, batched_regret, error, color=colors[a])
        plt.plot(horizon, batched_regret, colors[a] + shape[a], label='K=' + str(classifier_count[a]))

    # Location of the legend
    plt.legend(loc='upper left', numpoints=1)
    # plt.title("Cumulative Regret for different numbers of classifiers for " + str(data_points) + " samples")
    plt.ylabel("Cumulative Regret")
    plt.xlabel("Number of Samples")
    plt.savefig("output/final_plot/simple_" + str(data_points) + ".png", bbox_inches='tight')
    plt.close()


# Reading Data
# data_plot = [1000, 5000, 10000, 25000, 50000, 100000]
data_plot = [5000, 50000]
classifiers_count = [2, 3, 5]
# classifiers_count = [2]
runs = 20

# Reading Files
for d in range(len(data_plot)):
    classifiers_regret = []
    for c in classifiers_count:
        fileName = "output/regretFiles/2d/" + str(data_plot[d]) + "_" + str(c) + ".txt"
        resultFile = open(fileName)
        classifier_regret = []
        for r in range(runs):
            regrets = map(float, list(resultFile.readline().split("[")[1].split("]")[0].split(", ")))
            classifier_regret.append(regrets)
        classifiers_regret.append(classifier_regret)
        resultFile.close()

    regret_plotting(classifiers_regret, classifiers_count, data_plot[d])
