# !/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss

from os import getcwd


# ############## Algorithm for estimating classifiers #################
def linear_estimated_classifiers(contexts, classifiers, costs, algorithm_parameters):
    # Algorithm parameters
    # alpha = algorithm_parameters[0]       # Scaling Parameter
    lambda_val = algorithm_parameters[1]    # Perturbation amount
    K = algorithm_parameters[2]             # Number od Arms/Classifiers
    d = algorithm_parameters[3]             # Dimension of Context
    m = algorithm_parameters[4]             # Exploration Steps
    R = algorithm_parameters[5]             # Variance of Noise
    delta = algorithm_parameters[6]         # Probability of mis-classification
    S = np.sqrt(algorithm_parameters[7])    # Maximum norm of Theta^*
    L = np.sqrt(algorithm_parameters[8])    # Maximum value of Square of norm of Context

    mu, sigma = 0, np.sqrt(R)               # Mean and Standard Deviation of Noise

    # Initialization of Data storage
    V = np.zeros((K, d, d))             # Initialization of data matrix
    V_inv = np.zeros((K, d, d))         # Initialization of inverse of data matrix
    M = lambda_val * np.zeros((K, d))   # Initial aggregated margin*context value
    Q = lambda_val * np.zeros((K, d))   # Initial estimate of Classifiers
    arm_counter = m * np.ones(K)        # Keeps how many time arm is used
    regret = []                         # Keeps Regret of each round

    for a in range(K):
        V[a] = lambda_val * np.identity(d)

    # Initial Testing Part
    for t in range(m):
        x_t = contexts[t][:]
        I_t = K
        x_t_outer = np.outer(x_t, x_t)
        for a in range(I_t):
            V[a] += x_t_outer
            margin = np.inner(classifiers[a], x_t)
            M[a] += margin * x_t + np.random.normal(mu, sigma, 1)[0]

    # Predicting classifiers based on sufficient field testing results
    V_inv[0] = np.linalg.inv(V[0])  # As same data for all Classifiers
    for a in range(K):
        V_inv[a] = V_inv[0]
        Q[a] = V_inv[0].dot(M[a])

    # Main part: Explore and Exploit Step
    Y = np.zeros(K)         # True label
    Y_cap = np.zeros(K)     # Estimated label
    I_t = 1                 # Best arm selected by algorithm
    optimal_arm = 1         # Optimal arm

    for t in range(m, data_points):
        x_t = contexts[t][:]

        # Estimate of labels by classifiers
        for a in range(K):
            Y[a] = np.inner(classifiers[a], x_t)

            beta_t = np.sqrt(lambda_val) * S + R * np.sqrt(2 * np.log(1 / delta) +
                                                           d * np.log(1 + ((arm_counter[a] * L) / (lambda_val * d))))

            conf_half_width = beta_t * x_t.dot(V_inv[a]).dot(x_t)
            Y_cap[a] = np.inner(Q[a], x_t) + conf_half_width

        # Finding best arm to use by algorithm
        for a in range(K):
            best_arm_flag = 1
            for b in range(a + 1, K):  # ***Jump directly to mismatch arm***
                if Y_cap[a] * Y_cap[b] < 0:
                    best_arm_flag = 0
                    break

            if best_arm_flag == 1:
                I_t = a + 1
                break

        # Optimal arm using all classifiers
        for a in range(K):
            best_arm_flag = 1
            for b in range(a + 1, K):  # ***Jump directly to mismatch arm***
                if Y[a] * Y[b] < 0:
                    best_arm_flag = 0
                    break

            if best_arm_flag == 1:
                optimal_arm = a + 1
                break

        # Updating Arm selection details
        if optimal_arm == I_t:
            regret.append(0)
        else:
            r_t = costs[I_t - 1] - costs[optimal_arm - 1] + 1  # Keep this '1' for mistake regret
            regret.append(r_t)

        # Update Historical data
        for a in range(I_t):
            arm_counter[a] += 1
            V[a] += np.outer(x_t, x_t)
            V_inv[a] = np.linalg.inv(V[a])
            M[a] += Y[a] * x_t + np.random.normal(mu, sigma, 1)[0]
            Q[a] = V_inv[a].dot(M[a])

    return regret


# Mistake based regret
def accumulative_mistake_regret(instantaneous_regret):
    time_horizon = [0]
    count = 0
    dis_count = 0
    batch = data_points / 10
    mismatch = [0]
    for s in range(len(instantaneous_regret)):
        count += 1
        if instantaneous_regret[s] == 1:
            dis_count += 1
        if count == batch:
            count = 0
            time_horizon.append(time_horizon[-1] + batch)
            mismatch.append(dis_count)

    if len(instantaneous_regret) % (1.0 * batch) != 0:
        time_horizon.append(time_horizon[-1] + count)
        mismatch.append(dis_count)

    return time_horizon, mismatch


# Mistake based regret
def accumulative_cost_regret(instantaneous_regret):
    time_horizon = [0]
    count = 0
    batch_regret = 0
    batch = data_points / 10
    mismatch = [0]
    for s in range(len(instantaneous_regret)):
        count += 1
        batch_regret += instantaneous_regret[s]
        if count == batch:
            count = 0
            time_horizon.append(time_horizon[-1] + batch)
            mismatch.append(batch_regret)

    if len(instantaneous_regret) % (1.0 * batch) != 0:
        time_horizon.append(time_horizon[-1] + count)
        mismatch.append(batch_regret)

    return time_horizon, mismatch


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
def regret_plotting(regret, classifier_count):
    colors = list("rgbcmyk")
    shape = ['--*', '--+', '--H', '--d', '--X', '--^', '--v']
    # Scatter Error bar with scatter plot
    for a in range(len(classifier_count)):
        horizon, batched_regret, error = accumulative_regret_error(np.array(regret[a]))
        plt.errorbar(horizon, batched_regret, error, color=colors[a])
        plt.plot(horizon, batched_regret, colors[a] + shape[a], label=str(classifier_count[a]))

    # Location of the legend
    plt.legend(loc='upper left', numpoints=1)
    plt.title("Regret of Estimated Classifier based Algorithm for USS_Contextual")
    plt.ylabel("Expected Cumulative Regret")
    plt.xlabel("Horizon")
    plt.savefig("output/plots/regret_svm6_" + str(data_points) + ".png", bbox_inches='tight')
    plt.close()


# ####################### Reading given data #########################
# Classifiers and costs
classifiers = [
    [0.0363464616835, 0.0400968397315, -0.00202832397867, -0.000200555036311, -0.00139258805386, 0.028568173147],
    [0.275848002822, 0.301119072278, -0.0199196671776, -0.00482792794835, -0.00670114085099, 0.101690059792],
    [0.54134236337, 0.582487749707, -0.0394917046798, -0.0173918885094, -0.0142252723678, 0.131119244903],
    [0.71562621184, 0.763363739206, -0.0599438739335, -0.0282056963818, -0.0211302036632, 0.153605608978],
    [0.840588823556, 0.892090579801, -0.0798482022659, -0.0339790819425, -0.0237368208808, 0.171513666791]
    ]

classifiers2 = [classifiers[1], classifiers[3]]
classifiers3 = [classifiers[0], classifiers[2], classifiers[4]]
classifiers4 = [classifiers[0], classifiers[1], classifiers[2], classifiers[3]]

synthetic_classifiers = np.array([classifiers2, classifiers3, classifiers4, classifiers])

classifier_costs = [[0.2, 0.7],
                    [0.14, 0.43, 0.81],
                    [0.16, 0.30, 0.54, 0.9],
                    [0.12, 0.28, 0.45, 0.68, 0.95]
                    ]

# Algorithm parameters: [alpha, lambda_val, K, d, m, R, delta, S, L]
parameters = [1, 0.5, 2, 6, 1, 0.01, 0.1, 5.24, 3]

classifiers_count = []
for c in range(len(synthetic_classifiers)):
    classifiers_count.append(len(synthetic_classifiers[c]))

# Reading Synthetically generated Contextual Data
path_to_data = getcwd() + "/input/syntheticDataset/"
fileName = path_to_data + "labeled_data_25000_5.csv"
data_samples = pd.read_csv(fileName, header=None).iloc[:, 0:-1]
data_samples[len(data_samples.columns)] = 1
data_points = len(data_samples)

classifier_regret = []
for c in range(len(synthetic_classifiers)):
    file_name = open("output/regretFiles/5d_SVM/" + str(data_points) + "_" + str(classifiers_count[c]) + ".txt", 'w')
    parameters[2] = classifiers_count[c]
    run_regret = []
    for i in range(1, 21):
        data = data_samples.iloc[:, :].sample(frac=1).as_matrix()
        iter_regret = linear_estimated_classifiers(data, synthetic_classifiers[c], classifier_costs[c], parameters)
        run_regret.append(iter_regret)
        file_name.write(str(iter_regret) + "\n")
    classifier_regret.append(run_regret)
    file_name.close()

regret_plotting(classifier_regret, classifiers_count)
