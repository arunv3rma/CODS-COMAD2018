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
    plt.savefig("output/plots/regret_wine_" + str(data_points) + ".png", bbox_inches='tight')
    plt.close()


# ####################### Reading given data #########################
# Classifiers and costs
classifiers_list = [
    [-0.112686061428, -1.73520192549, 0.0538283226732, 0.0134859259498, -0.106845726172, 0.00216131175246,
     -0.00434902396605, -0.478159973456, -0.274163169373, 0.447218598088, 0.329282883481, -0.475466457654],
    [-0.0376906988752, -0.0144552233885, 0.000416889751187, -0.00111307744918, -0.00166563737742, 0.00810073086017,
     -0.00478788762534, -0.00519871283191, -0.0113747650396, 0.00348772519312, 0.0981892928749, -0.00505996442695],
    [-0.111835169812, -1.35837971432, 0.0549965308494, 0.0179187637036, -0.084787297418, 0.0103611121291,
     -0.000312782634392, -0.369560615301, -0.312507085616, 0.35681966802, 0.31200423107, -0.368627692319],
    [-0.104743978936, -2.05024877094, 0.0200071504778, 0.0199910141274, -0.120484161349, 0.0153946478341,
     -0.00134387713487, -0.588637838883, -0.240065102631, 0.525102515497, 0.337147096239, -0.582270206906],
    [-0.111942521976, -1.93228257835, 0.0362215910379, 0.0215206853346, -0.11644084628, 0.0147437514062,
     -0.00282567879571, -0.541299338245, -0.255712699305, 0.491351990604, 0.339417530119, -0.536717968536]
    ]

classifiers2 = [classifiers_list[0], classifiers_list[2]]
classifiers3 = [classifiers_list[0], classifiers_list[2], classifiers_list[4]]
classifiers4 = [classifiers_list[0], classifiers_list[1], classifiers_list[3], classifiers_list[4]]

synthetic_classifiers = np.array([classifiers2, classifiers3, classifiers4, classifiers_list])

classifier_costs = [[0.2, 0.45],
                    [0.2, 0.45, 0.95],
                    [0.2, 0.30, 0.68, 0.95],
                    [0.2, 0.30, 0.45, 0.68, 0.95]
                    ]

# Algorithm parameters: [alpha, lambda_val, K, d, m, R, delta, S, L]
parameters = [1, 0.5, 2, 12, 1, 0.01, 0.1, 5.4, 277291.231836]

classifiers_count = []
for c in range(len(synthetic_classifiers)):
    classifiers_count.append(len(synthetic_classifiers[c]))

# Reading Synthetically generated Contextual Data
path_to_data = getcwd() + "/input/realDataset/processed/"
fileName = path_to_data + "wine.csv"
data_samples = pd.read_csv(fileName, header=None).iloc[:, 0:-1]
data_samples[len(data_samples.columns)] = 1
data_points = len(data_samples)

classifier_regret = []
for c in range(len(synthetic_classifiers)):
    file_name = open("output/regretFiles/wine/" + str(data_points) + "_" + str(classifiers_count[c]) + ".txt", 'w')
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
