# !/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from os import getcwd


# Generating n random numbers
def generate_data(n, d, scale):

    random_data = pd.DataFrame()
    for num in range(0, n):

        random_numbers = []
        for i in range(d):
            random_numbers.append(np.random.uniform(-scale, scale, 1)[0])

        row = pd.DataFrame([random_numbers])
        random_data = pd.concat([random_data, row], ignore_index=True)

    return random_data


# Assigning labels to data
def assign_labels(data, data_samples):
    labels = np.ones(data_samples, int)
    for i in range(data_samples):
        if data[0][i] + data[1][i] + data[2][i] * data[3][i] + data[4][i]*data[4][i] < 0:
            labels[i] = -1

    data[len(data.columns)] = labels
    return data


# ######################## Random data generation ################################

# Data details
dimension = 5
samples = 1000
data_range = 1.0

synthetic_data = assign_labels(generate_data(samples, dimension, data_range), samples)

# File storing location
path_to_save = getcwd() + "/input/syntheticDataset/"
# file_name = path_to_save + "data_" + str(samples) + "_" + str(data_range) + "_" + str(dimension) + ".csv"
file_name = path_to_save + "labeled_data_" + str(samples) + "_" + str(dimension) + ".csv"
pd.DataFrame(synthetic_data).to_csv(file_name, index=False, header=False)
