# !/usr/bin/env python
# coding: utf-8

import pandas as pd

from os import getcwd

# File storing and reading location
path_to_save = getcwd() + "/input/realDataset/processed/"
path_to_data = getcwd() + "/input/realDataset/actual/"

fileName = "OnlineNewsPopularity.csv"
data = pd.read_csv(path_to_data + fileName, header=None, sep=", ")

valid_data = data.ix[1:, 2:].as_matrix()

for d in range(len(valid_data)):
    if int(valid_data[d, -1]) <= 1400:
        valid_data[d, -1] = -1
    else:
        valid_data[d, -1] = 1

file_name = path_to_save + "news.csv"
pd.DataFrame(valid_data).to_csv(file_name, index=False, header=False)


fileName = "winequality-white.csv"
data = pd.read_csv(path_to_data + fileName, header=None, sep=";")
valid_data = data.ix[1:, :].as_matrix()
for d in range(len(valid_data)):
    if int(valid_data[d, -1]) <= 5:
        valid_data[d, -1] = -1
    else:
        valid_data[d, -1] = 1

file_name = path_to_save + "wine.csv"
pd.DataFrame(valid_data).to_csv(file_name, index=False, header=False)

fileName = "winequality-red.csv"
data = pd.read_csv(path_to_data + fileName, header=None, sep=';')
