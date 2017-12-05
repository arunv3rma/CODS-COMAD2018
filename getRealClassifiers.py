# !/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from os import getcwd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Splitting data into test and train set
def split_data(samples, ratio, seed):
    train_data, test_data = train_test_split(samples, test_size=ratio, random_state=seed)
    return train_data, test_data


# Extracting Data details
def get_data_details(train_data, test_data):
    train_points = len(train_data)
    test_points = len(test_data)
    data_dim = len(train_data.columns)
    return train_points, test_points, data_dim


# Separate features and labels from the data
def get_features_labels(train_data, test_data, data_dim):
    # Selecting Features
    train_features = train_data.ix[:, 0:data_dim - 2]
    test_features = test_data.ix[:, 0:data_dim - 2]

    # print train_features.head(1)
    train_features[data_dim - 1] = 1
    test_features[data_dim - 1] = 1

    # Separating labels from data
    train_labels = train_data.ix[:, data_dim - 1]
    test_labels = test_data.ix[:, data_dim - 1]

    return train_features, test_features, train_labels, test_labels


# SVM Classifier
def get_svm_classifier(train_features, test_features, train_labels, test_labels, test_points, penalty):
    # Getting linear SVM classifier for givern data and parameter
    svr_lin = svm.LinearSVC(C=penalty, random_state=899)
    clf = svr_lin.fit(train_features, train_labels)
    weights = list(clf.coef_[0])
    # weights.append(clf.intercept_[0])
    y_predicted = clf.predict(train_features)
    y_predicted_t = clf.predict(test_features)
    correct_predict = 0
    for i in range(test_points):
        if y_predicted_t[i] == test_labels[i]:
            correct_predict += 1

    avg_error = 1 - (correct_predict / (1.0 * test_points))

    return weights, avg_error, np.hstack((y_predicted, y_predicted_t))


# Logistic Regression Classifier
def get_lr_classifier(train_features, test_features, train_labels, test_labels, test_points, penalty):
    # Getting linear SVM classifier for given data and parameter
    log_reg = LogisticRegression(C=penalty, random_state=899)
    clf = log_reg.fit(train_features, train_labels)
    weights = list(clf.coef_[0])
    # weights.append(clf.intercept_[0])
    y_predicted = clf.predict(train_features)
    y_predicted_t = clf.predict(test_features)
    correct_predict = 0
    for i in range(test_points):
        if y_predicted_t[i] == test_labels[i]:
            correct_predict += 1

    avg_error = 1 - (correct_predict / (1.0 * test_points))
    y_probability = np.hstack((clf.predict_proba(train_features)[:, 1], clf.predict_proba(test_features)[:, 1]))
    return weights, avg_error, np.hstack((y_predicted, y_predicted_t)), y_probability


# Store bandits in csv files
def save_bandits(weights, error, y_predicted, y_probability, k, svm_classifier, increment_features, ):
    path_to_store = getcwd() + "/input/realBandits/"
    if svm_classifier:
        file_extension = "svm"
    else:
        file_extension = "lr"

    if increment_features:
        bandits_file = open(path_to_store + "bandits_i_" + file_extension + "_p.txt", 'w')
    else:
        bandits_file = open(path_to_store + "bandits_" + file_extension + "_p.txt", 'w')
    bandits = [x for _, x in sorted(zip(error, weights), reverse=True)]
    # y_predicted = [x for _, x in sorted(zip(error, y_predicted), reverse=True)]
    # y_probability = [x for _, x in sorted(zip(error, y_probability), reverse=True)]
    error = sorted(error, reverse=True)

    # Writing into file
    bandits_file.write(str(k) + "\n")
    for e in error:
        bandits_file.write(str(e) + "\t" + str(np.exp(- 2 * e)) + "\n")

    for b in bandits:
        bandits_file.write('[' + ','.join(str(e) for e in b) + "]\n")

    # for y in y_predicted:
    #     bandits_file.write('\t'.join(str(l) for l in y) + "\n")
    #
    # if not svm_classifier:
    #     for p in y_probability:
    #         # print p
    #         bandits_file.write('\t'.join(str(l) for l in p) + "\n")

    bandits_file.close()


# Ordered K classifier with same Datasets
def get_classifiers(train_data, test_data, k, svm_classifier):
    # Extracting data
    train_points, test_points, data_dim = get_data_details(train_data, test_data)
    train_features, test_features, train_labels, test_labels = get_features_labels(train_data, test_data, data_dim)

    weights = []
    error = []
    y_predicted = []
    y_probabilities = []
    for i in range(k):
        if svm_classifier:
            error_penalty = 0.0001 + 0.001 * (i * i)
            weight, error_iter, predicted_labels = get_svm_classifier(
                train_features.as_matrix(), test_features.as_matrix(), train_labels.as_matrix(),
                test_labels.as_matrix(), test_points, error_penalty)
        else:
            error_penalty = 0.01 + 0.005 * (i * i * i)
            weight, error_iter, predicted_labels, probability = get_lr_classifier(
                train_features.as_matrix(), test_features.as_matrix(), train_labels.as_matrix(),
                test_labels.as_matrix(), test_points, error_penalty)
            y_probabilities.append(probability)

        weights.append(weight)
        error.append(error_iter)
        y_predicted.append(predicted_labels)

    save_bandits(weights, error, y_predicted, y_probabilities, k, svm_classifier, increment_features=False)


# Order classifier with different features
def get_increment_classifiers(train_data, test_data, k, svm_classifier):
    # Extracting data
    train_points, test_points, data_dim = get_data_details(train_data, test_data)
    train_features, test_features, train_labels, test_labels = get_features_labels(train_data, test_data, data_dim)

    features = 0
    weights = []
    error = []
    y_predicted = []
    y_probabilities = []
    while features < k:
        error_penalty = 0.1
        if svm_classifier:
            weight, error_iter, predicted_labels = get_svm_classifier(
                train_features.ix[:, features:].as_matrix(), test_features.ix[:, features:].as_matrix(),
                train_labels.as_matrix(), test_labels.as_matrix(), test_points, error_penalty)
        else:
            weight, error_iter, predicted_labels, probability = get_lr_classifier(
                train_features.ix[:, features:].as_matrix(), test_features.ix[:, features:].as_matrix(),
                train_labels.as_matrix(), test_labels.as_matrix(), test_points, error_penalty)

            y_probabilities.append(probability)

        features += 1
        weights.append(weight)
        error.append(error_iter)
        y_predicted.append(predicted_labels)

    save_bandits(weights, error, y_predicted, y_probabilities, k, svm_classifier, increment_features=True)


# ############### Getting Arms/Classifiers for LinDM ##################

path_to_data = getcwd() + "/input/realDataset/processed/"
fileName = "news.csv"  # or "wine.csv"

data = pd.read_csv(path_to_data + fileName, header=None)

split_ratio = 0.3
arms = 20
random_seed = 89
svm_classifiers = True

train_samples, test_samples = split_data(data, split_ratio, random_seed)

# get_increment_classifiers(train_samples, test_samples, arms, svm_classifier=svm_classifiers)
get_classifiers(train_samples, test_samples, arms, svm_classifier=svm_classifiers)

