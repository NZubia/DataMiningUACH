#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Normando Ali Zubia Hernandez

This file is created as helper to read files, transform data, etc
"""

import pandas
import numpy

from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    This function loads a csv file and return its numpy representation
    :param file_path: File Path
    :return: numpy array data
    """
    data = pandas.read_csv(file_path)

    return data

def data_splitting(data_features, data_targets, test_size):
    """
    This function returns four subsets that represents training and test data
    :param data: numpy array
    :return: four subsets that represents data train and data test
    """
    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(data_features,
                         data_targets,
                         test_size = test_size)

    return data_features_train, data_features_test, data_targets_train, data_targets_test


def convert_data_to_numeric(data, columns_to_convert):
    """
    This function convert a nominal representation to number to use the data with
    sklearn algorithms
    :param data: pandas feature vector
    :param columns_to_convert: array with nominals columns to convert
    :return: numpy array with numeric data
    """
    numpy_data = data.values

    for i in range(len(numpy_data[0])):
        temp = numpy_data[:, i]
        if i in columns_to_convert:
            dict = numpy.unique(numpy_data[:, i])
            for j in range(len(dict)):
                temp[numpy.where(numpy_data[:, i] == dict[j])] = j

            numpy_data[:, i] = temp

    return numpy_data
