#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Normando Ali Zubia Hernandez

This file is created as helper to read files, transform data, etc
"""

import pandas
import numpy


def load_data(file_path):
    """
    This function loads a csv file and return its numpy representation
    :param file_path: File Path
    :return: numpy array data
    """
    data = pandas.read_csv(file_path)

    return data


def convert_data_to_numeric(data):
    """
    This function convert a nominal representation to number to use the data with
    sklearn algorithms
    :param data: pandas feature vector
    :return: numpy array with numeric data
    """
    numpy_data = data.values

    for i in range(len(numpy_data[0])):
        temp = numpy_data[:, i]
        dict = numpy.unique(numpy_data[:, i])
        for j in range(len(dict)):
            temp[numpy.where(numpy_data[:, i] == dict[j])] = j

        numpy_data[:, i] = temp

    return numpy_data
