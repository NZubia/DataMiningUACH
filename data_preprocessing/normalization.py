#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Normando Ali Zubia Hernandez

This file is created to explain the use of normalization
with different tools in sklearn library.

Every function contained in this file belongs to a different tool.
"""

from sklearn import preprocessing
import logging

logger = logging.getLogger(__name__)


def z_score_normalization(feature_vector):
    """
    This functions standardize a numpy array
    :param feature_vector: Numpy array to be standardizing
    :return: Standardized numpy array
    """

    # Data standardization
    standardized_data = preprocessing.scale(feature_vector)

    # First 10 rows of new feature vector
    logger.debug('New feature vector: %s', standardized_data[:10])

    return standardized_data


def min_max_normalization(feature_vector):
    """
    This function uses min-max scaler to normalize a numpy array
    :param feature_vector: Numpy array to be standardizing
    :return: Normalized numpy array
    """

    # Data normalization
    min_max_scaler = preprocessing.MinMaxScaler()

    min_max_scaler.fit(feature_vector)

    # Model information:
    logger.debug('Model information:')
    logger.debug('Data min: %s', min_max_scaler.data_min_)
    logger.debug('Data max: %s', min_max_scaler.data_max_)

    normalized_data = min_max_scaler.transform(feature_vector)

    # First 10 rows of new feature vector
    logger.debug('New feature vector: %s', normalized_data[:10])

    return normalized_data
