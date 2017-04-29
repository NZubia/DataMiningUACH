#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Normando Ali Zubia Hernandez

This file is created to explain the use of dimensionality reduction
with different tools in sklearn library.

Every function contained in this file belongs to a different tool.
"""
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import logging

logger = logging.getLogger(__name__)


def principal_components_analysis(n_components, feature_vector):
    """
    This functions transform a numpy array using PCA algorithm
    :param n_components: number of components in new feature vector, i
    f n_components < 1 then a variance percentage is used
    :param feature_vector: numpy array to be transformed
    :return: PCA transformation in numpy array form
    """

    # Model declaration
    if n_components < 1:
        pca = PCA(n_components=n_components, svd_solver='full')
    else:
        pca = PCA(n_components=n_components)

    # Model training
    pca.fit(feature_vector)

    # Model transformation
    new_feature_vector = pca.transform(feature_vector)

    # Model information:
    logger.debug('Model information:')
    logger.debug('Number of components elected: %s', pca.n_components)
    logger.debug('New feature dimension: %s', pca.n_components_)
    logger.debug('Variance of every feature: %s', pca.explained_variance_ratio_)

    # First 10 rows of new feature vector
    logger.debug('New feature vector: %s', new_feature_vector[:10])

    return new_feature_vector


def attribute_subset_selection_with_trees(feature_vector, targets, type):
    """
    This function use gain info to select the best features of a numpy array
    :param type: Classification or Regression
    :param feature_vector: Numpy array to be transformed
    :param targets: feature vector targets
    :return: Numpy array with the selected features
    """

    if type == 'Regression':
        extra_tree = ExtraTreesRegressor()
    else:
        # Model declaration
        extra_tree = ExtraTreesClassifier()

    # Model training
    extra_tree.fit(feature_vector, targets)

    # Model information:
    logger.debug('Model information:')

    # display the relative importance of each attribute
    logger.debug('Importance of every feature: %s', extra_tree.feature_importances_)

    # If model was training before prefit = True
    model = SelectFromModel(extra_tree, prefit=True)

    # Model transformation
    new_feature_vector = model.transform(feature_vector)

    # First 10 rows of new feature vector
    logger.debug('New feature vector: %s', new_feature_vector[:10])

    return new_feature_vector


def recursive_feature_elimination(n_attributes, feature_vector, targets):
    """
    This function use recursive feature elimination to select the best
    features of a numpy array
    :param n_attributes: attributes number to be elected
    :param feature_vector: Numpy array to be transformed
    :param targets: feature vector targets
    :return: Numpy array with the selected features
    """

    # Create a base classifier used to evaluate a subset of attributes
    # model_eval = ExtraTreesClassifier()

    # Note: Feature selection change with different models
    model_eval = LogisticRegression()

    # Create the RFE model
    rfe = RFE(model_eval, n_attributes)
    rfe = rfe.fit(feature_vector, targets)

    # Summarize the selection of the attributes
    # Model information:
    logger.debug('Model information:')
    logger.debug('New feature dimension: %s', rfe.n_features_)
    logger.debug('Feature Ranking: %s', rfe.ranking_)
    logger.debug('Selected features: %s', rfe.support_)

    # Model transformation
    new_feature_vector = rfe.transform(feature_vector)

    # First 10 rows of new feature vector
    logger.debug('New feature vector: %s', new_feature_vector[:10])

    return new_feature_vector


def select_k_best_features(n_attributes, feature_vector, targets):
    """
    This function use select k best features to select the best
    features of a numpy array
    :param n_attributes: attributes number to be elected
    :param feature_vector: Numpy array to be transformed
    :param targets: feature vector targets
    :return: Numpy array with the selected features
    """

    # Model declaration
    kbest = SelectKBest(score_func=chi2, k=n_attributes)

    # Model training
    kbest.fit(feature_vector, targets)

    # Summarize the selection of the attributes
    # Model information:
    logger.debug('Model information:')
    logger.debug('Feature Scores: %s', kbest.scores_)

    # Model transformation
    new_feature_vector = kbest.transform(feature_vector)

    # First 10 rows of new feature vector
    logger.debug('New feature vector: %s', new_feature_vector[:10])

    return new_feature_vector
