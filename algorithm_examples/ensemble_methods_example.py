"""
This file show the use of Ensemble Methods functions of sklearn library
for more info: http://scikit-learn.org/stable/modules/ensemble.html

Author: Normando Zubia
Universidad Autonoma de Chihuahua
"""

from utils import utils
from data_preprocessing import normalization

from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn import tree

#Bagging method
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor

#Boosting method
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor

#Random Forest method
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def ensemble_methods_classifiers_with_iris_dataset():
    """
    This function show the use of bagging algorithm with iris dataset
    """

    iris = datasets.load_iris()
    data_features = iris.data
    data_targets = iris.target

    # Data normalization
    data_features_normalized = normalization.z_score_normalization(data_features)

    # Data splitting
    data_features_train, data_features_test, data_targets_train, data_targets_test = utils.data_splitting(
        data_features_normalized,
        data_targets,
        0.25)

    # Model declaration
    """
    Parameters to select:

    n_estimators: The number of base estimators in the ensemble.
            Values: Random Forest and Bagging. Default 10
                    AdaBoost. Default: 50

    ###Only for Bagging and Boosting:###
    base_estimator: Base algorithm of the ensemble. Default: DecisionTree

    ###Only for Random Forest:###
    criterion: "entropy" or "gini": default: gini
    max_depth: maximum depth of tree, default: None
    """

    names = ["Bagging Classifier", "AdaBoost Classifier", "Random Forest Classifier"]

    models = [
        BaggingClassifier(
            base_estimator=tree.DecisionTreeClassifier(
                criterion='gini',
                max_depth=10)
        ),
        AdaBoostClassifier(
            base_estimator=tree.DecisionTreeClassifier(
                criterion='gini',
                max_depth=10)
        ),
        RandomForestClassifier(
            criterion='gini',
            max_depth=10
        )
    ]

    for name, em_clf in zip(names, models):
        logger.info("###################---" + name + "---###################")

        em_clf.fit(data_features_train, data_targets_train)

        # Model evaluation
        test_data_predicted = em_clf.predict(data_features_test)
        score = metrics.accuracy_score(data_targets_test, test_data_predicted)

        logger.debug("Model Score: %s", score)


def ensemble_methods_regressor_forest_dataset():

    data = utils.load_data('forestfires.csv')

    new_data = utils.convert_data_to_numeric(data, [2, 3])

    feature_vector = new_data[:, 0:-1]
    targets = new_data[:, -1]

    # Data normalization
    data_features_normalized = normalization.z_score_normalization(feature_vector)

    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(data_features_normalized,
                         targets,
                         test_size=0.25)

    # Model declaration
    """
    Parameters to select:

    n_estimators: The number of base estimators in the ensemble.
            Values: Random Forest and Bagging. Default 10
                    AdaBoost. Default: 50

    ###Only for Bagging and Boosting:###
    base_estimator: Base algorithm of the ensemble. Default: DecisionTree

    ###Only for Random Forest:###
    criterion: "entropy" or "gini": default: gini
    max_depth: maximum depth of tree, default: None
    """

    names = ["Bagging Regressor", "AdaBoost Regressor", "Random Forest Regressor"]

    models = [
        BaggingRegressor(
            base_estimator=tree.DecisionTreeRegressor(
                criterion='mse',
                max_depth=10)
        ),
        AdaBoostRegressor(
            base_estimator=tree.DecisionTreeRegressor(
                criterion='mse',
                max_depth=10)
        ),
        RandomForestRegressor(
            criterion='mse',
            max_depth=10
        )
    ]

    for name, em_reg in zip(names, models):
        logger.info("###################---" + name + "---###################")

        em_reg.fit(data_features_train, data_targets_train)

        # Model evaluation
        test_data_predicted = em_reg.predict(data_features_test)

        error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)

        logger.debug('Total Error: %s', error)

if __name__ == '__main__':

    # Classification example
    ensemble_methods_classifiers_with_iris_dataset()

    # # Regression example
    ensemble_methods_regressor_forest_dataset()
