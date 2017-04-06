"""
This file show the use of Decision Tree function of sklearn library
for more info: http://scikit-learn.org/stable/modules/tree.html

Author: Normando Zubia
Universidad Autonoma de Chihuahua
"""

import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
from sklearn import tree

import pydotplus
#
# from sklearn.model_selection import cross_val_score

# def load_data(file_path):
#     """
#     This function loads a csv file and return its numpy representation
#     :param file_path: File Path
#     :return: numpy array data
#     """
#     data = pandas.read_csv(file_path)
#
#     return data.values
#
# def convert_data_to_numeric(data):
#     """
#     This function convert a nominal representation to number to use the data with
#     sklearn algorithms
#     :param data: numpy array feature vector
#     :return: numpy array with numeric data
#     """
#     numpy_data = data.values
#
#     for i in range(len(numpy_data[0])):
#         temp = numpy_data[:,i]
#         dict = numpy.unique(numpy_data[:,i])
#         # print(dict)
#         for j in range(len(dict)):
#             # print(numpy.where(numpy_data[:,i] == dict[j]))
#             temp[numpy.where(numpy_data[:,i] == dict[j])] = j
#
#         numpy_data[:,i] = temp
#
#     return numpy_data

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

def decision_tree_training():
    """
    This function train and return a decision tree model
    :param data: numpy array
    :return: decision tree model
    """

    iris = datasets.load_iris()
    data_features = iris.data
    data_targets = iris.target

    #Data splitting
    data_features_train, data_features_test, data_targets_train, data_targets_test = data_splitting(data_features, data_targets, 0.25)

    #Model declaration
    """
    Parameters to select:
    criterion: "entropy" or "gini": default: gini
    max_depth: maximum depth of tree, default: None
    """
    dec_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    dec_tree.fit(data_features_train, data_targets_train)

    #Model evaluation
    test_data_predicted = dec_tree.predict(data_features_test)
    score = metrics.accuracy_score(data_targets_test, test_data_predicted)

    print("Model Score: " + str(score))
    print("Probability of each class: \n")
    #Measure probability of each class
    prob_class = dec_tree.predict_proba(data_features_test)
    print(prob_class)

    print("Feature Importance: \n")
    print(dec_tree.feature_importances_)

    # Draw the tree
    dot_data = tree.export_graphviz(dec_tree, out_file=None,
                                         feature_names=iris.feature_names,
                                         class_names=iris.target_names,
                                         filled=True, rounded=True,
                                         special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("iris.pdf")

if __name__ == '__main__':
    decision_tree_training()
