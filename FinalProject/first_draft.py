"""
This file show the use of several algorithms to solve
Sberbank Russian Housing Market competition

Author: Normando Zubia
Universidad Autonoma de Chihuahua
"""

from utils import utils
from data_preprocessing import normalization
import numpy
import pandas
import matplotlib.pyplot as plt
import csv

from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn import metrics
from sklearn import model_selection

if __name__ == '__main__':
    # Open data
    utils.logger.info("DATA LOADING...")
    train_data = utils.load_data("../data/train.csv")

    # Outlier Example
    plt.boxplot(train_data['full_sq'])
    plt.show()

    # Get current column ordering to order concat dataset
    right_order = []
    for column in train_data.columns:
        right_order.append(column)

    test_data = utils.load_data("../data/test.csv")

    # Merge datasets
    frames = [train_data, test_data]

    complete_dataset = pandas.concat(frames)
    complete_dataset = complete_dataset[right_order]

    # Defining attribute type columns:
    nominal_columns = [1, 11, 12, 29, 33, 34, 35, 36, 37, 38, 39, 40, 106,
                       114, 118, 152]


    numeric_columns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18,
                       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32,
                       41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                       54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                       67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                       80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
                       93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
                       105, 107, 108, 109, 110, 111, 112, 113, 115, 116, 117,
                       119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                       130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
                       141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
                       153, 154, 155, 156, 157, 158, 159, 160, 161, 162,
                       163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
                       174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184,
                       185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
                       196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206,
                       207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217,
                       218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228,
                       229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
                       240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250,
                       251, 252, 253, 254, 254, 255, 256, 257, 258, 259, 260,
                       261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
                       272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282,
                       283, 284, 285, 286, 287, 288, 289, 290]
    target = [291]

    utils.logger.info("FILLING MISSING VALUES WITH CONSTANT")

    for i in numeric_columns:

        complete_dataset.fillna(value='missing')

    # Convert data
    numpy_data = complete_dataset.values

    utils.logger.info("REPLACING MISSING VALUES WITH CONSTANT")

    # Replace values
    for i in range(291):
        if i in nominal_columns:
            for j in range(len(numpy_data)):
                if type(numpy_data[j][i]) is float:
                    numpy_data[j][i] = 'Missing'

        if i in numeric_columns:
            for j in range(len(numpy_data)):
                # if pandas.isnull(numpy_data[j][i]):
                #     numpy_data[j][i] = -1.0
                if numpy.isnan(numpy_data[j][i]):
                    numpy_data[j][i] = -1.0

    utils.logger.info("TRANSFORMING NOMINAL DATA WITH NUMERICAL")

    # Conver data to numerical
    for i in range(len(numpy_data[0])):
        temp = numpy_data[:, i]
        dict = numpy.unique(numpy_data[:, i])

        if i in nominal_columns:
            # utils.logger.debug("Dict column %d: %s ", i, dict)
            for j in range(len(dict)):
                # print(numpy.where(numpy_data[:,i] == dict[j]))
                temp[numpy.where(numpy_data[:, i] == dict[j])] = j

        numpy_data[:, i] = temp

    # Split features and target
    train_dataset = numpy_data[0:30470]
    test_dataset = numpy_data[30471: len(numpy_data)]

    feature_vector = train_dataset[:, 1:-1]
    targets = train_dataset[:, -1]

    test_temp = test_dataset[:, 1:-1]

    # Data normalization
    data_features_normalized = normalization.z_score_normalization(feature_vector)

    # Data splitting
    data_features_train, data_features_test, data_targets_train, data_targets_test = utils.data_splitting(
        data_features_normalized,
        targets,
        0.25)

    # Algorithms declaration
    names = [
        "Bagging_Regressor",
        "AdaBoost_Regressor",
        "Random_Forest_Regressor",
        "Neural_Network_Regressor",
        "Decision_Tree_Regressor",
        "Support_Vector_Machine_Regressor",
        "K-Neighbor_Regressor"
    ]

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
        ),
        MLPRegressor(
            hidden_layer_sizes=(50),
            activation="relu",
            solver="adam"
        ),
        tree.DecisionTreeRegressor(
            criterion='mse'
        ),
        SVR(
            kernel='rbf',
            C=1e3,
            gamma=0.1
        ),
        KNeighborsRegressor()
    ]

    # Algorithm implementation
    for name, em_clf in zip(names, models):
        utils.logger.info("###################---" + name + "---###################")

        em_clf.fit(data_features_train, data_targets_train)

        # Model evaluation
        test_data_predicted = em_clf.predict(data_features_test)

        # Cross validation
        scores = model_selection.cross_val_score(em_clf, data_features_normalized, targets, cv=10)
        mean_error = scores.mean()

        utils.logger.debug('Cross validation result: %s', mean_error)

        # Get predictions to Kaggle
        kaggle_predictions = em_clf.predict(test_dataset[:, 1:-1])

        # Generate CSV for Kaggle with csv package:
        path = "../data/predicted_kaggle_" + str(name) +".csv"
        # with open(path, "w") as csv_file:
        #     writer = csv.writer(csv_file, delimiter=',')
        #     writer.writerow(["id", "price_doc"])
        #
        #     for i in range(len(kaggle_predictions)):
        #         writer.writerow([test_dataset[i][0], kaggle_predictions[i]])

        # Generate CSV for Kaggle with pandas (easiest way)
        df_predicted = pandas.DataFrame({'id': test_dataset[:,0], 'price_doc': kaggle_predictions})

        df_predicted.to_csv(path, index=False)

        error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)

        utils.logger.debug('Total Error: %s', error)