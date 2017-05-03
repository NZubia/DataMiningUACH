import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

def convert_data_to_numeric(data):
    """
    This function convert a nominal representation to number to use the data with
    sklearn algorithms
    :param data: numpy array feature vector
    :return: numpy array with numeric data
    """
    numpy_data = data.values

    for i in range(len(numpy_data[0])):
        temp = numpy_data[:,i]
        if i == 2 or i == 3:
            dict = numpy.unique(numpy_data[:,i])
            # print(dict)
            for j in range(len(dict)):
                # print(numpy.where(numpy_data[:,i] == dict[j]))
                temp[numpy.where(numpy_data[:,i] == dict[j])] = j

        numpy_data[:,i] = temp

    return numpy_data

if __name__ == '__main__':

    data = pandas.read_csv('forestfires.csv')

    new_data = convert_data_to_numeric(data)


    feature_vector = new_data[:,0:-1]
    targets = new_data[:,-1]

    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(feature_vector,
                         targets,
                         test_size=0.25)

    # Model declaration
    """
    Parameters to select:
    criterion: "mse"
    max_depth: maximum depth of tree, default: None
    """
    dec_tree_reg = DecisionTreeRegressor(criterion='mse', max_depth=3)
    dec_tree_reg.fit(data_features_train, data_targets_train)

    # Model evaluation
    test_data_predicted = dec_tree_reg.predict(data_features_test)

    error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)

    print('Total Error: ' + str(error))