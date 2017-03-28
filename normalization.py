from sklearn import preprocessing
from sklearn import datasets

def z_score_normalization():
    # import data
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # First 10 rows
    print('Training Data:\n\n' + str(X[:10]))
    print('\n')
    print('Targets:\n\n' + str(Y[:10]))

    # Data standarization
    standardized_data = preprocessing.scale(X)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(standardized_data[:10])

def min_max_scaler():
    # import data
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # First 10 rows
    print('Training Data:\n\n' + str(X[:10]))
    print('\n')
    print('Targets:\n\n' + str(Y[:10]))

    # Data normalization
    min_max_scaler = preprocessing.MinMaxScaler()

    min_max_scaler.fit(X)

    # Model information:
    print('\nModel information:\n')
    print('Data min: ' + str(min_max_scaler.data_min_))
    print('Data max: ' + str(min_max_scaler.data_max_))

    new_feature_vector = min_max_scaler.transform(X)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])


if __name__ == '__main__':
    # z_score_normalization()
    min_max_scaler()