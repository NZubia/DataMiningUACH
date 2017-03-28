"""
Author: Normando Ali Zubia Hernández

This file is created to explain the use of dimensionality reduction
with different tools in sklear library.

Every function contained in this file belongs to a different tool.
"""
from sklearn import datasets
from sklearn.decomposition import PCA

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

def principal_components_analysis(n_components):

    # import data
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    #First 10 rows
    print('Training Data:\n\n' + str(X[:10]))
    print('\n')
    print('Targets:\n\n' + str(Y[:10]))

    #Model declaration
    if n_components < 1:
        pca = PCA(n_components = n_components, svd_solver = 'full')
    else:
        pca = PCA(n_components=n_components)

    #Model training
    pca.fit(X)

    #Model transformation
    new_feature_vector = pca.transform(X)

    #Model information:
    print('\nModel information:\n')
    print('Number of components elected: ' + str(pca.n_components))
    print('New feature dimension: ' + str(pca.n_components_))
    print('Variance of every feature: ' + str(pca.explained_variance_ratio_))

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])

    #Print complete dictionary
    #print(pca.__dict__)

if __name__ == '__main__':
    principal_components_analysis(2)
    principal_components_analysis(.93)
