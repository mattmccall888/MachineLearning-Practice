#This is the KNN/ K-nearest neighbors algorithm built from scratch
#This algorithm is used for classification problems, the one of the two kinds of specific machine learning problems

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pandas import read_csv
#from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])


#In order to calculate distances, we use euclidan distance, which is calculated as follows:
#d = sqrt(sum from i=1 to n (qi-pi)^2)
#This is the distance between two points in a 2D plane



class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        #This method will return a list of predictions for the test set
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):
        #This method will return a single prediction for a single data point
        #Compute distances
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        #Get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        #Majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def _euclidean_distance(self, x1, x2):
        #This method will return the euclidean distance between two points
        return np.sqrt(np.sum((x1-x2)**2))
    
    def accuracy(self, y_true, y_pred):
        #This method will return the accuracy of the model
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
#Load the iris dataset
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
iris = read_csv("C:/Users/mattb/CompSci/MachineLearning-Practice/algorithms/KNN#1/iris.data", names=names)
array = iris.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 1

X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=validation_size, random_state=seed)
clf = KNN(k=13)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_validation)

print("KNN classification accuracy:", clf.accuracy(Y_validation, predictions))



