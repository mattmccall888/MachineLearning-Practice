import numpy as np
import pandas as pd
#from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

#Load the iris dataset
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
iris = pd.read_csv("iris.data", names=names)
X, y = iris.data, iris.target

#Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

from KNNtest import KNN
clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print("KNN classification accuracy:", clf.accuracy(y_test, predictions))