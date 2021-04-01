# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 21:38:35 2016

@author: Nik
"""
import numpy as np
import pandas as pd
from math import sqrt
import operator


location = r'C:\Users\Nik\Desktop\iris.txt'
data = pd.read_csv(location, header=None)
data.columns = [u'sepal length in cm', u'sepal width in cm', u'petal length in cm', u'petal width in cm', 'Class']


def Treining(data, proportion):
    mask = np.random.rand(len(data)) < proportion
    return data[mask], data[~mask]
train, test = Treining(data, 0.50)


def EuclideanDistance(point1,point2):
    d = [(i-j)**2 for i,j in zip(point1,point2)]
    return sqrt(sum(d))


def Neighbours(instance, train,k):
    distances = []
    for i in train.ix[:,:-1].values:
        distances.append(EuclideanDistance(instance,i))
    distances = tuple(zip(distances, train[u'Class'].values))
    return sorted(distances,key=operator.itemgetter(0))[:k]


from collections import Counter
def Response(neigbours):
    return Counter(neigbours).most_common()[0][0][1]


def Predictions(train, test, k):
    predictions = []
    for i in test.ix[:,:-1].values:
        neigbours = Neighbours(i,train,k)
        response = Response(neigbours)
        predictions.append(response)
    return predictions
