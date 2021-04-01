#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

f = open('breast-cancer-wisconsin.data', 'r')
data_lines = f.read().splitlines()
m = len(data_lines)
n = data_lines[0].count(',') - 1
X = [] # Design matrix
y = []
for line in data_lines:
    line = line.replace('?', '0')
    line_values = line.split(',')
    X.append(line_values[1:len(line_values) - 1])
    y.append(line_values[-1])
X = np.array(X, dtype = np.int32)
y = np.array(y, dtype = np.int32)

dim = X.shape[0]
train_n = int(dim * 9 / 10)
X_train = X[:train_n, :]
y_train = y[:train_n]
X_test = X[train_n:, :]
y_test = y[train_n:]

clf = svm.SVC()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

successes = 0
for right_value, prediction in zip(y_test, predictions):
    if right_value == prediction:
        successes += 1

print "Accuracy: %0.2f" % (float(successes) / float(y_test.size) * 100)
