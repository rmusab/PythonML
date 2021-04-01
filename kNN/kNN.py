#!/usr/bin/python
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

def get_k_neighbors(v, k, X):
    m = len(X)
    diff = [] # List of differences of v with all the training examples
    X1 = np.array(X, dtype = np.float32)
    for i in range(m):
        diff.append((np.sqrt(sum((v - X1[i, :]) ** 2)), i))
    diff_sorted = sorted(diff, key = lambda tup: tup[0])
    result = []
    for i in range(k):
        result.append(diff_sorted[i])
    return result

def shuffle(X, y):
    for i in range(len(X)):
        rand_index = random.randint(0, len(X) - 1)
        temp = X[i]
        X[i] = X[rand_index]
        X[rand_index] = temp
        temp = y[i]
        y[i] = y[rand_index]
        y[rand_index] = temp

def detect_class(v, k, X, y):
    neighbors = get_k_neighbors(v, k, X)
    neighbor_classes = {}
    for i in range(len(neighbors)):
        current_class = y[neighbors[i][1]]
        if current_class not in neighbor_classes.keys():
            neighbor_classes[current_class] = 0
        neighbor_classes[current_class] += 1
    print neighbor_classes
    return neighbor_classes.keys()[neighbor_classes.values().index(max(
        neighbor_classes.values()))]

def print_table_out(X, y):
    row_format = '{:<3}' + '{:>12}' * (len(X[0])) + '{class_:>20}'
    print row_format.format('No.', 'Sepal L', 'Sepal W', 'Petal L',
                            'Petal W', class_ = 'Class')
    for i, row_num_vals, cl in zip(range(0, len(X)), X, y):
        print row_format.format(i, *row_num_vals, class_ = cl)

def plot_object(v, i, y, pattern = 'o'):
    if y[i] == 'Iris-setosa':
        plt.plot(v[0], v[1], 'r' + pattern)
    elif y[i] == 'Iris-versicolor':
        plt.plot(v[0], v[1], 'g' + pattern)
    else:
        plt.plot(v[0], v[1], 'b' + pattern)

f = open('iris.txt', 'r')
data_lines = f.read().splitlines()
m = len(data_lines) # Number of training examples in data
n = data_lines[0].count(',') # Number of features
X = [] # Design matrix
y = [] # Vector of classes for each respective training example
for i, line in enumerate(data_lines):
    line_values = line.split(',')
    X.append([float(value) for value in line_values[:n]])
    y.append(line_values[n])
print_table_out(X, y);
print '-' * 80
shuffle(X, y)
print_table_out(X[:100], y[:100])
print '-' * 80

k = input('Enter k --> ')
print ''
correct_num = 0
y1 = []
for i in range(0, 100):
    y1.append(0)
for i in range(100, 150):
    print 'Test example no. {0:d}'.format(i)
    print 'Features vector: {!r}'.format(X[i])
    detected_class = detect_class(X[i], k, X[:100], y[0:100])
    y1.append(detected_class)
    print 'Predicted class: {}\n'.format(detected_class)
    if detected_class == y[i]:
        correct_num += 1
    plot_object(X[i], i, y1, 's')
print '-' * 80
print 'Productivity percentage: {0:.3f}%'.format((correct_num / 50.) * 100)

print '-' * 80
#X1 = np.array(X, np.float32)
for i in range(len(X)):
   plot_object(X[i], i, y)
plt.show()
