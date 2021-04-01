#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

def split_line(line):
    result = []
    i = 0
    while not i == len(line):
        if line[i] == ' ':
            i += 1
            continue
        else:
            next_num = ''
            j = i
            while not line[j] == ' ':
                next_num += line[j]
                j += 1
                if j == len(line):
                    break
            result.append(float(next_num))
            i = j
    return result

def import_file(file_name):
    f = open(file_name, 'r')
    data_lines = f.read().splitlines()
    result = []
    for line in data_lines:
        line_values = split_line(line)
        result.append(line_values)
    result = np.array(result, dtype = np.float32)
    return result

def euclid(x, y):
    result = 0
    for x_i, y_i in zip(x, y):
        result += (x_i - y_i) ** 2
    return result ** (1 / 2.)

def average(x):
    sum1 = 0
    sum2 = 0
    for i in range(len(x)):
        sum1 += x[i][0]
        sum2 += x[i][1]
    sum1 /= float(len(x))
    sum2 /= float(len(x))
    return [sum1, sum2]

X = import_file('ex7data2.txt')
X = np.array(X, dtype = np.float32)
max_iter = 10
k = 3
centroids = np.array([[3, 3], [6, 2], [8, 5]], dtype = np.float32)
for iteration in range(max_iter):
    c = []
    # Cluster assignment step
    for i in range(X.shape[0]):
        dist = [euclid(X[i, :], centroids[j, :]) for j in range(k)]
        c.append(dist.index(min(dist)))
    # Move centroids step
    for m in range(k):
        centroids[m, :] = average([X[i, :] for i in range(X.shape[0]) if c[i] == m])
    plt.figure(1)
    plt.plot(X[:, 0], X[:, 1], 'k+')
    plt.plot(centroids[:, 0], centroids[:, 1], 'yo')
    plt.show()
