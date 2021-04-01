#!/usr/bin/python
import numpy as np
from math import exp
import copy
import matplotlib
import matplotlib.pyplot as plt

def euclid(x, y):
    result = 0
    for x_i, y_i in zip(x, y):
        result += (x_i - y_i) ** 2
    return result ** (1 / 2.)

def hypothesis(x, theta):
    z = np.dot(theta.transpose(), x.transpose())
    return 1. / (1. + exp(-z))

f = open('ex2data1.txt', 'r')
data_lines = f.read().splitlines()
m = len(data_lines)
n = data_lines[0].count(',')
X = [] # Design matrix
y = []
for line in data_lines:
    line_values = line.split(',')
    X.append(line_values[:n])
    y.append(line_values[n])
X = np.array(X, dtype = np.float32)
X = np.insert(X, 0, np.ones(m), axis = 1)
y = np.array(y, dtype = np.float32)

epsilon = 0.001
alpha = 0.001
theta0 = np.array([epsilon for i in range(n + 1)], np.float32).reshape((n + 1, 1))
theta1 = np.zeros((n + 1, 1))
p = 0
while p <= 40000:
    theta0 = copy.deepcopy(theta1)
    delta = np.zeros((n + 1, 1))
    for i in range(m):
        hyp_xi = hypothesis(X[i, :], theta0)
        delta += ( (hyp_xi - y[i]) * X[i, :].transpose() ).reshape(n + 1, 1)
    theta1 = theta0 - alpha / float(m) * delta
    p += 1
    # print euclid(theta0, theta1)

class_1_indices = [i for i in range(m) if y[i] == 1]
class_0_indices = [i for i in range(m) if y[i] == 0]
plt.figure(1)
plt.plot(X[class_1_indices, 1], X[class_1_indices, 2], 'k+')
plt.plot(X[class_0_indices, 1], X[class_0_indices, 2], 'yo')

"""
theta(0) + theta(1) * x1 + theta(2) * x2 = 0
x2 = - [theta(1) / theta(2)] * x1 - theta(0) / theta(2)
"""

print theta1
T = np.linspace(min(X[:, 1]), max(X[:, 1]), 200)
Y =  - theta1[1]/ theta1[2] * T - theta1[0] / theta1[2]
plt.plot(T, Y, '-c')
plt.show()
