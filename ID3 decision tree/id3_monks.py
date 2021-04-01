#!/usr/bin/python
import numpy as np
from math import log
import copy

def print_table_out(X, y, attr_names):
    max_length = max([len(X[i]) for i in range(len(X))])
    row_format = '{:<3}' + ('{:>' + str(max_length + 4) + '}') * \
                 (len(X[0])) + '{class_:>10}'
    print row_format.format('No.', *attr_names, class_ = 'Class')
    for i, row_num_vals, cl in zip(range(0, len(X)), X, y):
        print row_format.format(i, *row_num_vals, class_ = cl)

def entropy(S):
    result = 0
    for value in set(S):
        e = [x for x in S if x == value]
        p = float(len(e)) / float(len(S))
        result -= p * log(p) / log(2)
    return result

def detect_attribute(X, y, attr_names):
    if entropy(y) == 0:
        return [y[0]]
    else:
        n = X.shape[1]
        m = X.shape[0]
        IG = []
        for j in range(n):
            attr_values = set(X[:, j])
            IG.append(entropy(y))
            for value in attr_values:
                ys_of_value = [x for (i, x) in enumerate(y)
                               if X[i, j] == value]
                IG[-1] -= (float(len(ys_of_value)) / float(m)) * \
                           entropy(ys_of_value)
        current_attr_no = max([(ig, j) for (j, ig) in enumerate(IG)])[1]
        result = {}
        for value in set(X[:, current_attr_no]):
            X_value = copy.deepcopy(X)
            y_value = copy.deepcopy(y)
            needed_rows = [i for i in range(m) if
                           X[i, current_attr_no] == value]
            X_value = X_value[needed_rows, :]
            y_value = y_value[needed_rows]
            X_value = np.delete(X_value, current_attr_no, 1)
            attr_names_value = copy.deepcopy(attr_names)
            del attr_names_value[current_attr_no]
            result[str(attr_names[current_attr_no]) + ':' +
                   str(value)] = detect_attribute(X_value, y_value,
                                                  attr_names_value)
        return result
        
def predict(x, X, y, attr_names):
    next_node = detect_attribute(X, y, attr_names)
    while True:
        if type(next_node) == list:
            return next_node[0]
        node_attr = next_node.keys()[0].split(':')[0]
        obj_attr_val = x[attr_names.index(node_attr)]
        try:
            next_node = next_node[node_attr + ':' + str(obj_attr_val)]
        except KeyError:
            return '?'

f = open('monks-2.train', 'r')
f_test = open('monks-2.test', 'r')
lines = f.read().splitlines()
lines_test = f_test.read().splitlines()
n = lines[0].count(' ') - 2 # We intend to get rid off the last column
m = len(lines) # Number of training examples
X = []
X_test = []
y = []
y_test = []
attr_names = ['Feature' + str(i + 1) for i in range(n)]
for i in range(m):
    lines[i] = lines[i].strip()
    lines_test[i] = lines_test[i].strip()
    splitted_line = lines[i].split(' ')
    splitted_test_line = lines_test[i].split(' ')
    del splitted_line[-1]
    del splitted_test_line[-1]
    y.append(splitted_line[0])
    X.append(splitted_line[1:])
    y_test.append(splitted_test_line[0])
    X_test.append(splitted_test_line[1:])
X = np.array(X, dtype = object)
y = np.array(y, dtype = object)
print_table_out(X, y, attr_names)
print '-' * 80
print detect_attribute(X, y, attr_names)
print '-' * 80
right_ans = 0
for i, x in enumerate(X_test):
    if predict(x, X, y, attr_names) == y_test[i]:
        right_ans += 1
print 'Productivity percentage: {0:.3f}%'.format((float(right_ans) / \
                                                  float(len(X_test))) * 100)
