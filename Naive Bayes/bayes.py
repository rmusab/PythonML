#!/usr/bin/python
import pickle
import numpy as np
from math import log

DIC_SIZE = 2500

def predict(x, X_train, pwc):
    p_c = {} # Probabilities of classes given feature vector x \
          # in the form: Class - Corresponding Probability
    m = X_train.shape[0]
    for c in set(X_train[:, -1]):
        prob_c = float(sum([1 for i in range(m) if X_train[i, -1] == c])) / \
                 float(m)
        prob = 0
        for x_j, j_word_prob in zip(x, pwc[c]):
            prob += x_j * log(j_word_prob)
        prob += log(prob_c)
        p_c[c] = prob
    return p_c.keys()[p_c.values().index(max(p_c.values()))]

with open('X_train.dat', 'r') as f:
    X_train = pickle.load(f)
with open('X_test.dat', 'r') as f:
    X_test = pickle.load(f)
pwc = {} # Probabilities of words in gives classes
X_train = np.array(X_train, np.int32)
X_test = np.array(X_test, np.int32)
m = X_train.shape[0]
n = X_train.shape[1]
for c in set(X_train[:, -1]):
    needed_rows = [i for i in range(m) if X_train[i, -1] == c]
    X_train_cropped = X_train[needed_rows, :n-1]
    p = [] # Probabilities of words to be in class c
    for j in range(n - 1):
        p_j = float(np.sum(X_train_cropped[:, j]) + 1) / \
              float(np.sum(X_train_cropped) + DIC_SIZE)
        p.append(p_j)
    pwc[c] = p

m_test = X_test.shape[0]
correct_predictions = 0
for i in range(m_test):
    if predict(X_test[i, :-1], X_train, pwc) == X_test[i, -1]:
        correct_predictions += 1
accuracy = float(correct_predictions) / float(m_test) * 100
print '*'*80
print "Attained accuracy: {0:.3f}%".format(accuracy)
