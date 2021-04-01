#!/usr/bin/python
import os
import glob
import pickle
import numpy as np

def count_words(counted_so_far):
    file_names = glob.glob('*.txt')
    for file_name in file_names:
        f = open(file_name, 'r')
        words = f.read().strip().split(' ')
        for word in words:
            if not word in counted_so_far.keys():
                counted_so_far[word] = 0
            counted_so_far[word] += 1
        f.close()
    return counted_so_far

def extract_features(X_so_far, d, spam):
    file_names = glob.glob('*.txt')
    for file_name in file_names:
        with open(file_name, 'r') as f:
            words = f.read().strip().split(' ')
        feature_vector = [0 for i in range(len(d))]
        for j, dict_word in enumerate(d):
            feature_vector[j] = words.count(dict_word)
        if spam:
            feature_vector.append(1)
        else:
            feature_vector.append(0)
        X_so_far.append(feature_vector)
    return X_so_far

os.chdir('./ex6DataEmails')
d = {} # Dictionary
os.chdir('./nonspam-train')
d = count_words(d)
os.chdir('../spam-train')
d = count_words(d)
os.chdir('../nonspam-test')
d = count_words(d)
os.chdir('../spam-test')
d = count_words(d)
os.chdir('../..')

d_list = [] # resulting list of dictionary containing 2500 \
       # most frequent words
for i in range(2500):
    max_freq = max(d.values())
    word = d.keys()[d.values().index(max_freq)]
    d_list.append((word, max_freq))
    del d[word]
print d_list
d_list = [tup[0] for tup in d_list]
with open('dictionary.dat', 'wb') as f:
    pickle.dump(d_list, f)

# Main data matrices of features
X_train = []
X_test = []
os.chdir('./ex6DataEmails')
os.chdir('./nonspam-train')
X_train = extract_features(X_train, d_list, False)
os.chdir('../spam-train')
X_train = extract_features(X_train, d_list, True)
os.chdir('../nonspam-test')
X_test = extract_features(X_test, d_list, False)
os.chdir('../spam-test')
X_test = extract_features(X_test, d_list, True)
os.chdir('../..')
# print X_train
# print X_test
with open('X_train.dat', 'wb') as f:
    pickle.dump(X_train, f)
with open('X_test.dat', 'wb') as f:
    pickle.dump(X_test, f)
