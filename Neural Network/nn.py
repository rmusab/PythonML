#!/usr/bin/python
import numpy as np
import scipy.optimize as opt

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

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1. - sigmoid(z))

def feedforward(p, input_layer_size, hidden_layer_size, num_labels, X, y, lamda):
    Theta1 = np.reshape(p[0 : hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(p[hidden_layer_size * (input_layer_size + 1) : ], (num_labels, hidden_layer_size + 1))
    m = X.shape[0]

    # Feedforward step
    X = np.insert(X, 0, np.ones(m), axis = 1) # 5000 * 401
    z2 = np.dot(Theta1, X.T).T # 500 * 25
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, np.ones(m), axis = 1) # 5000 * 26
    z3 = np.dot(Theta2, a2.T).T  # 5000 * 10
    a3 = sigmoid(z3)

    return (Theta1, Theta2, z2, a2, z3, a3)

def nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lamda):
    (Theta1, Theta2, z2, a2, z3, a3) = feedforward(p, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)
    m = X.shape[0]

    # Computing cost function
    Y = np.zeros((m, num_labels), dtype = np.float32) # 5000 * 10
    for i in range(m):
        Y[i, y[i] - 1] = 1.
    J = -1./float(m) * np.sum(Y * np.log(a3) + (1 - Y) * np.log(1 - a3))

    # Regularizing cost function:
    J += float(lamda) / float(2 * m) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))

    return J

def nnGradFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lamda):
    # Well-vectorized backprop algorithm
    (Theta1, Theta2, z2, a2, z3, a3) = feedforward(p, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)
    m = X.shape[0]
    X = np.insert(X, 0, np.ones(m), axis = 1)

    Y = np.zeros((m, num_labels), dtype = np.float32) # 5000 * 10
    for i in range(m):
        Y[i, y[i] - 1] = 1.
    z2 = np.insert(z2, 0, np.ones(m), axis = 1) # 5000 * 26
    delta3 = a3 - Y # 5000 * 10
    delta2 = np.dot(Theta2.T, delta3.T).T * sigmoid_gradient(z2)
    delta2 = delta2[:, 1:] # 5000 * 25
    DELTA1 = np.dot(delta2.T, X)
    DELTA2 = np.dot(delta3.T, a2)

    Theta1_grad = 1. / float(m) * DELTA1
    Theta2_grad = 1. / float(m) * DELTA2

    # Adding regularization to gradient
    for i in range(Theta1_grad.shape[0]):
        for j in range(1, Theta1_grad.shape[1]):
            Theta1_grad[i, j] += float(lamda) / float(m) * Theta1[i, j]
    for i in range(Theta2_grad.shape[0]):
        for j in range(1, Theta2_grad.shape[1]):
            Theta2_grad[i, j] += float(lamda) / float(m) * Theta2[i, j]

    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))
    return grad

def rand_weights(L_in, L_out):
    epsilon_init = 0.12
    return np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

def predict(x, Theta1, Theta2):
    x = np.insert(x, 0, np.ones(x.shape[0]), axis = 1)
    z2 = np.dot(Theta1, x.T).T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, np.ones(x.shape[0]), axis = 1)
    z3 = np.dot(Theta2, a2.T).T
    a3 = sigmoid(z3)
    return a3
"""
def shuffle(X, y):
    for i in range(X.shape[0]):
        rand_index = random.randint(0, X.shape[0] - 1)
        temp = X[i, :]
        X[i, :] = X[rand_index, :]
        X[rand_index, :] = temp
        temp = y[i]
        y[i] = y[rand_index]
        y[rand_index] = temp
"""
m = 5000
n = 400
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
X = import_file('dataX.txt')
Theta1 = import_file('Theta1.txt')
Theta2 = import_file('Theta2.txt')
y = import_file('dataY.txt')
y = np.array(y, dtype = np.int32).flatten()
"""
#shuffle(X, y)
train_n = int(m * 7 / 10)
X_train = X[:train_n, :]
y_train = y[:train_n]
X_test = X[train_n:, :]
y_test = y[train_n:]
"""
lamda = 0 # Without regularization
nn_params = np.concatenate((Theta1.flatten(), Theta2.flatten()))
J0 = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)
print 'Cost at initial Theta without regularization: ', J0

lamda = 1 # With regularization
nn_params = np.concatenate((Theta1.flatten(), Theta2.flatten()))
J1 = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)
print 'Cost at initial Theta with regularization: ', J1

initial_Theta1 = rand_weights(input_layer_size, hidden_layer_size)
initial_Theta2 = rand_weights(hidden_layer_size, num_labels)
initial_nn_params =  np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
lamda = 1
res = opt.fmin_cg(nnCostFunction, initial_nn_params, fprime = nnGradFunction, args = (input_layer_size, hidden_layer_size, num_labels, X, y, lamda), maxiter = 400, full_output = True, disp = True)
opt_theta = res[0]

Theta1 = np.reshape(opt_theta[0 : hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
Theta2 = np.reshape(opt_theta[hidden_layer_size * (input_layer_size + 1) : ], (num_labels, hidden_layer_size + 1))
predictions = predict(X, Theta1, Theta2)
successes = 0
for right_value, prediction in zip(y, predictions):
    if right_value == prediction.argmax(axis=0) + 1:
        successes += 1

print "Accuracy: %0.2f" % (float(successes) / float(y.size) * 100)
print 'Optimal cost: ', res[1]
