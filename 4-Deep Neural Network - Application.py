import numpy as np
import time
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage

from 3-Building your Deep Neural Network - Step by Step.py import initialize_parameters, linear_activation_forward, compute_cost, linear_activation_backward, update_parameters, initialize_parameters_deep, L_model_backward, L_model_forward
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig = np.zeros()
train_y = np.zeros()
test_x_orig = np.zeros()
test_y = np.zeros()
classes = np.zeros()

index = 10

plt.imshow(train_x_orig[index])

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1 ).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1 ).T

# Standardize
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000):
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters (n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):

        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')

        cost = compute_cost(A2, Y)

        dA2 = - (np.divide(Y, A2) - np.divide(1-Y, 1-A2))


        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if i% 100 == 0 :
            print (cost)
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
# 72% accuracy

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000):
    costs = []
    np.random.seed(1)

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0 , num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            print(cost)
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
# 84% accuracy



