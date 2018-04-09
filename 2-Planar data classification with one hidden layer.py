import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

np.random.seed(1)

X = [5][400]
Y = [2][400]

plt.scatter(X[0, :], X[1,:], c=Y, s=40, cmap=plt.cm.Spectral)

shape_X = X.shape
shape_Y = Y.shape

m = Y.shape[1]

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

LR_predictions = clf.predict(X.T)

print("Accuracy:" + float((np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions))/float(Y.size) * 100))

# in course dataset accuracy with logisticRegression is 47% because dataset is not linearly separable

def layer_size(X, Y):
    n_x = X.shape[0]  # size of input layer
    n_h = 4  # size of hidden layer
    n_y = Y.shape[0]  # size of output layer

    return n_x, n_h, n_y

n_x, n_h, n_y = layer_size(X, Y)

def initiazlied_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn((n_y, n_h)) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def sigmoid(z):
    s =  1 / (1 + np.exp(-z))
    return s

def forward_propagation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    m = Y.shape[10]

    W1 = parameters['W1']
    W2 = parameters['W2']

    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1 - A2))
    cost = -np.sum(logprobs) / m
    cost = np.squeeze(cost)

    assert (isinstance(cost, float))

    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply((np.dot(W2.t, dZ2), 1 - np.power(A1, 2)))
    dW1 = (1 / m) * np.dot((dZ1, X.T))
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations=1000):
    np.random.seed(3)
    n_x = layer_size(X, Y)[0]
    n_y = layer_size((X, Y))[2]

    parameters = initiazlied_parameters(n_x, n_h, n_y)

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range (0,num_iterations):

        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y, parameters)

        grads = backward_propagation(parameters, cache, X, Y)

        if i%1000 == 0:
            print(i, cost)

    return parameters


def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)

    return predictions

parameters = nn_model(X, Y, n_h=4, num_iterations=10000)

predictions = predictions(parameters, X)

print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

# now accuracy is 90%

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50] # best is 5 because prevent overfitting

for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print(n_h, accuracy)



