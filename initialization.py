import numpy as np
import sklearn
import sklearn.datasets


# The performance is really bad, and the cost does not really decrease,
# and the algorithm performs no better than random guessing.

# general, initializing all the weights to zero results in the network failing
# to break symmetry. This means that every neuron in each layer will
# learn the same thing, and you might as well be training a neural network
# with $n^{[l]}=1$ for every layer, and the network is no more powerful than
# a linear classifier such as logistic regression.

def initialize_parameters_zeros(layer_dims):

    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


# To break symmetry, lets intialize the weights randomly.
# Following random initialization, each neuron can then
# proceed to learn a different function of its inputs

def initialize_parameters_random(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn((layer_dims[l], layer_dims[l - 1])) * 10
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

# Finally, try "He Initialization"; this is named for the first author of He
# et al., 2015. (If you have heard of "Xavier initialization", this is similar
# except Xavier initialization uses a scaling factor for the weights $W^{[l]}$
# of sqrt(1./layers_dims[l-1]) where He initialization would use sqrt(2./layers_dims[l-1]).)

def initialize_parameters_he(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) - 1

    for l in range(1 , L + 1 ):
        parameters['W' + str(l)] = np.random.randn((layer_dims[l], layer_dims[l - 1])) * np.sqrt(2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

# The model with He initialization separates the blue and the red
# dots very well in a small number of iterations.
# 96%

