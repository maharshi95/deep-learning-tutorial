from typing import List, Dict
import numpy as np

def initialize_parameters(layer_dims: List[int]) -> Dict[str, List[np.ndarray]]:
    parameters = {
        'W': [],
        'b': [],
    }
    L = len(layer_dims)  # number of layers in the network
    for l in range(L):
        W = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        b = np.zeros((layer_dims[l], 1))
        parameters['W'].append(W)
        parameters['b'].append(b)
    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def activation_forward(g, Z):
    return g(Z), Z


def linear_activation_forward(A_prev, W, b, g):
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = activation_forward(g, Z)
    cache = (linear_cache, activation_cache)

    return A, cache


def forward_prop(X, parameters, activations):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        A_prev = A
        W, b, g = parameters['W'][l], parameters['b'][l], activations[l]
        A, cache = linear_activation_forward(A_prev, W, b, g)
        caches.append(cache)
    return A, caches


def compute_cost(AL: np.ndarray, Y: np.ndarray):
    m = Y.shape[1]

    cost = - np.sum(Y * np.log(AL) + (1 - Y) * np.log((1 - AL))) / m
    cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def activation_backward(dA, cache, g_grad):
    return np.dot(dA, g_grad(cache))


def linear_activation_backward(dA, cache, g_grad):
    linear_cache, activation_cache = cache
    dZ = activation_backward(dA, activation_cache, g_grad)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def backward_prop(AL, Y, caches, g_grads):
    grads = {
        'dA': {},
        'dW': {},
        'db': {}
    }

    L = len(caches)  # the number of layers
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    dA = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # derivative of cost with respect to AL
    grads['dA'][L] = dA

    for l in reversed(range(L)):
        g_grad = g_grads[l]
        current_cache = caches[l]
        dA_prev = dA
        dA, dW, db = linear_activation_backward(dA_prev, current_cache, g_grad)
        grads["dA"][l], grads["dW"][l], grads["db"][l] = dA, dW, db

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network
    W, b = parameters['W'], parameters['b']
    dW, db = grads['dW'], grads['db']
    for i in range(L):
        l = i + 1
        W[l] -= learning_rate * dW[l]
        b[l] -= learning_rate * db[l]

    return parameters
