import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

from deep_learning.nn_model.activations import get_activation_func, get_activation_grad_func
from deep_learning.nn_model.app_utils import *
from deep_learning.nn_model.helpers import forward_prop, compute_cost, backward_prop

def L_layer_model(X, Y, layers_dims, activations, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    activation_funcs = [get_activation_func(g_type) for g_type in activations]
    activation_grad_funcs = [get_activation_grad_func(g_type) for g_type in activations]

    for i in range(0, num_iterations):

        AL, caches = forward_prop(X, parameters, activation_funcs)
        cost = compute_cost(AL, Y)
        grads = backward_prop(AL, Y, caches, activation_grad_funcs)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

def predict(X, y, parameters, activations):
    m = X.shape[1]
    L = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    probas, caches = forward_prop(X, parameters, activations)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

