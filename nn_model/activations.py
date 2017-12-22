from enum import Enum
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def relu(z):
    return np.max(z, 0)


def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh_grad(z):
    return 1 - tanh(z) ** 2


def relu_grad(z):
    return (z >= 0) * 1


_activation_funcs = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
}

_activation_grad_funcs = {
    'sigmoid': sigmoid_grad,
    'tanh': tanh_grad,
    'relu': relu_grad,
}

def get_activation_func(activation_type):
    return _activation_funcs[activation_type]

def get_activation_grad_func(activation_type):
    return _activation_grad_funcs[activation_type]