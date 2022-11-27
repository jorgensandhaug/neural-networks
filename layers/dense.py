import numpy as np
from layers.layer import Layer
class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size) / input_size
        self.biases = np.random.rand(output_size, 1) / input_size

    def forward(self, x):
        self.input = x
        self.output = np.dot(self.weights, x) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        self.weights -= weights_gradient * learning_rate
        self.biases -= output_gradient * learning_rate

        return input_gradient