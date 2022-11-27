import numpy as np
from layers.layer import Layer

class ActivationLayer(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, x):
        self.input = x
        self.output = self.activation(x)
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient * self.activation_derivative(self.input)
        return input_gradient


class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_derivative)

    def sigmoid(self, x):
        sig = 1 / (1 + np.exp(-x, dtype=np.float64))
        # sig = np.minimum(sig, 0.9999)  # Set upper bound
        # sig = np.maximum(sig, 0.0001)  # Set lower bound
        return sig

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

class Tanh(ActivationLayer):
    def __init__(self):
        super().__init__(self.tanh, self.tanh_derivative)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

class ReLU(ActivationLayer):
    def __init__(self):
        super().__init__(self.relu, self.relu_derivative)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

class Softmax(Layer):
    def forward(self, x):
        self.input = x
        # Subtract the max value to avoid overflow
        x = x - np.max(x)
        exp = np.exp(x)
        self.output = exp / np.sum(exp, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)