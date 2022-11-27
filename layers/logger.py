import numpy as np
from layers.layer import Layer
class Logger(Layer):
    def __init__(self, name):
        self.name = name
    def forward(self, x):
        self.input = x
        self.output = x
        print("Logger "+self.name, x.shape, np.max(x), np.min(x))

        return self.output
    def backward(self, output_gradient, learning_rate):
        print("Logger "+self.name + " BACKWARD", output_gradient.shape, output_gradient)
        return output_gradient