import numpy as np
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x):
        pass

    def backward(self, output_gradient, learning_rate):
        pass