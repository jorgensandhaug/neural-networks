from layers.layer import Layer
class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x):
        self.input = x
        self.output = x.reshape(self.output_shape)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)
