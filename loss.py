import numpy as np

class Loss:
    def __init__(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative

    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)

    def derivative(self, y_true, y_pred):
        return self.loss_derivative(y_true, y_pred)

class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__(self.mse, self.mse_derivative)

    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def mse_derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


class CrossEntropy(Loss):
    def __init__(self):
        super().__init__(self.cross_entropy, self.cross_entropy_derivative)

    def cross_entropy(self, y_true, y_pred):
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


    def cross_entropy_derivative(self, y_true, y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

    

