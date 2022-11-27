import numpy as np
from layers.activations import Sigmoid, Tanh, ReLU, Softmax
from layers.dense import Dense
from loss import MeanSquaredError, CrossEntropy
from network import NeuralNetwork
import matplotlib.pyplot as plt
from layers.logger import Logger
from layers.reshape import Reshape


if __name__ == "__main__":
    # XOR
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2, 1))
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]).reshape((4, 2, 1))
    print(x.shape, y.shape)
    epochs = 10000
    print_every = 1000
    batch_size = 1
    learning_rate = 0.1

    loss_classes = [MeanSquaredError, CrossEntropy]
    hidden_activation_classes = [Sigmoid, Tanh, ReLU]
    output_activation_classes = [Sigmoid, Softmax]

    for loss_class in loss_classes:
        for hidden_activation_class in hidden_activation_classes:
            for output_activation_class in output_activation_classes:
                print("Loss: ", loss_class.__name__, "Hidden Activation: ", hidden_activation_class.__name__, "Output Activation: ", output_activation_class.__name__)
                layers = [
                    Dense(2, 3),
                    hidden_activation_class(),
                    Dense(3, 2),
                    output_activation_class(),
                ]

                loss = loss_class()
                model = NeuralNetwork(layers, loss)
                model.train(x, y, epochs=epochs, print_every=print_every, batch_size=batch_size, learning_rate=learning_rate, x_test=x, y_test=y)
    
                # Make predictions
                for i in range(len(x)):
                    print("Prediction: ", model.forward(x[i]), "Actual: ", y[i])

                print("------------------------------------------------------")

