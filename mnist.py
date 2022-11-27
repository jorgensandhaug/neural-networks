import numpy as np
from layers.layer import Layer
from layers.activations import Sigmoid, Tanh, ReLU, Softmax
from layers.dense import Dense
from layers.convolutional_layer import Convolutional
from layers.reshape import Reshape
from loss import MeanSquaredError, CrossEntropy
from network import NeuralNetwork
import matplotlib.pyplot as plt
from layers.logger import Logger

# Test it on the mnist dataset
if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist

    # Load the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Grab only the first 5000 samples
    x_train = x_train[:10000]
    y_train = y_train[:10000]





    # Normalize the dataset
    x_train = x_train / 255
    x_test = x_test / 255

    # Print the max and min values
    print(np.max(x_train), np.min(x_train))




    # Print shape of the dataset
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


    # One-hot encode the labels to shape (y_train.shape[0], 10, 1)
    y_train = np.eye(10)[y_train].reshape((y_train.shape[0], 10, 1))
    y_test = np.eye(10)[y_test].reshape((y_test.shape[0], 10, 1))



    print(x_train.shape, y_train.shape)




    kernel_size = 3
    n = 28-kernel_size+1
    layers = [
        Reshape((28, 28), (1, 28, 28)),
        Convolutional((1, 28, 28), kernel_size, 4),
        ReLU(),
        Reshape((4, n, n), (4*n*n, 1)),
        Dense(4*n*n, 128),
        ReLU(),
        Dense(128, 10),
        Softmax(),
    ]
    loss = CrossEntropy()
    model = NeuralNetwork(layers, loss)

    # Train the model
    model.train(x_train, y_train, learning_rate=0.01, epochs=100, batch_size=32, x_test=x_test, y_test=y_test)

    # Make predictions
    predictions = []
    for i in range(len(x_test)):
        pred = model.forward(x_test[i].reshape((1,) + x_test[i].shape))
        predictions.append(np.argmax(pred))


    # Display the prediction using matplotlib
    fig = plt.figure()
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(x_test[i].reshape((28, 28)))
        plt.title(f"Prediction: {predictions[i]}")
    

    plt.show()