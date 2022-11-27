import numpy as np


class NeuralNetwork:
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, output_gradient, learning_rate):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)

    def train(self, x, y, learning_rate=0.01, epochs=100, print_every=10, batch_size=32, x_test=None, y_test=None):
        """
        Train the neural network using stochastic gradient descent
        """
        def display_accuracy():
            if(x_test is not None and y_test is not None):
                predictions = np.zeros(y_test.shape)
                for i in range(len(x_test)):
                    y_pred = self.forward(x_test[i])
                    predictions[i] = y_pred

                accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))

                # Print accuracy in red if it's below 50%, otherwise green
                if accuracy < 0.9:
                    print(f"\033[91mTest Accuracy: {accuracy}\033[0m")
                else:
                    print(f"\033[92mTest Accuracy: {accuracy}\033[0m")



        for epoch in range(epochs):
            loss = 0

            # Select a random subset of the data
            # This is called stochastic gradient descent
            # Shuffle the data
            permutation = np.random.permutation(x.shape[0])
            x_perm = x[permutation]
            y_perm = y[permutation]
        

            # Select a single batch
            x_batch = x_perm[:len(x_perm) // batch_size]
            y_batch = y_perm[:len(y_perm) // batch_size]



            for i in range(len(x_batch)):
                x_sample = x_batch[i]
                y_true = y_batch[i]
                y_pred = self.forward(x_sample)
                y_true = y_true.reshape(y_pred.shape)
                loss_gradient = self.loss.derivative(y_true, y_pred)
                loss += self.loss(y_true, y_pred)
                self.backward(loss_gradient, learning_rate)
            
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch}: Mean loss = {loss/len(x_batch)}")
                display_accuracy()


        print(f"Epoch {epoch}: Mean loss = {loss/len(x_batch)}")
        display_accuracy()
    

            
    def __repr__(self):
        return f"NeuralNetwork({self.layers}, {self.loss}, {self.loss.derivative})"

    def __str__(self):
        return f"NeuralNetwork({self.layers}, {self.loss}, {self.loss.derivative})"
