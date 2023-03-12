import numpy as np


class NeuralNetwork:
    """
    Implement a neural network from scratch.

    Parameters
    ----------
    layers : list
        A list of integers which represents the actual architecture of the
        feedforward network. For example, a value of [2, 2, 1] would imply that
        our first input layer has two nodes, our hidden layer has two nodes,
        and our final output layer has one node.
    alpha : float
        The learning rate of the neural network. This value is applied during
        the weight update phase.
    """

    def __init__(self, layers: list[int], alpha: float = 0.1, addBias: bool = True):
        # initialise the list of weights as matrices, then store the network
        # architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha
        self.addBias = addBias

        # start looping from the index of the first layer but stop before we
        # reach the last two layers
        for i in np.arange(0, len(layers) - 2):
            # randomly initialise a weight matrix connecting the number of
            # nodes in each respective layer together, adding an extra node
            # for the bias if addBias=True
            if self.addBias:
                w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            else:
                w = np.random.randn(layers[i], layers[i + 1])
            self.W.append(w / np.sqrt(layers[i]))

        # the last two layers are a special case where the input connections
        # need a bias term but the output does not
        if self.addBias:
            w = np.random.randn(layers[-2] + 1, layers[-1])
        else:
            w = np.random.randn(layers[-2], layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self) -> str:
        # construct and return a string that represents the network
        # architecture
        return "NeuralNetwork: {}".format("-".join(str(1) for layer in self.layers))

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # compute and return the sigmoid activation value for a given input
        # value
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x: np.ndarray) -> np.ndarray:
        # compute the derivative of the sigmoid function assuming that "x"
        # has already been passed through the "sigmoid" function.
        return x * (1 - x)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | list,
        epochs: int = 1000,
        displayUpdate: int = 100,
    ) -> None:
        # insert a column of 1's as the last entry in the feature matrix if addBias=True
        if self.addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point and train the network on it
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x: np.ndarray | list, y: int):
        # construct our list of output activations for each layer as our data
        # point flows through the network; the first activation is a special
        # caes -- it's just the input feature vecor itself
        A = [np.atleast_2d(x)]

        # FEEDFORWARD
        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer by taking
            # the dot product between the activation and the weight
            # matrix -- this is called the "net input" to the current layer
            net = A[layer].dot(self.W[layer])

            # computing the "net output" is simply applying our nonlinear
            # activation function to the net input
            out = self.sigmoid(net)

            # once we have the net output, add it to our list of activations
            A.append(out)

            # BACKPROPAGATION
            # the first phase of backpropagation is to compute the difference
            # between our prediction and the true target value
            error = A[-1] - y

            # from here, we need to apply the chain rule and build our list
            # of deltas 'D'; the first entry in the deltas is simply the error
            # of the output layer times the derivative of our activation
            # function for the output value.
            D = [error * self.sigmoid_deriv(A[-1])]

            for layer in np.arange(len(A) - 2, 0, -1):
                # the delta for the current layer is equal to the delta of
                # the previous layer dotted with the weight matrix of the
                # current layer, followed by multiplying the delta by the
                # derivative of the nonlinear activation function for the
                # activations of the current layer
                delta = D[-1].dot(self.W[layer].T)
                delta = delta * self.sigmoid_deriv(A[layer])
                D.append(delta)

            # since we looped over our layers in reverse order, we need to
            # reverse the deltas
            D = D[::-1]

            # WEIGHT UPDATE
            # loop over the layers
            for layer in np.arange(0, len(self.W)):
                # update our weights by taking the dot product of the layer
                # activations with their respective deltas, then multiplying
                # this value by some small learning arate and adding to our
                # weight matrix -- this is where the actual learning take
                # place
                self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X: np.ndarray, addBias: bool | None = None) -> np.ndarray:
        # initialise the output prediction as the input features -- this value
        # will be (forward) propagated through the network to obtain the final
        # prediction
        if addBias is None:
            addBias = self.addBias
        p = np.atleast_2d(X)
        if addBias:
            # insert a column of 1's
            p = np.c_[p, np.ones((p.shape[0]))]

        # loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
            # computing the output prediction by taking the dot product
            # between the current activation value "p" and the weight matrix
            # associated with the current layer, then passing this value
            # through a nonlinear activation function
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X: np.ndarray, targets: np.ndarray | list) -> float:
        # make the predictions for the input data points then compute the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss
