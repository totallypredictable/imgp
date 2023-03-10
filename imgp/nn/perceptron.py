import numpy as np


class Perceptron:
    def __init__(
        self, N: int, alpha: float = 0.1, random_state: int = 0, addBias: bool = True
    ):
        # initialise the weight matrix and store the learning rate
        np.random.seed(random_state)
        self.addBias = addBias
        if self.addBias:
            self.W = np.random.randn(N + 1) / np.sqrt(N)
        else:
            self.W = np.random.randn(N) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x: int | float) -> int:
        # apply the step function
        return 1 if x > 0 else 0

    def fit(self, X: np.ndarray, y: np.ndarray | list, epochs: int = 10) -> None:
        # insert a column of 1's as the last entry in the feature matrix
        # this little trick allows us to treat the bias as a trainable
        # parameter within the weight matrix
        if self.addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                # take the dot product between the input features
                # and the weight matrix, then pass this value through
                # the step function to obtain the prediction
                p = self.step(np.dot(x, self.W))

                # only perform a weight update if our prediction
                # does not match the target
                if p != target:
                    # determine the error
                    error = p - target

                    # update the weight matrix
                    self.W += -self.alpha * error * x

    def predict(self, X: np.ndarray):
        # ensure our input is a matrix
        X = np.atleast_2d(X)

        # check to see if the bias column should be added
        if self.addBias:
            # insert a column of 1's as the last entry in the feature
            # matrix (bias)
            X = np.c_[X, np.ones((X.shape[0]))]
        # take the dot product between the input features and the
        # weight matrix, then pass the value through the step function
        return self.step(np.dot(X, self.W))
