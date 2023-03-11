from imgp.nn import Perceptron
import numpy.testing as npt
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y = np.array([[0], [0], [0], [1]])


def test_step(X=X):
    neuron = Perceptron(X.shape[1])
    assert neuron.step(-29384792834) == 0
    assert neuron.step(81239491293) == 1
    assert neuron.step(0) == 0


def test_fit(X=X, y=y):
    p = Perceptron(X.shape[1])
    p.fit(X, y, epochs=5)
    desired = np.array([0.74737338, -0.01704612, -0.40792773])
    actual = p.W
    npt.assert_almost_equal(
        actual=actual,
        desired=desired,
        decimal=7,
        err_msg="Desired and actual weights are not close enough",
    )