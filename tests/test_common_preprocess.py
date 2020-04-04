from iris.common import preprocess
import numpy as np


def test_balanceSample():
    X = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
    y = np.array([1, 1, 1, 2])
    newX, newy = preprocess.balanceSample(X, y)
    print(newX, newy)
    assert list(newy) == list(np.array([1, 1, 2, 2]))
