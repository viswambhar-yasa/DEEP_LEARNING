import pytest
from optimization import gradient_descent_momentum,adam_optimizer
import numpy as np
def test_gradient_descent_momentum():
    np.random.seed(1)
    w1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    w2 = np.random.randn(3, 3)
    b2 = np.random.randn(3, 1)

    dw1 = np.random.randn(2, 3)
    db1 = np.random.randn(2, 1)
    dw2 = np.random.randn(3, 3)
    db2 = np.random.randn(3, 1)
    parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    grads = {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2}
    v = {'w1': np.array([[0.,  0.,  0.],
                          [0.,  0.,  0.]]), 'w2': np.array([[0.,  0.,  0.],
                                                             [0.,  0.,  0.],
                                                             [0.,  0.,  0.]]), 'b1': np.array([[0.],
                                                                                                [0.]]), 'b2': np.array([[0.],
                                                                                                                         [0.],
                                                                                                                         [0.]])}
    parameters, v = gradient_descent_momentum(parameters, grads, v, learning_rate=0.01, beta=0.9)
    expected_parameters = {'w1': np.array([[1.62544598, -0.61290114, -0.52907334],
                                           [-1.07347112,  0.86450677, -2.30085497]]),
                           'b1': np.array([[1.74493465],
                                           [-0.76027113]]),
                           'w2': np.array([[0.31930698, -0.24990073,  1.4627996],
                                           [-2.05974396, -0.32173003, -0.38320915],
                                           [1.13444069, -1.0998786, -0.1713109]]),
                           'b2': np.array([[-0.87809283],
                                           [0.04055394],
                                           [0.58207317]])}
    expected_v = {'w1': np.array([[-0.11006192,  0.11447237,  0.09015907],
                                   [0.05024943,  0.09008559, -0.06837279]]),
                  'w2': np.array([[-0.02678881,  0.05303555, -0.06916608],
                                   [-0.03967535, -0.06871727, -0.08452056],
                                   [-0.06712461, -0.00126646, -0.11173103]]),
                  'b1': np.array([[-0.01228902],
                                   [-0.09357694]]),
                  'b2': np.array([[0.02344157],
                                   [0.16598022],
                                   [0.07420442]])}
    assert np.all(parameters['w1']-expected_parameters['w1'] < 1e-5) == True
    assert np.all(expected_v['w1']-v['w1'] < 1e-5) == True