import numpy as np
import pytest 
from NN_L2_regularization import forward_propagation, backward_propagation, cost_function

def test__forward_propogation_2_hidden_layers():
    np.random.seed(1)
    X = np.random.randn(4, 2)
    w1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    w2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}
    AL, caches = forward_propagation(X, parameters)
    expected_AL = np.array([[0.17007265, 0.2524272]])
    test = np.all(AL-expected_AL < 1e-4)
    assert  test == True


def test__forward_propogation_3_hidden_layers():
    np.random.seed(6)
    X = np.random.randn(5, 4)
    w1 = np.random.randn(4, 5)
    b1 = np.random.randn(4, 1)
    w2 = np.random.randn(3, 4)
    b2 = np.random.randn(3, 1)
    w3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2,
                  "w3": w3,
                  "b3": b3}
    AL, caches = forward_propagation(X, parameters)
    #print(caches)
    expected_AL = np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])
    test = np.all(AL-expected_AL < 1e-5)
    assert test == True

def test__cost_function():
    np.random.seed(1)
    Y = np.array([[1, 1, 0, 1, 0]])
    w1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    w2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    w3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}
    A = np.array(
        [[0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])
    lambd = 0.1
    expected_output = float(1.7864859451590758)
    cost = cost_function(A, Y, parameters)
    test = cost-expected_output < 1e-5
    assert test==True
