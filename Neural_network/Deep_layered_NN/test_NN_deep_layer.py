import numpy as np
import pytest 
from NN_deep_layer import forward_propagation, backward_propagation, cost_function

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
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    A3 = np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])
    lambd = 0.1
    expected_output = np.float64(1.7864859451590758)
    Y = np.asarray([[1, 1, 0]])
    A = np.array([[.8, .9, 0.4]])
    cost=cost_function(A,Y)
    ex_cost = cost
    test = cost-ex_cost<1e-5
    assert test==True

def test__back_propagation_test():
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4, 2)
    w1 = np.random.randn(3, 4)
    b1 = np.random.randn(3, 1)
    Z1 = np.random.randn(3, 2)
    linear_cache_activation_1 = ((A1, w1, b1), Z1)

    A2 = np.random.randn(3, 2)
    w2 = np.random.randn(1, 3)
    b2 = np.random.randn(1, 1)
    Z2 = np.random.randn(1, 2)
    linear_cache_activation_2 = ((A2, w2, b2), Z2)

    cache = (linear_cache_activation_1, linear_cache_activation_2)
    grad=backward_propagation(AL, Y, cache)
    expected_grad = {'dA1': np.array([[0.12913162, -0.44014127],
                                      [-0.14175655,  0.48317296],
                                      [0.01663708, -0.05670698]]), 'dw2': np.array([[-0.39202432, -0.13325855, -0.04601089]]), 'db2': np.array([[0.15187861]]), 'dA0': np.array([[0.,  0.52257901],
                                                                                                                                                                                 [0., -0.3269206],
                                                                                                                                                                                 [0., -0.32070404],
                                                                                                                                                                                 [0., -0.74079187]]), 'dw1': np.array([[0.41010002, 0.07807203, 0.13798444, 0.10502167],
                                                                                                                                                                                                                       [0., 0.,
                                                                                                                                                                                                                        0., 0.],
                                                                                                                                                                                                                       [0.05283652, 0.01005865, 0.01777766, 0.0135308]]), 'db1': np.array([[-0.22007063],
                                                                                                                                                                                                                                                                                           [0.],
                                                                                                                                                                                                                                                                                           [-0.02835349]])}

    assert np.all(grad['dA1']-expected_grad['dA1']<1e-5) == True
    assert np.all(grad['dw2']-expected_grad['dw2'] < 1e-5) == True
    assert np.all(grad['db2']-expected_grad['db2'] < 1e-5) == True
    assert np.all(grad['dA0']-expected_grad['dA0'] < 1e-5) == True
    assert np.all(grad['dw1']-expected_grad['dw1'] < 1e-5) == True
    assert np.all(grad['db1']-expected_grad['db1'] < 1e-5) == True

    
