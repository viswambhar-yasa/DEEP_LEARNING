## Implementation of Neural networks and comparing it with logistic regression
import numpy as np
import matplotlib.pyplot as plt


def activation_function(Z,type='sigmoid'):
    '''
    Compute the activation function required from build NN(Neural Networks)

    Inputs:
    Z -- A scalar or numpy array of any size.

    Returns:
    s -- activation_function(z)
    '''
    if type=='sigmoid':
        AF=1/(1+np.exp(-Z))
    return AF
