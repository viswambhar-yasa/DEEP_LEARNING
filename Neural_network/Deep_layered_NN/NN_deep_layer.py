
import numpy as np

def activation_function(Z, typ):
    if typ == 'sig':
        A = 1/(1+np.exp(-Z))
        assert A.shape == Z.shape
        return A
    elif typ == 'tanh':
        A = np.tanh(Z)
        assert A.shape == Z.shape
        return A
    elif typ == 'relu':
        A = np.maximum(0, Z)
        assert A.shape == Z.shape
        return A


def der_activation_fun(dA, cache, typ):
    if typ == 'sig':
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        assert (dZ.shape == Z.shape)
        return dZ
    elif typ == 'relu':
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
        return dZ
    elif typ == 'tanh':
        Z = cache
        s = 1-np.tanh(Z)**2
        dZ = dA*s
        assert (dZ.shape == Z.shape)
        return dZ


def hypothesis(X, w, b):
    h = np.dot(w, X)+b
    return h


def cost_function(A, Y):
    m = Y.size
    logpro = np.multiply(Y, np.log(A))+np.multiply((1-Y), np.log(1-A))
    cost = -np.sum(logpro)/m
    return float(cost.ravel())


def forward_propagation(X, parameters, typ='relu'):
    fw_cache = []
    n_layer = int(len(parameters)/2)
    #print(n_layer)
    A_pre = X
    #hidden layers
    for i in range(1, n_layer):
        w = parameters['w'+str(i)]
        b = parameters['b'+str(i)]
        Z = hypothesis(A_pre, w, b)
        ln_cache = (A_pre, w, b)
        A = activation_function(Z, typ='relu')
        cache = (ln_cache, Z)
        fw_cache.append(cache)
        A_pre = A
    #ouput layer
    w = parameters['w'+str(n_layer)]
    b = parameters['b'+str(n_layer)]
    Z = hypothesis(A_pre, w, b)
    ln_cache = (A_pre, w, b)
    AL = activation_function(Z, typ='sig')
    #print('AL', AL)
    cache = (ln_cache, Z)
    fw_cache.append(cache)
    return AL, fw_cache


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def backward_propagation(AL, Y, cache, typ='relu'):
    grad = {}
    n_layer = len(cache)
    #print(n_layer)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    #back pro for last layer
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    #print('dAL', dAL)
    current_cache = cache[-1]
    linear_cache, activation_cache = current_cache
    #derivative of activation function
    dZ = der_activation_fun(dAL, activation_cache, typ='sig')
    #derivatives of parameters in that hidden layyer
    dA_prev_temp, dW_temp, db_temp = linear_backward(dZ, linear_cache)
    grad["dA" + str(n_layer-1)] = dA_prev_temp
    grad["dw" + str(n_layer)] = dW_temp
    grad["db" + str(n_layer)] = db_temp
    for i in reversed(range(n_layer-1)):
        #print(i)
        current_cache = cache[i]
        #print(current_cache)
        linear_cache, activation_cache = current_cache
        #derivative of activation function
        dA = grad["dA" + str(i + 1)]
        dZ = der_activation_fun(dA, activation_cache, typ='relu')
        #derivatives of parameters in that hidden layyer
        dA_prev_temp, dW_temp, db_temp = linear_backward(dZ, linear_cache)
        grad["dA" + str(i)] = dA_prev_temp
        grad["dw" + str(i+1)] = dW_temp
        grad["db" + str(i+1)] = db_temp
    return grad


def initialization_parameter(layer_dims, typ):
    L = len(layer_dims)
    parameters = {}
    if typ == 'ranm':
        for l in range(1, L):
            parameters['w' + str(l)] = np.random.randn(layer_dims[l],
                                                       layer_dims[l-1])*10
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            assert(parameters['w' + str(l)].shape ==
                   (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        return parameters
    elif typ == 'zero':
        for l in range(1, L):
            parameters['w' +
                       str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            assert(parameters['w' + str(l)].shape ==
                   (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        return parameters
    elif typ == 'He':
        for l in range(1, L):
            parameters['w'+str(l)] = np.random.randn(layer_dims[l],
                                                     layer_dims[l-1])*(np.sqrt(2/layer_dims[l-1]))
            parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))
            assert(parameters['w' + str(l)].shape ==
                   (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        return parameters


def NN_deep_layered(X, Y, layer_dim, num_iter=1000, learning_rate=0.05, print_cost=False,p_ty='ranm'):
    np.random.seed(1)
    parameters = {}
    NN_cost = []
    layer_dims = [X.shape[0]]+layer_dim
    L = len(layer_dims)
    print('Number of Hidden layers in this Neural Network',L-1)
    print('Units in each layer', layer_dim)
    A0 = X
    # initialization of intial weight parameters
    parameters = initialization_parameter(layer_dims, p_ty)
    #print(parameters)
    for i in range(num_iter):
        #performing forward propagationn and storing the require parameters for back prop
        #print(i)
        AL, fw_cache = forward_propagation(A0, parameters)
        #print(AL)
        # calculating cost function
        cst = cost_function(AL, Y)
        NN_cost.append(cst)
        if (i % 1000 == 0) and print_cost:
            print('At Iteration', i, ' Cost Function ', cst)
            
        #performing back prop and obtained derivatives of all parameters
        grad = backward_propagation(AL, Y, fw_cache)
        #print(grad)
        # performing gradient descent parameter update procedure
        for j in range(L-1):
            #print(j)
            parameters['w' + str(j+1)] -= learning_rate * grad['dw' + str(j+1)]
            parameters['b' + str(j+1)] -= learning_rate * grad['db' + str(j+1)]

    return parameters, NN_cost


def predict(X, Y, parameters, tol=0.5):
    m = X.shape[1]
    n_layers = len(parameters) // 2
    print('Hidden layer of the Neural Network', n_layers)
    p = np.zeros((1, m))
    # Forward propagation
    probas, fw_caches = forward_propagation(X, parameters)
    # Activation function
    for i in range(0, probas.shape[1]):
        if probas[0, i] > tol:
            p[0, i] = 1
        else:
            p[0, i] = 0
    print("Accuracy: " + str(np.sum((p == Y)/m)*100)+"%")
    return p
