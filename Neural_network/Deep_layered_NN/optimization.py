
import numpy as np
import math
from NN_deep_layer import forward_propagation, backward_propagation, predict, cost_function,initialization_parameter
def gradient_descent(parameter,grads,learning_rate):
    n_layers = int(len(parameter)/2)
    #print('n_layers gd', n_layers)
    for i in range(1,n_layers+1):
        parameter['w'+str(i)]-=learning_rate*grads['dw'+str(i)]
        parameter['b'+str(i)] -= learning_rate*grads['db'+str(i)]
    return parameter


def gradient_descent_momentum(parameter, grads,v,learning_rate,beta=0.9):
    n_layers = int(len(parameter)/2)
    for i in range(1, n_layers+1):
        # performing moving average to optimize gradient descent
        v['w'+str(i)] = beta*v['w'+str(i)]+(1-beta)*grads['dw'+str(i)]
        v['b'+str(i)] = beta*v['b'+str(i)]+(1-beta)*grads['db'+str(i)]
        #print(v)
        parameter['w'+str(i)] -= learning_rate*v['w'+str(i)]
        parameter['b'+str(i)] -= learning_rate*v['b'+str(i)]
    return parameter,v


def adam_optimizer(parameter, grads, v,s,t, learning_rate, beta1,beta2,tol=1e-8):
    n_layers = int(len(parameter)/2)
    v_corrected = {}
    s_corrected = {}
    for l in range(1, n_layers+1):
        # performing moving average to optimize gradient descent
        v['w' + str(l)] = (beta1*v['w' + str(l)])+(1-beta1)*grads['dw'+str(l)]
        v['b' + str(l)] = (beta1*v['b' + str(l)])+(1-beta1)*grads['db'+str(l)]
        v_corrected['w'+str(l)] = (v['w' + str(l)])/(1-(beta1**t))
        v_corrected['b'+str(l)] = (v['b' + str(l)])/(1-(beta1**t))

        s['w' + str(l)] = (beta2*s['w' + str(l)]) + (1-beta2)*(grads['dw'+str(l)]**2)
        s['b' + str(l)] = (beta2*s['b'+ str(l)]) + (1-beta2)*(grads['db'+str(l)]**2)
        s_corrected['w'+str(l)] = (s['w' + str(l)])/(1-(beta2**t))
        s_corrected['b'+str(l)] = (s['b' + str(l)])/(1-(beta2**t))

        parameter['w' + str(l)] -= learning_rate*(v_corrected['w' +str(l)]/(np.sqrt(s_corrected['w'+str(l)])+tol))
        parameter['b' + str(l)] -= learning_rate*(v_corrected['b' +str(l)]/(np.sqrt(s_corrected['b'+str(l)])+tol))

        '''
        v['w'+str(i)] = beta1*v['w'+str(i)]+(1-beta1)*grads['dw'+str(i)]
        v['b'+str(i)] = beta1*v['b'+str(i)]+(1-beta1)*grads['db'+str(i)]
        v_co['w'+str(i)] = (v['w'+str(i)])/(1-(beta1**t))
        v_co['b'+str(i)] = (v['b'+str(i)])/(1-(beta1**t))
        # performing Root mean square to optimize gradient descent
        s['w'+str(i)] = beta2*s['w'+str(i)]+(1-beta2)*grads['dw'+str(i)]**2
        s['b'+str(i)] = beta2*s['b'+str(i)]+(1-beta2)*grads['db'+str(i)]**2
        s_co['w'+str(i)] = (s['w'+str(i)])/(1-(beta2**t))
        s_co['b'+str(i)] = (s['b'+str(i)])/(1-(beta2**t))
        parameter['w'+str(i)] -= learning_rate * ((v_co['w'+str(i)])/(np.sqrt(s_co['w'+str(i)])+tol))
        parameter['b'+str(i)] -= learning_rate * ((v_co['b'+str(i)]/(np.sqrt(s_co['b'+str(i)])+tol)))
        '''
    return parameter,v,s


def lr_decay(epoch,learning_rate0,decayrt,type=0,time_interval=1000):
    if type==0:
        learning_rate= learning_rate0/(1+(decayrt*epoch))
        return learning_rate
    elif type==1:
        learning_rate = learning_rate0/(1+(decayrt*np.floor(epoch/time_interval)))
        return learning_rate
    else:
        learning_rate = (0.95**(epoch))*learning_rate0
        return learning_rate


def random_mini_batches(X, Y, mini_batch_size=64):

    m = X.shape[1]                  # number of training examples
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    inc = mini_batch_size
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:, k *mini_batch_size: (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k *mini_batch_size: (k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, mini_batch_size *math.floor(m/mini_batch_size):]
            mini_batch_Y = shuffled_Y[:, mini_batch_size *math.floor(m/mini_batch_size):]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def NN_deep_layered_optimized(X, Y, layer_dim, num_iter=1000, mini_batch_size=64, learning_rate=0.05, beta1=0.9, beta2=0.999, t=2, opt_type='gd', p_ty='ranm',print_cost=False, learning_dacay=False, decayrt=1, lr_type=0, time_interval=1000):
    np.random.seed(1)
    parameters = {}
    NN_cost = []
    layer_dims = [X.shape[0]]+layer_dim
    L = len(layer_dims)
    print('Number of Hidden layers in this Neural Network', L-1)
    print('Units in each layer', layer_dim)
    # initialization of intial weight parameters
    parameters = initialization_parameter(layer_dims, p_ty)
    #print(parameters.keys())
    if opt_type=='mom':
        #initializing dictionary to store exponentional moving weights
        v = initialization_parameter(layer_dims, p_ty)
        for l in v.keys():
            v[l]=np.zeros(v[l].shape)
    if opt_type == 'adam':
        #initializing dictionary to store exponentional moving weights and Root mean square 
        v = initialization_parameter(layer_dims, p_ty)
        s = initialization_parameter(layer_dims, p_ty)
        for l in v.keys():
            v[l] = np.zeros(v[l].shape)
            s[l] = np.zeros(s[l].shape)
    A0=X
    lr0 = learning_rate
    for i in range(num_iter):
        # getting mini batches to implement stchastic 
        #minibatches = random_mini_batches(X, Y, mini_batch_size)
        #for mini_bt in minibatches:
        #performing forward propagationn and storing the require parameters for back prop
        #A0,Y = mini_bt
        #print(A0.shape)
        AL, fw_cache = forward_propagation(A0, parameters)
        #print(AL)
        # calculating cost function
        cst = cost_function(AL, Y)
        NN_cost.append(cst)
        if (i % 1000 == 0) and print_cost:
            print('At Iteration', i, ' Cost Function ', cst)
        #performing back prop and obtained derivatives of all parameters
        grad = backward_propagation(AL, Y, fw_cache)
        
        if learning_dacay:
            if (i % 1000 == 0) and print_cost:
                print('learning rate after epoch', i, '', learning_rate)
            learning_rate=lr_decay(i, lr0, decayrt,lr_type, time_interval)
        #print(grad.keys())
        # performing gradient descent parameter update procedure
        if opt_type=='mom':
            #print(i)
            parameters,v=gradient_descent_momentum(parameters, grad, v, learning_rate, beta1)
        elif opt_type=='adam':
            parameters, v,s=adam_optimizer(parameters, grad, v, s, t, learning_rate, beta1, beta2, tol=1e-8)
        else:
            parameters=gradient_descent(parameters, grad, learning_rate)
        for j in range(L-1):
            #print(j)
            parameters['w' + str(j+1)] -= learning_rate * grad['dw' + str(j+1)]
            parameters['b' + str(j+1)] -= learning_rate * grad['db' + str(j+1)]

    return parameters, NN_cost
