import numpy as np
# Two -layered neural network with many hidden units

def activation_function(Z):
    '''
    Compute the activation function required from build NN(Neural Networks)

    Inputs:
    Z -- A scalar or numpy array of any size.

    Returns:
    s -- activation_function(z)
    '''

    AF=1/(1+np.exp(-Z))
    return AF

def hypothesis(X,w,b):
    h=np.dot(w,X)+b
    #print('X,w,b,h',X,w,b)
    #print('h',h)
    return h

def cost_function(A,Y):
    m=Y.size
    cost=(1/m)*np.sum((np.multiply(np.log(A),Y)+np.multiply(np.log(1-A),(1-Y))))
    return float(np.squeeze(cost))

def forward_propagation(X,parameters):
    '''
    This function calculates parameter obtained after forward propagation 

    '''
    w1=parameters['w1']
    b1=parameters['b1']
    w2=parameters['w2']
    b2=parameters['b2']
    Z1=hypothesis(X,w1,b1)
    A1=np.tanh(Z1)
    #A1=activation_function(Z1,type='tanh')
    Z2=hypothesis(A1,w2,b2)
    A2=activation_function(Z2)
    #print(A2.shape)
    fw_pro_cache ={'A1':A1,
                   'A2':A2,
                   'Z1':Z1,
                   'Z2':Z2}

    return A2,fw_pro_cache
    
def back_propagation(X,Y,parameters,cache):
    m=X.shape[1]
    W2=parameters['w2']
    Z1=cache['Z1']
    A1=cache['A1']
    A2=cache['A2']
    dZ2=A2-Y
    dw2=np.dot(dZ2,A1.T)/m
    db2=np.sum(dZ2,axis=1,keepdims=True)/m
    dZ1=np.dot(W2.T,dZ2)*(1-A1**2)
    dw1=np.dot(dZ1,X.T)/m
    db1=np.sum(dZ1,axis=1,keepdims=True)/m

    grads = {"dw1": dw1,
             "db1": db1,
             "dw2": dw2,
             "db2": db2}
    
    return grads

def nn_model(X,Y,dims,num_iter=1000,learning_rate=0.01,print_cost=False):
    np.random.seed(2)
    n_x=X.shape[0]
    n_y=Y.shape[0]
    #print(n_x,n_y)
    w1=np.random.randn(dims,n_x) * 0.01
    b1=np.zeros((dims,1))
    w2=np.random.randn(n_y,dims) * 0.01
    b2=np.zeros((n_y,1))
    #print(w1,b1,w2,b2)
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}
    cost_ap=[]
    for i in range(0, num_iter):
         
        A2, fw_cache = forward_propagation(X, parameters)
        #print(A2.shape)
        cost = cost_function(A2, Y)
        cost_ap.append(cost)
        grads =  back_propagation(X,Y,parameters,fw_cache)
        
        w1 =parameters["w1"]
        b1 =parameters["b1"]
        w2 =parameters["w2"]
        b2 =parameters["b2"]
        
        dw1 =grads["dw1"]
        db1 =grads["db1"]
        dw2 =grads["dw2"]
        db2 =grads["db2"]
        
        w1-=learning_rate*dw1
        b1-=learning_rate*db1
        w2-=learning_rate*dw2
        b2-=learning_rate*db2
        # YOUR CODE ENDS HERE
        
        parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}

        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return parameters,cost_ap

def predict(X,parameters):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    return predictions