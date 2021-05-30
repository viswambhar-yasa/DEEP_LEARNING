from NN_activation_function import activation_function
import numpy as np
def initial_variable(dim):
    w=np.zeros((1,dim))
    b=0.0
    return w,b

def hypothesis(X,w,b):
    h=np.dot(w.T,X)+b
    #print('X,w,b,h',X,w,b)
    #print('h',h)
    return h

def cost_funtion(A,Y):
    m=Y.size
    cost=-np.mean(Y*np.log(A)+(1-Y)*np.log(1-A))
    return cost


def propagation(X,Y,w,b):
    '''
    This funtion implement the cost function and its gradient for the forward and backward propagation 

    Inputs:
    X -- data of size 
    Y -- true "label" vector 
    w -- weights, a numpy array of size 
    b -- bias, a scalar

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    '''
    m=Y.size
    ## for forward propagation
    z=hypothesis(X,w,b)
    A=activation_function(z)
    cost=cost_funtion(A,Y)
    #print(z)
    ### gradients of the weight and bias functions for backward propagation

    dw=np.dot(X,(A-Y).T)/m
    db=np.sum(A-Y)/m

    gradient_para={'dw':dw,'db':db}

    return cost,gradient_para


def gradient_descent(X,Y,w,b,learning_rate=0.01,num_iter=150):
    '''
    This function performs gradient descent which find slope to attain global optima
    '''
    cost_fun=[]
    for i in range(num_iter):
        cost,gradient_para=propagation(X,Y,w,b)
        dw=gradient_para['dw']
        db=gradient_para['db']
        #print(dw,db)
        w-=learning_rate*dw
        b-=learning_rate*db
        #print(w,b)
        cost_fun.append(cost)
        if i% 250==0:
            print('Cost function at %i:%f'%(i,cost))
    parameters={'w':w,'b':b}
    return parameters,gradient_para,cost_fun

def predict(X,para):
    m=X.shape[1]
    Y_predicted=np.zeros((1,m))
    w=para['w']
    b=para['b']
    z=hypothesis(X,w,b)
    A=activation_function(z)
    for i in range(m):
        if A[0,i]<=0.5:
            Y_predicted[0,i]=0
        else:
            Y_predicted[0,i]=1
    return Y_predicted

def logistic_model(X_train, Y_train, X_test, Y_test, num_iter=2000, learning_rate=0.5, print_cost=False):
    """
    Building logistic regression using Neural network framework
    
    Inputs:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to True to print the cost every 100 iterations
    
    Outputs:
    model_parameters -- dictionary containing information about the model.
    """
    dim=X_train.shape
    w=np.zeros((dim[0],1))
    b=0.0
    parameters,gradient_para,cost=gradient_descent(X_train,Y_train,w,b,learning_rate,num_iter)

    Y_train_predicted=predict(X_train,parameters)
    Y_test_predicted=predict(X_test,parameters)

    if print_cost:
        print('The train accuracy of the model',(100 - np.mean(np.abs(Y_train_predicted - Y_train)) * 100))
        print('The test accuracy of the model',(100 - np.mean(np.abs(Y_test_predicted - Y_test)) * 100))
    
    model_parameters={'w':w,
                        'b':b,
                        'learning_rate':learning_rate
                        ,'num_iter':num_iter
                        ,'cost':cost
                        ,'Y_train_predicted':Y_train_predicted
                        ,'Y_test_predicted':Y_test_predicted
                        }
    
    return model_parameters
