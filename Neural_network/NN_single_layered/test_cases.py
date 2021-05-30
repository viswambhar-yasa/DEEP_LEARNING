from NN_activation_function import activation_function
from NN_logistic_reg import propagation,gradient_descent,predict,logistic_model
import pytest 
import numpy as np

def test__activation_function_basic1():
    Z = 1
    actual_output=activation_function(Z)
    expected_output = 0.7310585786300049
    assert ((float(actual_output)==expected_output) and (actual_output.dtype=='float64')) is True



def test__activation_function_basic2():
    Z = np.array([1, 2, 3])
    expected_output = np.array([0.73105858,
                                0.88079708,
                                0.95257413])
    actual_output=activation_function(Z)
    assert np.array_equiv(np.round(actual_output,8),expected_output) is True



def test__propagation():
    X =np.array([[1., 2., -1.], [3., 4., -3.2]])
    Y = np.array([[1, 0, 1]])
    w =  np.array([[1.], [2.]])
    b = 2.  
    cost,para = propagation(X, Y, w, b)
    expected_output_dw= np.array([[0.99845601],
       [2.39507239]])
    expected_output_db=0.00145558
    dw=para['dw']
    db=para['db']   
    assert type(dw) == np.ndarray
    assert expected_output_db==float(round(db,8))
    assert ((np.array_equiv(expected_output_dw,np.round(dw,8)))   and (float(round(cost,8))==5.80154532)) is True



def test_gradient_descent():
    X =np.array([[1., 2., -1.], [3., 4., -3.2]])
    Y = np.array([[1, 0, 1]])
    w =  np.array([[1.], [2.]])
    b = 2.0 
    parameters,grads,cost = gradient_descent(X,Y,w,b,learning_rate=0.009,num_iter=100)
    output_w=parameters['w']
    output_b=parameters['b']
    dw=grads['dw']
    db=grads['db']
    expected_w = np.array([[0.19033591]
                    ,[0.12259159]])
    expected_b = 1.92535983
    expected_dw = np.array([[0.67752042]
        ,[1.41625495]])
    expected_db = 0.21919450
    Costs = [np.array(5.80154532)]
    assert (np.array_equiv(expected_dw,np.round(dw,8))) is True
    assert expected_db==round(db,8)
    assert expected_b==round(output_b,8)
    assert (np.array_equiv(expected_w,np.round(output_w,8))) is True



def test__predict():
    w = np.array([[0.3], [0.5], [-0.2]])
    b = -0.33333
    X = np.array([[1., -0.3, 1.5],[2, 0, 1], [0, -1.5, 2]])
    para={'w':w,'b':b}
    pred = predict(X,para)
    assert type(pred) == np.ndarray
    assert np.array_equiv(pred,np.array([[1., 0., 1]])) is True


def model_test():
    np.random.seed(0)
    expected_output = {'costs': [np.array(0.69314718)],
                     'Y_prediction_test': np.array([[1., 1., 1.]]),
                     'Y_prediction_train': np.array([[1., 1., 1.]]),
                     'w': np.array([[ 0.00194946],
                            [-0.0005046 ],
                            [ 0.00083111],
                            [ 0.00143207]]),
                     'b': np.float64(0.000831188)
                      }
    
    dim, b, Y, X = 5, 3., np.array([1, 0, 1]).reshape(1, 3), np.random.randn(4, 3),
    X_test = X * 0.5
    Y_test = np.array([1, 0, 1])
    final_para=logistic_model(X, Y, X_test, Y_test, num_iter=2000, learning_rate=0.5, print_cost=False)
    expected_w=expected_output['w']
    output_w=final_para['w']
    expected_b=expected_output['b']
    output_b=final_para['b']
    assert expected_b==round(output_b,8)
    assert (np.array_equiv(expected_w,np.round(output_w,8))) is True