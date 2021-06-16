import numpy as np
from numpy.lib.function_base import average


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

def padding(X,pad):
    '''
    padding of images to generate  
    '''
    s=X.shape
    dim=len(s)
    D=((0,0),(pad,pad),(pad,pad),(0,0))

    X_padded = np.pad(X, pad_width=D, mode='constant', constant_values=(0, 0))
    return X_padded

def convolution(A_silded,w,b):
    z_sub = (np.sum(A_silded*w)+np.float(b)).flatten()
    return z_sub[0]

def foward_convol(A0,w,b,h_parameters,typ='relu'):
    m,n_h_pre,n_v_pre,n_c_pre=A0.shape
    f,f,n_c_pre,n_c=w.shape
    pad = h_parameters["pad"]
    stride = h_parameters["stride"]
    n_h=int((n_h_pre-f+2*pad)/stride)+1
    n_v=int((n_v_pre-f+2*pad)/stride)+1
    z=np.zeros((m,n_h,n_v,n_c))
    for i in range(m):
        for j in range(n_h):
            for k in range(n_v):
                for l in range(n_c):
                    h_str = j*stride
                    h_stp = h_str+f
                    v_str = k*stride
                    v_stp = v_str+f
                    A_slided=A0[i,h_str:h_stp,v_str:v_stp,:]
                    weights=w[:,:,:,l]
                    bias=b[:,:,:,l]
                    z[i, j, k, l] = convolution(A_slided, weights, bias)
    A = activation_function(z,typ)
    cache=(A0,w,b,h_parameters)
    return A,z,cache

def forward_pool(A0,h_parameters,mode='max'):
    m, n_h_pre, n_v_pre, n_c = A0.shape
    f = h_parameters["f"]
    stride = h_parameters["stride"]
    n_h = int(1 + (n_h_pre - f) / stride)
    n_v=int(1 + (n_v_pre - f) / stride)
    z = np.zeros((m, n_h, n_v, n_c))
    for i in range(m):
        for j in range(n_h):
            for k in range(n_v):
                for l in range(n_c):
                    h_str = j*stride
                    h_stp = h_str+f
                    v_str = k*stride
                    v_stp = v_str+f
                    A_slided = A0[i, h_str:h_stp, v_str:v_stp, l]
                    if mode=="max":
                        A = np.max(A_slided)
                    elif mode=="avg":
                        A=np.mean(A_slided)     
    cache = (A0,h_parameters)
    return A,cache


def backward_conv(dZ,cache):

    A_pre,w,b,h_parameters=cache
    m,n_h,n_v,n_c_pre=A_pre.shape
    f,f,n_c_pre,n_c=w.shape
    pad = h_parameters["pad"]
    stride=h_parameters["stride"]
    dA=np.zeros((m,n_h,n_v,n_c_pre))
    dw=np.zeros(f,f,n_c_pre,n_c)
    db=np.zeros((1,1,1,n_c))
    A_pad=padding(A_pre,pad)
    dA_pad = padding(dA,pad)
    for i in range(m):
        da_prev=dA_pad[i]
        for j in range(n_h):
            for k in range(n_v):
                for l in range(n_c):
                    v_st=j*stride
                    v_stp=v_st+f
                    h_st=k*stride
                    h_stp=h_st+f
                    a_slice=A_pad[i,v_st:v_stp,h_st:h_stp,:]
                    da_prev+=w[:,:,:,l]*dZ[i,j,k,l]
                    dw += a_slice*dZ[i, j, k, l]
                    db+=dZ[i,j,k,l]

        dA[i,:,:,:]=da_prev[pad:-pad,pad:-pad,:]
    return dA,dw,db


def masking(x):
    m = np.max(x)
    mask = (x == m)
    return mask

def averging(dz,shape):
    temp_matrix=np.ones(shape)
    n_h,n_v=shape
    average = (dz/(n_h*n_v))*temp_matrix
    return average

def backward_pool(dA,cache,mode='max'):
    A_pre,h_par=cache
    m,n_h_pre,n_v_pre,n_c_pr=A_pre.shape
    m, n_h, n_v, n_c=dA.shape
    f=h_par["f"]
    stride=h_par["stride"]
    dA_pre=np.zeros((m,n_h,n_v,n_c))
    for i  in range(m):
        A_pre_sl = A_pre[i]
        for j in range(n_h):
            for k in range(n_v):
                for l in range(n_c):
                        v_st=j*stride
                        v_stp=v_st+f
                        h_st=k*stride
                        h_stp=h_st+f
                        if mode=='max':
                            a_slice = A_pre_sl[v_st:v_stp, h_st:h_stp, l]
                            mx = masking(a_slice)
                            dA_pre[i, v_st:v_stp, h_st:h_stp,l] += np.multiply(mx,dA[i,j,k,l])
                        elif mode=='avg':
                            da_slice = dA[i,j,k,l]
                            shape=(f,f)
                            dA_pre[i, v_st:v_stp, h_st:h_stp, l] += averging(da_slice, shape)
        return dA_pre



    


np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = padding(x, 3)
print(x_pad.shape)
