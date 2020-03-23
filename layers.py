from builtins import range
import numpy as np
import math


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.
    The input x has shape (N, d_in) and contains a minibatch of N
    examples, where each example x[i] has d_in element.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_in)
    - w: A numpy array of weights, of shape (d_in, d_out)
    - b: A numpy array of biases, of shape (d_out,)
    Returns a tuple of:
    - out: output, of shape (N, d_out)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in the variable out. #
    ###########################################################################
   
    
    x1=x.reshape(x.shape[0],-1)
    out = np.dot(x1, w) + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, d_out)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_in)
      - w: Weights, of shape (d_in, d_out)
      - b: Biases, of shape (d_out,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d_in)
    - dw: Gradient with respect to w, of shape (d_in, d_out)
    - db: Gradient with respect to b, of shape (d_out,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    x1 = x.reshape(x.shape[0],-1)
    dx1 = dout.dot(w.T)
    dx = dx1.reshape(x.shape)
    dw = np.dot(x1.T,dout)
    db = np.sum(dout,axis=0)
    

   

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    
    out=np.maximum(0,x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    y=(x>0)
    x=y.astype(int)
    dx=np.multiply(dout,x)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We filter each input with F different filters, where each filter
    spans all C channels and has height H' and width W'. 
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, H', W')
    Returns a tuple of:
    - out: Output data, of shape (N, F, HH, WW) where H' and W' are given by
      HH = H - H' + 1
      WW = W - W' + 1
    - cache: (x, w)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass based on the definition  #
    # of Y in Q1(c).                                                          #
    ###########################################################################

                
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape


    out_h = (1+(H-HH))
    out_w = (1+(W-WW))

    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, W, 1)
    strides = x.itemsize * np.array(strides)
    x_stride = np.lib.stride_tricks.as_strided(x,
                  shape=shape, strides=strides)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * out_h * out_w)

    res = w.reshape(F, -1).dot(x_cols)
    res.shape = (F, N, out_h, out_w)
    out = res.transpose(1, 0, 2, 3)
    out = np.ascontiguousarray(out)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w) as in conv_forward
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################

    X, W = cache
    n_filter, d_filter, h_filter, w_filter = W.shape
    
    N, C, H, W1 = X.shape
    out_height = (H - h_filter) + 1
    out_width = (W1 - w_filter) + 1

    i0 = np.repeat(np.arange(h_filter), w_filter)
    i0 = np.tile(i0, C)
    i1 = np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(w_filter), h_filter * C)
    j1 = np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), h_filter * w_filter).reshape(-1, 1)
    cols = X[:, k, i, j]
    C = X.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(h_filter * w_filter * C, -1)
    X_col = cols

    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    dW = dout_reshaped @ X_col.T
    dw = dW.reshape(W.shape)

    W_reshape = W.reshape(n_filter, -1)
    dX_col = W_reshape.T @ dout_reshaped

    N, C, H, W1 = X.shape
    H_padded, W_padded = H , W1 
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=dX_col.dtype)
    
    cols_reshaped = dX_col.reshape(C * h_filter * w_filter, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    dx = x_padded

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.
    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here. Output size is given by 
    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################

    
    N, C, H, W = x.shape
    height = pool_param['pool_height']
    width = pool_param['pool_width']
    stride = pool_param['stride']
    H1 = 1+(H-height)//stride
    W1 = 1+(W-width)//stride
  
    out=np.zeros((N,H1,W1,C))
    x1=x.reshape(N,H,W,C).copy()
    
    for i in range(H1):
        v_start = i * stride
        v_end = v_start + height

        for j in range(W1):
            h_start = j * stride
            h_end = h_start + width
            a_prev_slice = x1[:, v_start:v_end, h_start:h_end, :]
            out[:, i, j, :] = np.max(a_prev_slice, axis=(1, 2))

    out=out.reshape(N,C,H1,W1) 

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################

            
    x, pool_param = cache
    N, C, H, W = x.shape
    height = pool_param['pool_height']
    width = pool_param['pool_width']
    stride = pool_param['stride']
    H1 = ((H-height)//stride) + 1
    W1 = ((W-width)//stride) + 1
  
    dx=np.zeros((N,H,W,C))
    
    x1=x.reshape(N,H,W,C).copy()
    dout1=dout.reshape(dout.shape[0],dout.shape[2],dout.shape[3],dout.shape[1]).copy()

    xx=-1
    yy=-1
    for i in range(H1):
        v_start = i * stride
        v_end = v_start + height
        yy+=1

        for j in range(W1):
            h_start = j * stride
            h_end = h_start + width
            xx +=1
            
            x2 = x1[:, v_start:v_end, h_start:h_end, :].copy()
            mask = np.zeros_like(x2)
            reshaped_x = x2.reshape(x2.shape[0], x2.shape[1] * x2.shape[2], x2.shape[3])
            idx = np.argmax(reshaped_x, axis=1)
            
            ax1, ax2 = np.indices((x2.shape[0], x2.shape[3]))
            mask.reshape(mask.shape[0], mask.shape[1] * mask.shape[2], mask.shape[3])[ax1, idx, ax2] = 1

            dx[:, v_start:v_end, h_start:h_end, :] += dout1[:, i:i+1, j:j+1, :] * mask
        xx = -1
            
    dx=dx.reshape(N,C,H,W)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.
  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  loss, dx = None, None
  N,C=x.shape
  y_predicted=np.exp(x-np.max(x, axis = 1,keepdims = True))/np.sum(np.exp(x-np.max(x, axis = 1,keepdims = True)),axis=1,keepdims=True)
  loss=-np.sum(np.log(y_predicted[range(N),y]))/N
  gradient = y_predicted.copy()
  gradient[range(N),y] -= 1
  gradient= gradient/N
  dx = gradient
      
  
  return loss, dx
