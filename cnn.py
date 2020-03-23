import numpy as np

from layers import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - fc - relu - fc - softmax 
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network, with the following architecture                                 #
    # conv - relu - 2x2 max pool - fc - relu - fc - softmax                    #
    # Weights should be initialized from a Gaussian with standard              #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights for the convolutional layer using the keys 'W1' (here      #
    # we do not consider the bias term in the convolutional layer);            #
    # use keys 'W2' and 'b2' for the weights and biases of the                 #
    # hidden fully-connected layer, and keys 'W3' and 'b3' for the weights     #
    # and biases of the output fully-connected layer.                          #
    ############################################################################
    #self.params['W1']=np.random.normal(loc=0.0,scale=weight_scale,size=(input_dim,hidden_dim))
    #w1=(F,C,H',W')
    self.params['W1'] = weight_scale * np.random.randn(num_filters,input_dim[0],filter_size,filter_size)

    #w2=(F,C,H',W')
    #after convolution=(N,F,HH = H - H' + 1,WW = W - W' + 1)
    HH=input_dim[1]-filter_size+1
    WW=input_dim[2]-filter_size+1
    H_dash=1+(HH-2)//2
    W_dash=1+(WW-2)//2

    out = None
    
    self.params['W2'] = weight_scale * np.random.randn(num_filters * H_dash * W_dash, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    """
    W1 = self.params['W1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    # Hint: you may need to flatten the features after the max pooling layer   #
    # before feeding them into the fully connected layer.                      #
    
        # conv - relu - 2x2 max pool - fc - relu - fc - softmax       
    ############################################################################
    
    layer1_out,cache1=conv_forward(X,W1)
    relu_1,relu1_cache=relu_forward(layer1_out)
    max_pool_layer,max_pool_cache=max_pool_forward(relu_1,pool_param)
    Layer1_fc,Layer1_cache=fc_forward(max_pool_layer,W2,b2)
    Layer_relu,relu_cache=relu_forward(Layer1_fc)
    Layer2_fc,Layer2_cache=fc_forward(Layer_relu,W3,b3)
    scores=Layer2_fc
    
 

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k].                                                      #
    ############################################################################
    
    loss,loss_gradient=softmax_loss(Layer2_fc, y)
    dx3,grads['W3'],grads['b3']=fc_backward(loss_gradient,Layer2_cache)
    dx_relu=relu_backward(dx3,relu_cache)
    dx1,grads['W2'],grads['b2']=fc_backward(dx_relu,Layer1_cache)    
    dx3=max_pool_backward(dx1, max_pool_cache)
    dx_relu1=relu_backward(dx3,relu1_cache)
    dx4,grads['W1']=conv_backward(dx_relu1, cache1)
    
    
 

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  

