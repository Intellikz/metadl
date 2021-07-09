import tensorflow as tf
import torch
import torch.nn as nn
import contextlib
import math
import numpy as np

def create_grads_shell(model):
    """ Create list of gradients associated to each trainable layer in model.
    
    Returns:
    -------
    list_grads, array-like : each element of this list is tensor representing 
        the associated layer's gradient.
    """


    list_grads = []
    for layer in model.trainable_variables :
        list_grads.append(tf.Variable(tf.zeros_like(layer)))

    return list_grads

def reset_grads(meta_grads):
    """Reset the variable that contains the meta-learner gradients.
    Arguments:
    ----------
    meta_grads : list of tf.Variable

    Note : Each element is guaranteed to remain a tf.Variable. Using
    tf.zeros_like on tf.Variable does not transform the element to 
    tf.Tensor
    """
    for ele in meta_grads :
        ele.assign(tf.zeros_like(ele))


def app_custom_grads(model, inner_gradients, lr):
    """ Apply gradient update to the model's parameters using inner_gradients.
    """
    i = 0
    #print(inner_gradients)
    for k, layer in enumerate(model.layers) :
        if 'kernel' in dir(layer) : 
            #print(layer.kernel.shape)
            layer.kernel.assign_sub(tf.multiply(lr, inner_gradients[i]))
            i+=1
        elif 'normalization' in layer.name:
            layer.trainable_weights[0].assign_sub(\
                tf.multiply(lr, inner_gradients[i]))
            
            i+=1
        if 'bias' in dir(layer):
            layer.bias.assign_sub(tf.multiply(lr, inner_gradients[i]))
            i+=1
        elif 'normalization' in layer.name:
            layer.trainable_weights[1].assign_sub(\
                tf.multiply(lr, inner_gradients[i]))
            i+=1


def put_on_device(dev, tensors):
    """Put arguments on specific device

    Places the positional arguments onto the user-specified device

    Parameters
    ----------
    dev : str
        Device identifier
    tensors : sequence/list
        Sequence of torch.Tensor variables that are to be put on the device
    """
    for i in range(len(tensors)):
        if not tensors[i] is None:
            tensors[i] = tensors[i].to(dev)
    return tensors



def get_params(model, dev):
    """Get parameters of the model (ignoring batchnorm)

    Retrieves all parameters of a given model and computes slices 

    Parameters
    ----------
    model : nn.Module
        Pytorch model from which we extract the parameters
    dev : str
        Device identifier to place the parameters on

    Returns
    ----------
    params
        List of parameter tensors
    slices
        List of tuples (lowerbound, upperbound) that delimit layer parameters
        E.g., if model has 2 layers with 50 and 60 params, the slices will be
        [(0,50), (50,110)]
    """

    params = []
    slices = []

    lb = 0
    ub = 0
    for m in model.modules():
        # Ignore batch-norm layers
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            # All parameters in a given layer
            mparams = m.parameters()
            sizes = []
            for mp in mparams:
                sizes.append(mp.numel())
                params.append(mp.clone().detach().to(dev)) 
            # Compute the number of parameters in the layer
            size = sum(sizes)
            # Compute slice indices of the given layer 
            ub += size
            slices.append(tuple([lb, ub]))
            lb += size
    return params, slices


loss_to_init_and_op = {
    nn.MSELoss: (float("inf"), min),
    nn.CrossEntropyLoss: (-float("inf"), max)
}

def get_init_score_and_operator(criterion):
    """Get initial score and objective function

    Return the required initialization score and objective function for the given criterion.
    For example, if the criterion is the CrossEntropyLoss, we want to maximize the accuracy.
    Hence, the initial score is set to -infty and we with to maximize (objective function is max)
    In case that the criterion is MSELoss, we want to minimize, and our initial score will be
    +infty and our objective operator min. 
    
    Parameters
    ----------
    criterion : loss_fn
        Loss function used for the base-learner model
    """
    
    for loss_fn in loss_to_init_and_op.keys():
        if isinstance(criterion, loss_fn):
            return loss_to_init_and_op[loss_fn]

