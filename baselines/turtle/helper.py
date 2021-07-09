import pdb
import copy
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ConvBlock(nn.Module):
    """
    Initialize the convolutional block consisting of:
     - 64 convolutional kernels of size 3x3
     - Batch normalization 
     - ReLU nonlinearity
     - 2x2 MaxPooling layer
     
    ...

    Attributes
    ----------
    cl : nn.Conv2d
        Convolutional layer
    bn : nn.BatchNorm2d
        Batch normalization layer
    relu : nn.ReLU
        ReLU function
    mp : nn.MaxPool2d
        Max pooling layer
    running_mean : torch.Tensor
        Running mean of the batch normalization layer
    running_var : torch.Tensor
        Running variance of the batch normalization layer
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    """
    
    def __init__(self, dev, indim=3, pool=True):
        """Initialize the convolutional block
        
        Parameters
        ----------
        indim : int, optional
            Number of input channels (default=3)
        """
        
        super().__init__()
        self.dev = dev
        self.cl = nn.Conv2d(in_channels=indim, out_channels=64,
                            kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=64, momentum=1) #momentum=1 is crucial! (only statistics for current batch)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size=2)
        self.pool = pool
        
    def forward(self, x):
        """Feedforward pass of the network

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 

        Returns
        ----------
        tensor
            The output of the block
        """

        x = self.cl(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool:
            x = self.mp(x)
        return x
    
    def forward_weights(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        
        # Apply conv2d
        x = F.conv2d(x, weights[0], weights[1], padding=1) 

        # Manual batch normalization followed by ReLU
        running_mean =  torch.zeros(64).to(self.dev)
        running_var = torch.ones(64).to(self.dev)
        x = F.batch_norm(x, running_mean, running_var, 
                         weights[2], weights[3], momentum=1, training=True)
        if self.pool:                   
            x = F.max_pool2d(F.relu(x), kernel_size=2)
        return x
    
    def reset_batch_stats(self):
        """Reset Batch Normalization stats
        """
        
        self.bn.reset_running_stats()


class ConvX(nn.Module):
    """
    Convolutional neural network consisting of X ConvBlock layers.
     
    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Full sequential specification of the model
    in_features : int
        Number of input features to the final output layer
    train_state : state_dict
        State of the model for training
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
        
    freeze_layers()
        Freeze all hidden layers
    """
    
    def __init__(self, dev, train_classes, eval_classes, criterion=nn.CrossEntropyLoss(), num_blocks=4, img_size=(1,3,84,84)):
        """Initialize the conv network

        Parameters
        ----------
        dev : str
            Device to put the model on
        train_classes : int
            Number of training classes
        eval_classes : int
            Number of evaluation classes
        criterion : loss_fn
            Loss function to use (default = cross entropy)
        """
        
        super().__init__()
        self.num_blocks = num_blocks
        self.dev = dev
        self.in_features = 3*3*64
        self.criterion = criterion
        self.train_classes = train_classes
        self.eval_classes = eval_classes

        rnd_input = torch.rand(img_size)

        d = OrderedDict([])
        for i in range(self.num_blocks):
            indim = 3 if i == 0 else 64
            pool = i < 4 
            d.update({'conv_block%i'%i: ConvBlock(dev=dev, pool=pool, indim=indim)})
        d.update({'flatten': Flatten()})
        self.model = nn.ModuleDict({"features": nn.Sequential(d)})

        self.in_features = self.get_infeatures(rnd_input).size()[1]
        self.model.update({"out": nn.Linear(in_features=self.in_features,
                                            out_features=self.train_classes).to(dev)})


    def get_infeatures(self,x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        return x

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        return x

    def forward_weights(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[-2], weights[-1])
        return x
    
    def forward_get_features(self, x):
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[-2], weights[-1])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen
    
    def load_params(self, state_dict):
        del state_dict["model.out.weight"]
        del state_dict["model.out.bias"]
        self.load_state_dict(state_dict, strict=False)

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self, freeze):
        """Freeze all hidden layers
        """
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.out = nn.Linear(in_features=self.in_features,
                                   out_features=self.eval_classes).to(self.dev)
        self.model.out.bias = nn.Parameter(torch.zeros(self.model.out.bias.size(), device=self.dev))


