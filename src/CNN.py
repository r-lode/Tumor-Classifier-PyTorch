import torch
from torch import nn

class Custom_TinyVGG(nn.Module):
    '''
    Neural network based on the TinyVGG architecture

    Args:

    input_shape (int): Number of color channels
    hidden_units (int): Number of hidden neurons in network layers
    output_shape (int): Number of tumor classes 
    '''