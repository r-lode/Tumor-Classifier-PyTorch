import torch
from torch import nn

class Custom_TinyVGG(nn.Module):
    '''
    Neural network based on the TinyVGG architecture

    Args:

    input_shape (int): Number of color channels
    num_hidden_neurons (int): Number of hidden neurons in network layers
    output_shape (int): Number of tumor classes 

    Attributes:
        conv_block_1 (nn.Sequential): First convolutonal block, contains two
        layers and max pooling
        conv_block_2 (nn.Sequential): Second convolutional block, contains two
        layers and max pooling
        classifier (nn.Sequential): classifier layer

    Example Usage:

    model = Custom_TinyVGG(input_shape = 3, #RBG color channels
            num_hidden_neurons = 10, 
            output_shape = 4) #There are four tumor classes
    '''
    def __init__(self, input_shape, num_hidden_neurons,  output_shape: int, flattened_shape) -> None:
        super().__init__()


        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=num_hidden_neurons, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=num_hidden_neurons, 
                      out_channels=num_hidden_neurons,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(num_hidden_neurons, 
                      num_hidden_neurons, 
                      kernel_size=3, 
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(num_hidden_neurons, 
                      num_hidden_neurons, 
                      kernel_size=3, 
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_shape,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):

        return self.classifier(self.conv_block_2(self.conv_block_1(x)))
    

