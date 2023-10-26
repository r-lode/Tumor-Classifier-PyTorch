

def calculate_flattened_size(height, width , num_hidden_neurons, num_conv_blocks=2):
    
    """

    Get size of the flattened feature map in a CNN. Computation done by applying the series of convolutional
    and max-pooling operations on the input height and width, as well as the number of hidden neurons.


    Params:
        height (int): The height of the input feature map.
        width (int): The width of the input feature map.
        num_hidden_neurons (int): The number of hidden neurons in the network.
        num_conv_blocks (int, optional): The number of convolutional blocks in the netwok. Set default
        manually based on number of blocks

    Note:
        Function assumes a specific network architecture with 3x3 convolutional kernels,
        padding, and 2x2 max-pooling operations applied in each convolutional block.

    Function walkthrough example:
        To calculate the flattened size for a network with an input feature map of size 224x224:

        Sample input:

        height = 224
        width = 224
        num_hidden_neurons = 10
        num_conv_blocks = 2

        Calling function:

        flattened_size = calculate_flattened_size(height, width , num_hidden_neurons, num_conv_blocks=2)

        Given the convolutional block with a 3x3 kernel and padding, we perform these adjustments to calculate flattened shape: 
            -Subtracting 3 (the kernel size) accounts for the region covered by the convolution kernel. 
            -Given a padding of 1, adding 2 ensures the output size accommodates the kernel, preventing edge data loss. 
            -Dividing by 1 (stride) and adding 1 maintains spatial dimensions after each convolution operation. 
            
            This calculaton ensures output size is consistent with the kernel size.

            max-pooling with a 2x2 kernel and stride 2 further reduces spatial dimensions by a factor of 2, 
            capturing essential features while downsizing the feature map.
        
        for _ in range(num_conv_blocks):
        
       
            height = (height - 3 + 2) // 1 + 1
            width = (width - 3 + 2) // 1 + 1
            
            # max-pooling further reduces the spatial dimensions by a factor of 2:
            height = height // 2
            width = width // 2
            
    

        # calculate the total flattened size by multiplying adjusted side lengths with the number of hidden neurons

        flattened_size = num_hidden_neurons * height * width


    """

    #simulate a convolutional block with a 3x3 kernel and padding
    for i in range(num_conv_blocks):

        # subtracting 3 (kernel size) and adding 2 (padding) in both height and width adjusts for the
        # impact of a 3x3 convolution kernel with padding:

        height = (height - 3 + 2) // 1 + 1
        width = (width - 3 + 2) // 1 + 1

       # account for max pooling in calculation
        height = height // 2
        width = width // 2

    # calculate the total flattened size by multiplying adjusted side lengths with the number of hidden neurons
    flattened_size = num_hidden_neurons * height * width

    return flattened_size