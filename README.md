# Tumor-Classifier-PyTorch

## Project Tasks Status: 

### Data Preparation
- [:heavy_check_mark:] Create function to load data into the working directory
- [:heavy_check_mark:] Create a custom data class

### Model Development
- [:heavy_check_mark:] Build a baseline neural network class
- [:heavy_check_mark:] Build train and test model functions

### Project Management
- [:heavy_check_mark:] Create a function to train and test the model
- [:heavy_check_mark:] Containerize the project with Docker

### Ongoing Development
- [ ] Model Optimization
- [ ] Create Log to Track Model Changes
- [ :heavy_check_mark:] Add code to export generated models

### Documentation
- [ ] Finish README
    - [ ] Update Example Usage After model optimization functionality added
    - [ ] Create and Add Screenshots of Code Execution

## Table of Contents

1. [ Project Description](#project-description)
2. [ Libraries and Dependencies](#lib&dep)
3. [ Example Usage](#ex-use) 
4. [ Features](#features)
5. [ Acknowledgements](#ack)


<a name="project-description"></a>
## Project Description

The goal of this project is to use the PyTorch library to build a convolutional neural network to classify tumors given image data. Tumors are divided into four classes: no tumor, glioma tumor, meningioma_tumor, and pituitary tumor.

Data Source: [Brain Tumor Classification dataset](https://www.kaggle.com/datasets/prathamgrover/brain-tumor-classification)

Below is a sample of tumor images and their respective labels:

![Sample Image](./images/sample_images.png)

<a name="lib&dep"></a>
## Libraries and Dependencies

- **Libraries:**
  - PyTorch

- **Dependencies:**
  - torch
  - torchvision
  - torchmetrics
  - matplotlib
  - os
  - timeit
  - tqdm.auto
  - mlxtend
  - pathlib
  - PIL

<a name="ex-use"></a>
## Example Usage

Here we set global parameters:

-NUM_EPOCHS: Number of iterations our model will train
-SQUARE_IMAGE_SIZE: Images will be resized to nxn pixels. 
  -NOTE: At time of writing, this project only supports square images
-NUM_NEURONS: Number of hidden neurons in network layers
-MODEL_NAME: Model name that will be included in the model and log files

<pre>
```
    NUM_EPOCHS = 50
    SQUARE_IMAGE_SIZE = 200
    NUM_NEURONS = 10
    WORKING_IN_CONTAINER = 0
    MODEL_NAME = "model_1_10_26_2023"

```
</pre>

Here is the basic functionality of the main program. First we create an instane of the model, set up a
loss function and optimizer, and then train and evaluate the model. Model logging is handled in the 
train_evaluate_model function. Finally, we save the model state dict for later use.

<pre>

```
    #create an instance of model
    model_1 = Custom_TinyVGG(input_shape=3, #input num of color channels 1 for greyscale, 3 for RGB
                    num_hidden_neurons=NUM_NEURONS, 
                    flattened_shape= flattened_size,
                    output_shape=4)
    model_1.to(device)

    #set up loss function
    loss_fn = nn.CrossEntropyLoss()

    #set up optimizer
    optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)


    #train model
    model_1_results = train_evaluate_model(model=model_1, 
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            device=device,
                            logfile= logfile_path,
                            epochs=NUM_EPOCHS,
                            loss_fn=loss_fn)

    
    #save model state dict for later usage
    torch.save(model_1.state_dict(), model_save_path)


```
</pre>


<a name="features"></a>
## Features

This section will be completed after model optimization and adjustments to model architecture are made. 


<a name="ack"></a>
## Acknowledgements
VGG Architecture: [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

Background Knowledge: [https://www.learnpytorch.io/]


