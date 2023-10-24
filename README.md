# Tumor-Classifier-PyTorch

## Table of Contents

1. [ Project Description](#project-description)
2. [ Libraries and Dependencies](#lib&dep)
3. [ Example Usage](#ex-use) 
4. [ Features](#features)
5. [ Acknowledgements](#ack)


<a name="project-description"></a>
## Project Description

The goal of this project is to use the PyTorch library to build a convolutional neural network to classify tumors given image data. Tumors are divided into four classes: no tumor, glioma tumor, meningioma_tumor, and pituitary tumor.

Data Source: [Brain Tumor Classification dataset](https://www.kaggle.com/datasets/prathamgrover/brain-tumor-classification).

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

<pre>
```
    #define an image transform to adapt image data accordingly before modeling
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    #create 
    train_dataset = Custom_Image_Dataset(image_directory=training_data_dir, transform=data_transform)
    test_dataset = Custom_Image_Dataset(image_directory=testing_data_dir, transform=data_transform)

    #create a train and test data loader to fetch batch data
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

    #set num iterations to train model
    NUM_EPOCHS = 50

    #create an instance of model
    model_0 = Custom_TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                    num_hidden_neurons=10, 
                    output_shape=4)
    model_0.to(device)

    #loss function
    loss_fn = nn.CrossEntropyLoss()

    #set up optimizer
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

    #start timer
    from timeit import default_timer as timer 
    start_time = timer()

    #train model
    model_0_results = train_evaluate_model(model=model_0, 
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            device=device,
                            epochs=NUM_EPOCHS,
                            loss_fn=loss_fn)

    #end timer, output total time
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")
```

</pre>


<a name="features"></a>
## Features



<a name="ack"></a>
## Acknowledgements

