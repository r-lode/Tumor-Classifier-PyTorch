import torch
import os
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader


from custom_tumor_dataset import Custom_Image_Dataset
from CNN import Custom_TinyVGG
from model_ops import train_evaluate_model
from import_local_data import load_data



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Swap these two if working on cpu instead of cuda
    #torch.manual_seed(42) 
    torch.cuda.manual_seed(42)

    #If working in container, set directories accordinly:
    ZIP_SOURCE = "/pytorch_classifier/data/archive.zip"
    DESTINATION_DIR = "/pytorch_classifier/data/"

    training_data_dir = "/pytorch_classifier/data/Training"
    testing_data_dir = "/pytorch_classifier/data/Testing"

    #If not working in docker container:
    #ZIP_SOURCE = os.path.expanduser("~/pytorch_projects/data/brain_tumor_image_data/archive.zip")
    #DESTINATION_DIR = os.path.expanduser("~/pytorch_projects/data/")
    #load_data(ZIP_SOURCE,DESTINATION_DIR )


    # Set the directory where your training images are located
    #training_data_dir = os.path.expanduser("~/pytorch_projects/data/Training/")
    #testing_data_dir = os.path.expanduser("~/pytorch_projects/data/Testing/")

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