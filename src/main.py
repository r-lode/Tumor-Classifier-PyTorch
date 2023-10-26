import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import zipfile
from timeit import default_timer as timer 


from custom_tumor_dataset import Custom_Image_Dataset
from CNN import Custom_TinyVGG
from model_ops import train_evaluate_model
from calculate_shape import calculate_flattened_size
from import_local_data import load_data


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Swap these two if working on cpu instead of cuda
    #torch.manual_seed(42) 
    torch.cuda.manual_seed(42)

    #If working in container, set directories accordinly:
    CONTANER_ZIP_SOURCE = "/pytorch_classifier/data/archive.zip"
    CONTAINER_DESTINATION_DIR = "/pytorch_classifier/data/"
   
    #If not working in docker container:
    LOCAL_ZIP_SOURCE = os.path.expanduser("~/pytorch_projects/data/brain_tumor_image_data/archive.zip")
    LOCAL_DESTINATION_DIR = os.path.expanduser("~/pytorch_projects/data/")

    #GLOBAL CONSTANTS
    NUM_EPOCHS = 5
    SQUARE_IMAGE_SIZE = 200
    NUM_NEURONS = 10
    WORKING_IN_CONTAINER = 0

    if WORKING_IN_CONTAINER:
        #load data into working directory
        load_data(CONTANER_ZIP_SOURCE,CONTAINER_DESTINATION_DIR)
        #get training and test directories
        training_data_dir = "/pytorch_classifier/data/Training"
        testing_data_dir = "/pytorch_classifier/data/Testing"
        model_save_path = "/pytorch_classifier/models/model_state_dict.pth"


    else:

        #load data into working directory
        load_data(LOCAL_ZIP_SOURCE,LOCAL_DESTINATION_DIR )
        #get training and test directories
        training_data_dir = os.path.expanduser("~/pytorch_projects/data/Training/")
        testing_data_dir = os.path.expanduser("~/pytorch_projects/data/Testing/")
        model_save_path = os.path.expanduser("~/pytorch_projects/tumor_classifier/models/model1_state_dict.pth")
      

    #define an image transform to adapt image data accordingly before modeling
    data_transform = transforms.Compose([
        transforms.Resize((SQUARE_IMAGE_SIZE, SQUARE_IMAGE_SIZE)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    #create train and test datasets
    train_dataset = Custom_Image_Dataset(image_directory=training_data_dir, transform=data_transform)
    test_dataset = Custom_Image_Dataset(image_directory=testing_data_dir, transform=data_transform)

    #create a train and test data loader to fetch batch data
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

   
    #Calculate flattened size to pass to CNN
    flattened_size = calculate_flattened_size(SQUARE_IMAGE_SIZE, SQUARE_IMAGE_SIZE, num_hidden_neurons=10)

    #create an instance of model
    model_1 = Custom_TinyVGG(input_shape=3, #input num of color channels 1 for greyscale, 3 for RGB
                    num_hidden_neurons=NUM_NEURONS, 
                    flattened_shape= flattened_size,
                    output_shape=4)
    model_1.to(device)

    #loss function
    loss_fn = nn.CrossEntropyLoss()

    #set up optimizer
    optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)

    #start timer
    start_time = timer()

    #train model
    model_1_results = train_evaluate_model(model=model_1, 
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            device=device,
                            epochs=NUM_EPOCHS,
                            loss_fn=loss_fn)

    #end timer, output total time
    end_time = timer()

    print(f"Total model training time: {end_time-start_time:.3f} seconds")

    torch.save(model_1.state_dict(), model_save_path)

    