from tqdm.auto import tqdm
import torch
from torch import nn
from train_test import train_model, test_model
import sys
import os
from timeit import default_timer as timer 


def train_evaluate_model(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          device,
          epochs,
          logfile,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss() ):
    
    '''
    Trains and evaluates a neural network model over an input number of epochs.

    Params:
    - model (torch.nn.Module): neural network to be trained and tested on
    - train_dataloader (torch.utils.data.DataLoader): dataloader for fetching training data
    - test_dataloader (torch.utils.data.DataLoader): dataloader for fetching testing data
    - optimizer (torch.optim.Optimizer): optimzer to update model params
    - device: device to run code on, cuda or gpu, must be condition
    - loss_fn (torch.nn.Module): loss function used for training
    - epochs (int): The number of training epochs. (default: 5)

    Returns:
    - results (dict): A dictionary containing training and testing results.
      {
        "avg_train_loss": List of average training losses for each epoch,
        "avg_train_acc": List of average training accuracies for each epoch,
        "avg_test_loss": List of average testing losses for each epoch,
        "avg_test_acc": List of average testing accuracies for each epoch
      }
    
    '''

    #create file if it doesn't exist
    log_dir = os.path.dirname(logfile)
    os.makedirs(log_dir, exist_ok=True)
    
    #create empty results dictionary
    results = {"avg_train_loss": [],
        "avg_train_acc": [],
        "avg_test_loss": [],
        "avg_test_acc": []
    }

    #outputing model details and output to log file
    with open(logfile, "w") as log_file:
        original_stdout = sys.stdout
        sys.stdout = log_file


        # Extract details about the model
        num_parameters = sum(p.numel() for p in model.parameters())
        model_architecture = repr(model)
        model_summary = model.__str__()

        # Print details about the model
        print(f"Number of model parameters: {num_parameters}")
        print(f"Model Architecture:\n{model_architecture}")
        print()
        print(f"Model Training Results:")
        print()
       
        #start timer
        start_time = timer()
        
    
        #for every epoch, train, test, and update a neural network model
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_model(model=model,
                                              dataloader=train_dataloader,
                                              loss_fn=loss_fn,
                                              optimizer=optimizer,
                                              device=device)
            test_loss, test_acc = test_model(test_model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device)
            
            #output results of modeling
            print(
                f"Epoch: {epoch+1} | "
                f"avg_train_loss: {train_loss:.4f} | "
                f"avg_train_acc: {train_acc:.4f} | "
                f"avg_test_loss: {test_loss:.4f} | "
                f"avg_test_acc: {test_acc:.4f}"
            )

            #update results dictionary
            results["avg_train_loss"].append(train_loss)
            results["avg_train_acc"].append(train_acc)
            results["avg_test_loss"].append(test_loss)
            results["avg_test_acc"].append(test_acc)

        #end timer, output total time
        end_time = timer()
        total_training_time = end_time - start_time

        # Print total model training time to the log file
        print(f"Total model training time: {total_training_time:.3f} seconds")

        #restore standard output
        sys.stderr = original_stdout


    #return modeling results at the end of the epochs
    return results
