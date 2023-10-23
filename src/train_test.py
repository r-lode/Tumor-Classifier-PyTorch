import torch
from torch import nn

def train_model(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device):
    '''
    Trains a neural network model on the provided data using input loss function and optimizer.

    Args:
        model (torch.nn.Module): Input neural network model to be constructed using training images
        dataloader (torch.utils.data.DataLoader): dataloader to fetch training images.
        loss_fn (torch.nn.Module): input loss function
        optimizer (torch.optim.Optimizer): optimizer algorithm.
        device (str): input device, cuda or cpu, must be consistent

    Returns:
        adjusted_train_loss: average training loss per batch
        adjusted_train_acc: adjusted training accuracy per batch
    '''
    #start model in training mode
    model.train()
    
    sum_train_loss = 0
    train_accuracy = 0 

    #loop through training image batches
    for batch, (X, y) in enumerate(dataloader):
        #sending values to target device
        X = X.to(device)
        y = y.to(device)

        #perform forward pass
        model_predictions = model(X)

        #calculate loss using specified loss function
        loss = loss_fn(model_predictions, y)

        #total of training loss
        sum_train_loss += loss.item() 

        #reset gradients
        optimizer.zero_grad()

        #perform back propagation
        loss.backward()

        #update model params
        optimizer.step()

        #get predicted tumor class
        tumor_class_prediction = torch.argmax(torch.softmax(model_predictions, dim=1), dim=1)

        #update training accuracy
        train_accuracy += (tumor_class_prediction == y).sum().item() / len(model_predictions)

    #get average training loss and average training accuracy per batch
    adjusted_train_loss = sum_train_loss / len(dataloader)
    adjusted_train_acc = train_accuracy / len(dataloader)
    
    return adjusted_train_loss, adjusted_train_acc


def test_model(test_model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device):
    '''
    Evaluates a neural network model on the provided test data using the specified loss function.

    Args:
        test_model (torch.nn.Module): trained neural network model to be tested on 
        dataloader (torch.utils.data.DataLoader): dataloader to fetch testing images.
        loss_fn (torch.nn.Module): input loss function
        device (str): input device, cuda or cpu, must be consistent

    Returns:
        adjusted_test_loss: average test loss per batch
        adjusted_train_acc: average test accuracy per batch
    '''
    #start evaluation mode for testnig
    test_model.eval()

    sum_test_loss = 0
    test_accuracy = 0

    #start inference mode
    with torch.inference_mode():
        #loop through testing images in test dataloader
        for batch, (X, y) in enumerate(dataloader):
            #set X,y to input device
            X = X.to(device)
            y = y.to(device)

            #perform forward pass
            test_logits = test_model(X)

            #calculate loss
            loss = loss_fn(test_logits, y)

            #sum total test loss
            sum_test_loss += loss.item()

            #get predicted tumor label
            tumor_pred_labels = test_logits.argmax(dim=1)

            #update test accuracy
            test_accuracy += (tumor_pred_labels == y).sum().item() / len(tumor_pred_labels)

    #get average test loss and average test accuracy per batch
    adjusted_test_loss = sum_test_loss / len(dataloader)
    adjusted_test_accuracy = test_accuracy / len(dataloader)

    return adjusted_test_loss, adjusted_test_accuracy














    
