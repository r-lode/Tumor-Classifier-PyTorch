import torch
from torch import nn

def train_model(input_model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device):
    '''
    
    
    '''
    
    
    input_model.train()

    train_loss = 0
    train_accuracy = 0

    for batch, (X,y) in enumerate(dataloader):

        X = X.to(device)
        y = y.to(device)

        #forward pass
        y_predictions = input_model(X)

        #calculate loss
        loss = loss_fn(y_predictions, y)
        sum_train_loss = loss.item()

        #zero grad optimizer
        optimizer.zero_grad()

        #loss backward
        loss.backward()

        #optimizeer step
        optimizer.step()

        #calculate predicted class
        predicted_tumor_class = torch.argmax(torch.softmax(y_predictions, dim = 1), dim = 1)

        #update train accuracy
        train_accuracy += (predicted_tumor_class == y).sum().item()/len(y_predictions)

    adjusted_train_loss = train_loss / len(dataloader)
    adjusted_accuracy = train_accuracy / len(dataloader)
    return adjusted_train_loss, adjusted_accuracy


def test_model(test_model: torch.nn.Module,
               dataloader: torch.utils.DataLoader,
               loss_fn: torch.nn.Module,
               device):
    
    #start evaluation mode
    test_model.eval()

    sum_test_loss = 0
    test_accuracy = 0

    #start inference mode

    with torch.inference_mode():

        for batch, (X, y) in enumerate(dataloader):

            X = X.to(device)
            y = y.to(device)

            #forward pass
            test_logits = test_model(X)

            #loss
            loss = loss_fn(test_logits, y)
            sum_test_loss += loss.item()

            #calculate predicted test tumor label
            tumor_pred_labels = test_logits.argmax(dim = 1)
            test_accuracy += ((tumor_pred_labels == y).sum().item()/len(tumor_pred_labels))



    adjusted_test_loss = sum_test_loss / len(dataloader)
    adjusted_test_accuracy = test_accuracy / len(dataloader)

    return adjusted_test_loss, adjusted_test_accuracy
 














    
