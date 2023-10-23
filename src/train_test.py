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

        return train_loss, train_accuracy








    
