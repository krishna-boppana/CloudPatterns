import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
import datetime
import math

from data import augmentation_basic, augmentation

def train(model,train_set,batch_size,num_epochs,patience = None,val_set = None):
    '''
     Function that trains Resnet model, with modified fully connected layer, 
     tracks train set and validation set losses per epoch, and implements early stopping 

    :param model: Pre-trained pytorch model
    :param train_set: Training subset with no augmentation
    :param val_set: Validation subset with no augmentation
    :param num_epochs: Number of full passes over the training dataset
    :param patience: Number epochs to wait, without a decrease in validation loss, before stopping training (early stopping parameter)
    '''

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #tracks train and val loss per epoch. Format epoch:(train_loss,val_loss)
    epoch_log = {}
    
    #Freezing all parameters except for fully connected linear layer parameters
    for name,params in model.named_parameters():
        if not(name == 'fc.bias' or name == 'fc.weight'):
            params.requires_grad = False

    #data loaders
    train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,drop_last=True)
    if val_set:
        val_loader = DataLoader(val_set,batch_size)
    
    #optimizer
    optimizer = Adam(model.parameters(),lr = .001)
    #BCE loss function, Can account for multiple labels.
    loss_function = BCEWithLogitsLoss()
    
    epoch_range = range(1,num_epochs+1)

    if val_set:
        best_val_loss = math.inf
        patience_tracker = 0

    for epoch in epoch_range:
        if val_set:
            # Early Stopping
            if patience_tracker > patience:
                print("Stopping Training Loop: No decrease in validation loss after {} epochs".format(patience))
                return

        model.train()
        # un-freezing all parameters after 4 epochs
        if epoch == 5:
            for name,params in model.named_parameters():
                params.requires_grad = True
                # constructing new optimizer with smaller learning rate for fine tuning
            optimizer = Adam(model.parameters(),lr = .00001)
        
        train_epoch_loss = 0 
        for raw_imgs,labels in train_loader:
            #sending to gpu or device available (re-assigning b/c .to is not an inplace operation for tensors)
            raw_imgs = raw_imgs.to(device)
            labels = labels.to(device)

            imgs = augmentation(raw_imgs)
            predictions = model(imgs)

            batch_loss = loss_function(predictions,labels)
            train_epoch_loss = train_epoch_loss + batch_loss.item()

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            #print('batch completed')
        train_epoch_loss = train_epoch_loss / len(train_loader)

        if val_set:
            model.eval()
            val_epoch_loss = 0
            #computing val loss
            with torch.no_grad():
                for raw_imgs, labels in val_loader:
                    raw_imgs = raw_imgs.to(device)
                    labels = labels.to(device)
                    
                    val_imgs = augmentation_basic(raw_imgs)
                    val_predictions = model(val_imgs)
                    val_batch_loss = loss_function(val_predictions,labels)
                    val_epoch_loss = val_epoch_loss + val_batch_loss.item()
                    
                val_epoch_loss = val_epoch_loss / len(val_loader)

            epoch_log[epoch] = (train_epoch_loss, val_epoch_loss)

            #saving log every epoch (in the case the training task is interrupted)
            torch.save(epoch_log,'./logs/epoch_log.pt')
            #saving model if validation loss decreased
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer,
                    'loss' : best_val_loss
                    }, './models/early_stopping_state.pt')
                patience_tracker = 0
            else:
                patience_tracker = patience_tracker + 1
            print('{}, epoch {}, training loss {}, validation loss {}'.format(datetime.datetime.now(),epoch,train_epoch_loss,val_epoch_loss))
        else:
            epoch_log[epoch] = train_epoch_loss
            torch.save(epoch_log,'./logs/train_epoch_log.pt')
            print('{}, epoch {}, training loss {}'.format(datetime.datetime.now(),epoch,train_epoch_loss))
            
    torch.save(model.state_dict(),'./models/model_fitted.pt')