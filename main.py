import torch
from torch.utils.data import random_split, ConcatDataset

from data import CloudsDataset
from train import train
from test import metrics_with_TTA
from models import base_model_resnet34
import utils

import warnings
warnings.filterwarnings("ignore")

def main():
    data = CloudsDataset(labels_csv= "./data/labels.csv", image_directory= "./data/images/")
    train_set, val_set , test_set = random_split(data,[.60,.20,.20])

    #saving split classes
    torch.save({
                'data': data,
                'train_set':train_set,
                'val_set': val_set,
                'test_set':test_set,
                },'./logs/split_classes.pt')
    
    #comparing data distributions
    data.just_labels = True
    utils.compare_train_val_test_distributions(train_set,val_set,test_set,utils.bhattacharyya_coefficient)
    data.just_labels = False

    num_epochs = 200
    patience = 15
    batch_size = 32

    model = base_model_resnet34()
    
    print('fitting model and tuning number of epochs hyperparameter')
    train(model=model,train_set=train_set,val_set=val_set,batch_size=batch_size,num_epochs=num_epochs,patience=patience)

    print('plotting epoch log')
    utils.plot_epoch_log()
    
    #loading number of epochs hyperparameter
    early_stopping_state = torch.load('./models/early_stopping_state.pt')
    early_stopping_epoch = early_stopping_state['epoch']

    #combine train and val
    combined = ConcatDataset([train_set,val_set])

    #re-training model
    print('fitting model on combined train and validation set for {} epochs'.format(early_stopping_epoch))
    model = base_model_resnet34()
    train(model=model,train_set=combined,batch_size=batch_size,num_epochs=early_stopping_epoch)

    print('testing model')
    metrics_with_TTA(model,test_set)

if __name__ == '__main__':
    main()
    