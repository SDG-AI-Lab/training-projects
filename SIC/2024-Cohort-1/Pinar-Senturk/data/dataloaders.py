import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

def dataloaders():

    train_file_path = os.path.join(os.getcwd(), 'dataset', 'mitbih_train.csv')
    test_file_path = os.path.join(os.getcwd(), 'dataset', 'mitbih_test.csv')

    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    y_df = train_df.iloc[:, -1]
    y_df = y_df.astype('int')
    x_df = train_df.iloc[:, :-1]
    x_df = x_df.astype('float')

    class_names = {0:'Normal beat',
           1:'Supraventricular premature beat',
           2:'Premature ventricular contraction',
           3:'Fusion of ventricular and normal beat',
           4:'Unclassifiable beat'}
    
    x_np = x_df.to_numpy()
    x_np = x_np.reshape(-1, 1, 187)

    y_np = y_df.to_numpy()
    y_np = y_np.reshape(-1, 1)

    x_mean = x_np.mean()
    x_std = x_np.std()
    x_max = x_np.max()
    x_min = x_np.min()

    x_norm = x_np - x_mean
    x_norm *= 1/x_std

    np.savetxt('x_norm.csv', x_norm.squeeze(), delimiter=',')
    np.savetxt('y_np.csv', y_np, delimiter=',')

    x_norm = np.loadtxt('x_norm.csv', delimiter=',').reshape(-1, 1, 187)
    y_np = np.loadtxt('y_np.csv', delimiter=',').reshape(-1, 1)

    y_np = y_np.astype('int')
    x_train, x_val, y_train, y_val = train_test_split(x_norm, y_np, test_size=0.2, shuffle=True)

    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.squeeze(torch.from_numpy(y_train)).long())
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(x_val).float(), torch.squeeze(torch.from_numpy(y_val)).long())

    class_dist = np.unique(y_train, return_counts=True)[1] 
    class_dist = class_dist/np.sum(class_dist) 
    weights = 1/class_dist 

    data_weights = np.zeros(y_train.shape[0]) 
    for i in range(y_train.shape[0]): 
        data_weights[i] = weights[y_train[i, 0]]

    hyper_params = {'bs': 1024,
                                'lr': 1e-3,
                                'lr_decay': 0.3,
                                'epochs':50,
                                }
    
    train_loader = DataLoader(train_ds, batch_size=hyper_params['bs'], sampler=WeightedRandomSampler(weights=data_weights, num_samples= len(y_train), replacement=True))
    val_loader = DataLoader(val_ds, batch_size=hyper_params['bs'])
    
    return train_loader, val_loader 