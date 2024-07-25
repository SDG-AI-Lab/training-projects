import numpy as np
import os
import torch

from .train_one_epoch import train_one_epoch
from .validate import validate

def train(model,
          train_loader,
          val_loader,
          criterion,
          optimizer,
          num_epochs,
          best_path,
          last_path,
          device,
          scheduler
          ):
    
    current_path = os.getcwd()
    
    if not os.path.exists(os.path.join(current_path, "torch_models")):
        os.makedirs(os.path.join(current_path, "torch_models"))
    
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []  
    val_accuracy_list = [] 
    min_val_loss = np.inf
    
    model.to(device)
    
    for epoch in range(num_epochs):
        
        train_loss, train_accuracy = train_one_epoch(model, train_loader, device, criterion, optimizer, scheduler) 
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)  
        
        val_loss, val_accuracy = validate(model, val_loader, device, criterion)  
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy) 
        model.eval()

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(current_path, "torch_models", best_path)+".pt") 
            
        print("Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Min Val Loss: {:.4f}, Train Accuracy: {:.2f}%, Val Accuracy: {:.2f}%" 
          .format(epoch+1, num_epochs, train_loss, val_loss, min_val_loss, train_accuracy, val_accuracy)) 
        
        torch.save(model.state_dict(), os.path.join(current_path, "torch_models", last_path)+".pt")
        model.train()
                
    return train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list 