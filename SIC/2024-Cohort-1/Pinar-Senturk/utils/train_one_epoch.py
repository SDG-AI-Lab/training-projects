import torch

def train_one_epoch(model,
                    train_loader,
                    device,
                    criterion,
                    optimizer,
                    scheduler):
    
    model.train()
    
    running_loss = 0

    correct_predictions = 0
    total_predictions = 0 
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        
        outputs = model(images)
        train_loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        running_loss += train_loss.item()


        _, predicted = torch.max(outputs, 1)  
        correct_predictions += (predicted == labels).sum().item()  
        total_predictions += labels.size(0)  
        
        
    if scheduler is not None:
        scheduler.step()
        
    accuracy = 100 * correct_predictions / total_predictions  
        
    return running_loss/len(train_loader), accuracy  