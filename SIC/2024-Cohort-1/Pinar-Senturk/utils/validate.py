import torch

def validate(model,
             val_loader,
             device,
             criterion):
    
    model.eval()
    
    with torch.no_grad():

        running_loss = 0
        correct_predictions = 0  
        total_predictions = 0    


        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            
            running_loss += val_loss

            _, predicted = torch.max(outputs, 1)  
            correct_predictions += (predicted == labels).sum().item()  
            total_predictions += labels.size(0)  

    accuracy = 100 * correct_predictions / total_predictions  


    return running_loss/len(val_loader), accuracy  