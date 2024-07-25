import argparse
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data.dataloaders import dataloaders
from models.encoder_model import Encoder
from utils.seed_env import seed_env
from utils.train import train


def main(args):
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    torch.cuda.empty_cache()
    seed_env(args.seed)

    data_path = args.data
    save_best_path = args.save_best
    save_last_path = args.save_last

    batch_size = args.batch_size
    device = torch.device(args.device)
    print(device)  
    print(torch.cuda.get_device_name(0))  
    train_loader, val_loader = dataloaders()

    enc_space_dim = args.enc_space_dim
    model = Encoder(enc_space_dim).to(device)

    lr = args.lr
    num_epochs = args.epoch
    scheduler = args.scheduler
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 1.1, 2.5, 3, 1], dtype=torch.float, device=device)) 
    optimizer = torch.optim.Adam(model.parameters(), lr)

    print("Training...")

    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list = train(model,         
                                                                               train_loader, 
                                                                               val_loader, 
                                                                               criterion, 
                                                                               optimizer, 
                                                                               num_epochs, 
                                                                               save_best_path, 
                                                                               save_last_path, 
                                                                               device, 
                                                                               scheduler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data", default="dataset", help="Path to dataset")
    parser.add_argument("--save_best", type=str, default="best_model", help="Path to best model")
    parser.add_argument("--save_last", type=str, default="last_model", help="Path to last model")
    parser.add_argument("--enc_space_dim", type=int, default=5, help="Encoded space dimension for autoencoder")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--epoch", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--device", type=int, default=0, help="GPU ID")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--scheduler", type=bool, default=None, help="Enable LR scheduler")
    parser.add_argument("--seed", type=int, default=1, help="Seed for RNG")

    args = parser.parse_args()

    main(args)