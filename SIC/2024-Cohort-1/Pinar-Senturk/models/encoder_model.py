import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, enc_space_dim=5):
        super(Encoder, self).__init__()
        
        
        self.nin_block1 = self.nin_block(1, 48, kernel_size=11, strides=4, padding=0)
        self.maxpool1 = nn.MaxPool1d(3, stride=2)
        self.nin_block2 = self.nin_block(48, 128, kernel_size=5, strides=1, padding=2)
        self.maxpool2 = nn.MaxPool1d(3, stride=2)
        self.nin_block3 = self.nin_block(128, 256, kernel_size=3, strides=1, padding=1)
        self.maxpool3 = nn.MaxPool1d(3, stride=2)
        self.dropout = nn.Dropout(0.4)
        
        
        self.nin_block4 = self.nin_block(256, 5, kernel_size=3, strides=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
    
    def nin_block(self, in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, strides, padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.GELU()
        )

    def forward(self, x):
        x = self.nin_block1(x)
        x = self.maxpool1(x)
        x = self.nin_block2(x)
        x = self.maxpool2(x)
        x = self.nin_block3(x)
        x = self.maxpool3(x)
        x = self.dropout(x)
        
        
        x = self.nin_block4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        
        return x

    