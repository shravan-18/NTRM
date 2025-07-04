"""
Base CNN models for segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderBlock(nn.Module):
    """
    Decoder block for U-Net architecture
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ResNetUNet(nn.Module):
    """
    U-Net with ResNet18 encoder (reduced from ResNet50)
    """
    def __init__(self, n_classes=12, pretrained=True):
        super(ResNetUNet, self).__init__()
        
        # Load pre-trained ResNet18 (reduced from ResNet50)
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Extract encoder layers
        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )  # 64 channels
        
        self.encoder2 = self.resnet.layer1  # 64 channels
        self.encoder3 = self.resnet.layer2  # 128 channels
        self.encoder4 = self.resnet.layer3  # 256 channels
        self.encoder5 = self.resnet.layer4  # 512 channels
        
        self.pool = self.resnet.maxpool
        
        # Decoder (reduced filter counts)
        self.decoder1 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder3 = DecoderBlock(128, 64, 96)
        self.decoder4 = DecoderBlock(96, 64, 64)
        self.decoder5 = DecoderBlock(64, 0, 32)
        
        # Final classification layer
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)
        
    def get_encoder_features(self, x):
        """Get intermediate features from encoder"""
        x1 = self.encoder1(x)  # 64 channels
        x2 = self.pool(x1)
        x2 = self.encoder2(x2)  # 64 channels
        x3 = self.encoder3(x2)  # 128 channels
        x4 = self.encoder4(x3)  # 256 channels
        x5 = self.encoder5(x4)  # 512 channels
        
        return x1, x2, x3, x4, x5
        
    def forward(self, x):
        # Encoder
        x1, x2, x3, x4, x5 = self.get_encoder_features(x)
        
        # Decoder with skip connections
        d1 = self.decoder1(x5, x4)
        d2 = self.decoder2(d1, x3)
        d3 = self.decoder3(d2, x2)
        d4 = self.decoder4(d3, x1)
        d5 = self.decoder5(d4)
        
        # Final classification
        output = self.final(d5)
        
        return output
