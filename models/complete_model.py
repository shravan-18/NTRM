"""
Complete Neural Tissue Relation Modeling (NTRM) model for histopathology segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_models import ResNetUNet, DecoderBlock
from models.tissue_relation import TissueRelationModule


class NTRMNet(nn.Module):
    """
    Combined model with ResNet-UNet backbone and Neural Tissue Relation Modeling
    """
    def __init__(self, n_classes=12, hidden_dim=64, gnn_layers=3, enable_global_embeddings=True):
        super(NTRMNet, self).__init__()
        
        # Base ResNet-UNet model
        self.base_model = ResNetUNet(n_classes=n_classes)
        
        # Intermediate decoder for initial segmentation
        self.initial_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # Changed from 2048 to 512
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Changed from 512 to 256 and from 256 to 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, n_classes, kernel_size=1)  # Changed from 256 to 128
        )
        
        # Tissue relation module
        self.trm = TissueRelationModule(
            in_channels=128,           # Output from decoder2 (changed from 256 to 128)
            out_channels=128,          # To match decoder2 output channels (changed from 256 to 128)
            num_classes=n_classes,
            hidden_dim=hidden_dim,
            gnn_layers=gnn_layers,
            enable_global_embeddings=enable_global_embeddings
        )
        
        # Final decoder (takes enhanced features)
        self.final_decoder3 = DecoderBlock(128, 64, 96)    # Corresponds to decoder3 in base model
        self.final_decoder4 = DecoderBlock(96, 64, 64)     # Corresponds to decoder4 in base model
        self.final_decoder5 = DecoderBlock(64, 0, 32)      # Corresponds to decoder5 in base model
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            outputs: Dictionary containing:
                - final: Final segmentation logits [B, n_classes, H, W]
                - initial: Initial segmentation logits [B, n_classes, H, W]
        """
        # Get encoder features
        x1, x2, x3, x4, x5 = self.base_model.get_encoder_features(x)
        
        # Generate initial segmentation from first two decoder blocks
        d1 = self.base_model.decoder1(x5, x4)  # [B, 256, H/16, W/16]
        d2 = self.base_model.decoder2(d1, x3)  # [B, 128, H/8, W/8]
        initial_seg = self.initial_decoder(x5)  # [B, n_classes, H/32, W/32]
        
        # Upsample initial segmentation to match d2 size
        initial_seg = F.interpolate(
            initial_seg, size=d2.shape[2:], 
            mode='bilinear', align_corners=True
        )
        
        # Apply TRM to enhance features
        enhanced_features = self.trm(d2, F.softmax(initial_seg, dim=1))
        
        # Combine with original features
        combined_features = d2 + enhanced_features
        
        # Complete decoding with enhanced features
        d3 = self.final_decoder3(combined_features, x2)
        d4 = self.final_decoder4(d3, x1)
        d5 = self.final_decoder5(d4)
        final_seg = self.final_conv(d5)
        
        # Return both initial and final segmentation
        return {
            'final': final_seg,
            'initial': F.interpolate(
                initial_seg, size=final_seg.shape[2:], 
                mode='bilinear', align_corners=True
            )
        }
