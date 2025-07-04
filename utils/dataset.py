"""
Dataset and dataloader utilities for tissue segmentation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TissueSegmentationDataset(Dataset):
    """
    PyTorch Dataset for histopathology image segmentation.
    """
    def __init__(self, img_dir, mask_dir, transform=None, classes=None, colors=None):
        """
        Args:
            img_dir (str): Directory with input images
            mask_dir (str): Directory with mask images
            transform: Albumentations transforms
            classes (list): List of class names 
            colors (list): List of RGB values for each class
        """
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        # Get all image files
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        
        # Set up color mapping
        if classes is None or colors is None:
            from config import CLASS_NAMES, CLASS_COLORS
            self.classes = CLASS_NAMES
            self.colors = CLASS_COLORS
        else:
            self.classes = classes
            self.colors = colors
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Get file name
        img_name = self.img_files[idx]
        
        # Load image and mask
        img_path = self.img_dir / img_name
        mask_path = self.mask_dir / img_name
        
        # Read images
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Convert mask from RGB to class indices
        seg_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        
        for class_idx, color in enumerate(self.colors):
            class_mask = np.all(mask == color, axis=2)
            seg_mask[class_mask] = class_idx
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=seg_mask)
            image = augmented['image']
            seg_mask = augmented['mask']
        else:
            # Basic preprocessing
            transform = A.Compose([
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])
            augmented = transform(image=image, mask=seg_mask)
            image = augmented['image']
            seg_mask = torch.from_numpy(seg_mask).long()
        
        return {
            'image': image,
            'mask': seg_mask,
            'filename': img_name
        }


def get_train_transform(patch_size=512):
    """Returns training transformations"""
    return A.Compose([
        A.Resize(height=patch_size, width=patch_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

def get_val_transform(patch_size=512):
    """Returns validation transformations"""
    return A.Compose([
        A.Resize(height=patch_size, width=patch_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

def create_data_loaders(batch_size=4, patch_size=512):
    """
    Creates training and validation data loaders.
    
    Returns:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
    """
    from config import (
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        VAL_IMG_DIR, VAL_MASK_DIR,
        CLASS_NAMES, CLASS_COLORS
    )
    
    # Create datasets
    train_dataset = TissueSegmentationDataset(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        transform=get_train_transform(patch_size),
        classes=CLASS_NAMES,
        colors=CLASS_COLORS
    )
    
    val_dataset = TissueSegmentationDataset(
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        transform=get_val_transform(patch_size),
        classes=CLASS_NAMES,
        colors=CLASS_COLORS
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader
