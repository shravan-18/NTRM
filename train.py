"""
Main training script for Neural Tissue Relation Modeling.
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from tensorboardX import SummaryWriter

import config
from config import *
from utils.dataset import create_data_loaders
from utils.metrics import MetricTracker, WeightedCrossEntropyLoss
from utils.visualize import visualize_prediction, plot_to_image, create_uncertainty_map
from models.complete_model import NTRMNet


def parse_args():
    parser = argparse.ArgumentParser(description='Train NTRM model')

    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for data')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM, help='Hidden dimension for TRM')
    parser.add_argument('--gnn_layers', type=int, default=GNN_LAYERS, help='Number of GNN layers')
    parser.add_argument('--patch_size', type=int, default=PATCH_SIZE, help='Patch size')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device to use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    return parser.parse_args()


def train_epoch(model, loader, optimizer, criterion, device, tracker, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    tracker.reset()
    
    # Create progress bar - only show one tqdm per epoch
    pbar = tqdm(loader, total=len(loader), ncols=150, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", 
                leave=False, position=0)
    
    for batch_idx, batch in enumerate(pbar):
        # Get data
        images = batch['image'].to(device)
        masks = batch['mask'].to(device).long() 
        # print("Images shape: ", images.shape)
        # print("Masks shape: ", masks.shape)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        logits = outputs['final']
        
        # Calculate loss
        loss = criterion(logits, masks)
        
        # Add loss from initial segmentation if available
        if 'initial' in outputs:
            initial_loss = criterion(outputs['initial'], masks)
            loss += 0.5 * initial_loss  # Weight initial segmentation less
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        tracker.update(logits, masks, loss)
        
        # Update progress bar with current batch metrics
        batch_metrics = tracker.get_last_batch_metrics()
        pbar.set_postfix({
            'loss': f"{batch_metrics['loss']:.4f}",
            'iou': f"{batch_metrics['iou']:.4f}",
            'dice': f"{batch_metrics['dice']:.4f}"
        })
    
    # Return metrics
    return tracker.get_metrics()


def validate(model, loader, criterion, device, tracker, epoch, total_epochs):
    """Validate model"""
    model.eval()
    tracker.reset()
    
    # Create progress bar - only show one tqdm per validation
    pbar = tqdm(loader, total=len(loader), ncols=150, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", 
                leave=False, position=0)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Get data
            images = batch['image'].to(device)
            masks = batch['mask'].to(device).long() 
            
            # Forward pass
            outputs = model(images)
            logits = outputs['final']
            
            # Calculate loss
            loss = criterion(logits, masks)
            
            # Update metrics
            tracker.update(logits, masks, loss)
            
            # Update progress bar with current batch metrics
            batch_metrics = tracker.get_last_batch_metrics()
            pbar.set_postfix({
                'loss': f"{batch_metrics['loss']:.4f}",
                'iou': f"{batch_metrics['iou']:.4f}",
                'dice': f"{batch_metrics['dice']:.4f}"
            })
    
    # Return metrics
    return tracker.get_metrics()


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, checkpoint_path)


def visualize_batch(model, loader, device, writer, epoch, class_names, class_colors):
    """Visualize predictions for a single batch"""
    model.eval()
    
    # Get a single batch
    batch = next(iter(loader))
    images = batch['image'].to(device)
    masks = batch['mask'].to(device)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(images)
        logits = outputs['final']
        
        # Get predictions
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Create visualizations
        for i in range(min(4, images.shape[0])):  # Visualize up to 4 images
            # Denormalize image
            image = images[i].cpu().numpy().transpose(1, 2, 0)
            image = (image + 1) / 2.0  # Assuming [-1, 1] normalization
            
            # Create figure
            fig = visualize_prediction(
                image, 
                masks[i].cpu().numpy(), 
                preds[i].cpu().numpy(),
                class_colors
            )
            
            # Convert to image and log
            img = plot_to_image(fig)
            writer.add_image(f'predictions/sample_{i}', np.array(img).transpose(2, 0, 1), epoch)
            
            # Create uncertainty map
            uncertainty = create_uncertainty_map(probs[i].cpu().numpy())
            writer.add_image(f'uncertainty/sample_{i}', uncertainty[None, :, :], epoch)


def main():
    # Parse arguments
    args = parse_args()
    
    # Update DATA_ROOT and derived paths in config module
    config_module = sys.modules['config']
    
    # Update DATA_ROOT
    config_module.DATA_ROOT = Path(args.data_root)
    
    # Update all derived paths
    config_module.TRAIN_IMG_DIR = config_module.DATA_ROOT / "X_train"
    config_module.TRAIN_MASK_DIR = config_module.DATA_ROOT / "y_train"
    config_module.VAL_IMG_DIR = config_module.DATA_ROOT / "X_val"
    config_module.VAL_MASK_DIR = config_module.DATA_ROOT / "y_val"
    config_module.TEST_IMG_DIR = config_module.DATA_ROOT / "X_test"
    config_module.TEST_MASK_DIR = config_module.DATA_ROOT / "y_test"
    
    print(f"Data root set to: {config_module.DATA_ROOT}")
    
    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args.batch_size, args.patch_size)
    print(f"Created data loaders: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")
    
    # Create model
    model = NTRMNet(
        n_classes=NUM_CLASSES, 
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        enable_global_embeddings=ENABLE_GLOBAL_TISSUE_EMBEDDINGS
    )
    model.to(device)
    print(f"Created model: {model.__class__.__name__}")
    
    # Define loss function and optimizer

    # Optional: Define weight modifiers for specific classes
    # For example, to boost the weight of follicles (FOL) and inflammation (INF)
    # weight_mod = {
    #     CLASS_NAMES.index('FOL'): 1.2,  # Increase weight by 20%
    #     CLASS_NAMES.index('INF'): 1.2   # Increase weight by 20%
    # }

    criterion = WeightedCrossEntropyLoss(NUM_CLASSES, weight_mod=None) # weight_mod can be added if needed
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=args.verbose)
    
    # Initialize trackers
    train_tracker = MetricTracker(NUM_CLASSES, device)
    val_tracker = MetricTracker(NUM_CLASSES, device)
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=str(LOG_DIR / time.strftime("%Y%m%d-%H%M%S")))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print(f"Loading checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.checkpoint}")
    
    # Training loop
    best_val_iou = 0.0
    
    print(f"Starting training for {args.epochs} epochs")
    
    # Use only one progress bar for overall progress
    progress_bar = tqdm(range(start_epoch, args.epochs), total=args.epochs, initial=start_epoch, 
                         ncols=150, desc="Overall Progress", position=1)
    
    for epoch in progress_bar:
        # Train
        start_time = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, train_tracker, epoch, args.epochs)
        train_time = time.time() - start_time
        
        # Validate
        start_time = time.time()
        val_metrics = validate(model, val_loader, criterion, device, val_tracker, epoch, args.epochs)
        val_time = time.time() - start_time
        
        # Print metrics if verbose
        if args.verbose:
            metrics_str = (
                f"Epoch {epoch+1}/{args.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, "
                f"Dice: {train_metrics['dice']:.4f}, Time: {train_time:.2f}s | "
                f"Val Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, "
                f"Dice: {val_metrics['dice']:.4f}, Time: {val_time:.2f}s"
            )
            print(metrics_str)
        
        # Update overall progress bar
        progress_bar.set_postfix({
            'Val IoU': f"{val_metrics['iou']:.4f}", 
            'Val Dice': f"{val_metrics['dice']:.4f}",
            'Train Loss': f"{train_metrics['loss']:.4f}"
        })
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('IoU/train', train_metrics['iou'], epoch)
        writer.add_scalar('Dice/train', train_metrics['dice'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('IoU/val', val_metrics['iou'], epoch)
        writer.add_scalar('Dice/val', val_metrics['dice'], epoch)
        
        # Log confusion matrix
        train_cm_img = train_tracker.plot_confusion_matrix(CLASS_NAMES)
        val_cm_img = val_tracker.plot_confusion_matrix(CLASS_NAMES)
        writer.add_image('ConfusionMatrix/train', np.array(train_cm_img).transpose(2, 0, 1), epoch)
        writer.add_image('ConfusionMatrix/val', np.array(val_cm_img).transpose(2, 0, 1), epoch)
        
        # Visualize predictions
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            visualize_batch(model, val_loader, device, writer, epoch, CLASS_NAMES, CLASS_COLORS)
        
        # Save checkpoint
        checkpoint_path = CHECKPOINT_DIR / f"ntrm_epoch_{epoch:03d}.pth"
        save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)
        
        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            best_model_path = CHECKPOINT_DIR / "ntrm_best_model.pth"
            save_checkpoint(model, optimizer, epoch, val_metrics, best_model_path)
            if args.verbose:
                print(f"Saved best model with IoU: {best_val_iou:.4f}")
    
    # Close TensorBoard writer
    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
