"""
Evaluation script for Neural Tissue Relation Modeling (NTRM) model.

Usage:
    python evaluate.py --data_root ./data --checkpoint checkpoints/ntrm_best_model.pth
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score, accuracy_score, confusion_matrix

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from config import *
from models.complete_model import NTRMNet
from utils.dataset import TissueSegmentationDataset, get_val_transform
from utils.visualize import visualize_prediction, create_uncertainty_map, apply_color_map


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate NTRM model')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for data')
    parser.add_argument('--model_type', type=str, default='ntrm', choices=['ntrm', 'base', 'vggUnet', 'attUnet', 'deeplabv3'], 
                    help='Model type to evaluate')
    parser.add_argument('--use_test', action='store_true', help='Use test loader for results')
    parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM, help='Hidden dimension for TRM')
    parser.add_argument('--gnn_layers', type=int, default=GNN_LAYERS, help='Number of GNN layers')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/ntrm_best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--visualize_samples', type=int, default=10, help='Number of samples to visualize')
    return parser.parse_args()


def create_test_dataloader(data_root, batch_size, patch_size):
    """Create test data loader"""
    test_img_dir = Path(data_root) / "X_test"
    test_mask_dir = Path(data_root) / "y_test"
    
    test_dataset = TissueSegmentationDataset(
        test_img_dir,
        test_mask_dir,
        transform=get_val_transform(patch_size),
        classes=CLASS_NAMES,
        colors=CLASS_COLORS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader


def create_val_dataloader(data_root, batch_size, patch_size):
    """Create test data loader"""
    test_img_dir = Path(data_root) / "X_val"
    test_mask_dir = Path(data_root) / "y_val"
    
    test_dataset = TissueSegmentationDataset(
        test_img_dir,
        test_mask_dir,
        transform=get_val_transform(patch_size),
        classes=CLASS_NAMES,
        colors=CLASS_COLORS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader


def load_model(checkpoint_path, device, hidden_dim, gnn_layers, model_type='ntrm'):
    """Load model from checkpoint"""
    if model_type == 'ntrm':
        model = NTRMNet(
            n_classes=NUM_CLASSES, 
            hidden_dim=hidden_dim,
            gnn_layers=gnn_layers,
            enable_global_embeddings=ENABLE_GLOBAL_TISSUE_EMBEDDINGS
        )
    elif model_type == 'base':
        from models.base_paper_model import create_base_paper_model
        model = create_base_paper_model(n_classes=NUM_CLASSES, pretrained=False)
    elif model_type == 'vggUnet':
        from models.VGG_UNet import create_vgg_unet_model
        model = create_vgg_unet_model(n_classes=NUM_CLASSES, pretrained=False)
    elif model_type == 'attUnet':
        from models.Attention_UNet import create_attention_unet_model
        model = create_attention_unet_model(n_classes=NUM_CLASSES, pretrained=False)
    elif model_type == 'deeplabv3':
        from models.deeplab_v3 import create_deeplabv3plus_model
        model = create_deeplabv3plus_model(n_classes=NUM_CLASSES, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def calculate_class_metrics(true_masks, pred_masks, num_classes):
    """Calculate per-class metrics"""
    print("Computing metrics across all test images...")
    
    # Flatten masks for metric calculation
    true_flat = true_masks.reshape(-1)
    pred_flat = pred_masks.reshape(-1)
    
    # Calculate metrics
    accuracy = accuracy_score(true_flat, pred_flat)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Per-class metrics
    class_accuracy = {}
    class_iou = {}
    class_f1 = {}
    class_precision = {}
    class_recall = {}
    class_support = {}
    
    # Calculate metrics per class
    print("Computing per-class metrics...")
    for c in tqdm(range(num_classes), desc="Classes", ncols=150):
        # Create binary masks for this class
        true_class = (true_flat == c)
        pred_class = (pred_flat == c)
        
        # Skip if class not present
        if np.sum(true_class) == 0:
            continue
            
        # Calculate metrics
        class_accuracy[c] = accuracy_score(true_class, pred_class)
        class_iou[c] = jaccard_score(true_class, pred_class, zero_division=0)
        class_f1[c] = f1_score(true_class, pred_class, zero_division=0)
        
        # Calculate precision and recall
        tp = np.sum(np.logical_and(pred_class, true_class))
        fp = np.sum(np.logical_and(pred_class, np.logical_not(true_class)))
        fn = np.sum(np.logical_and(np.logical_not(pred_class), true_class))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        class_precision[c] = precision
        class_recall[c] = recall
        class_support[c] = np.sum(true_class)
    
    # Calculate mean metrics (weighted by support)
    total_support = sum(class_support.values())
    mean_iou = sum(class_iou[c] * class_support[c] for c in class_support) / total_support
    mean_f1 = sum(class_f1[c] * class_support[c] for c in class_support) / total_support
    
    # Calculate Dice coefficient (equivalent to F1 score for binary classification)
    # For multiclass, we use macro-averaging
    dice = mean_f1
    
    print(f"Mean IoU: {mean_iou:.4f}, Mean F1/Dice: {mean_f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'class_iou': class_iou,
        'class_f1': class_f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_support': class_support,
        'mean_iou': mean_iou,
        'mean_f1': mean_f1,
        'dice': dice
    }


def print_metrics(metrics, class_names):
    """Print metrics in a formatted way"""
    print("\n" + "="*80)
    print(f"{'EVALUATION METRICS':^80}")
    print("="*80)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Mean IoU:  {metrics['mean_iou']:.4f}")
    print(f"  Mean F1:   {metrics['mean_f1']:.4f}")
    print(f"  Dice:      {metrics['dice']:.4f}")
    
    print("\nPer-Class Metrics:")
    headers = ["Class", "Accuracy", "IoU", "F1", "Precision", "Recall", "Support"]
    print(f"  {headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10} {headers[5]:<10} {headers[6]:<10}")
    print("  " + "-"*75)
    
    for c in sorted(metrics['class_accuracy'].keys()):
        class_name = class_names[c]
        acc = metrics['class_accuracy'][c]
        iou = metrics['class_iou'][c]
        f1 = metrics['class_f1'][c]
        prec = metrics['class_precision'][c]
        rec = metrics['class_recall'][c]
        support = metrics['class_support'][c]
        
        print(f"  {class_name:<15} {acc:<10.4f} {iou:<10.4f} {f1:<10.4f} {prec:<10.4f} {rec:<10.4f} {support:<10}")
    
    print("\n" + "="*80)


def save_confusion_matrix(true_masks, pred_masks, class_names, output_dir):
    """Create and save confusion matrix visualization"""
    print("Generating confusion matrix...")
    
    # Flatten masks
    true_flat = true_masks.reshape(-1)
    pred_flat = pred_masks.reshape(-1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_flat, pred_flat, labels=range(len(class_names)))
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
    
    # Plot
    plt.figure(figsize=(12, 10))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix (Recall)')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.0
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, f"{cm_normalized[i, j]:.2f}",
                     ha="center", va="center",
                     color="white" if cm_normalized[i, j] > thresh else "black")
                    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save to file
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")


def save_visualizations(model, dataloader, device, output_dir, num_samples, class_names, class_colors):
    """Generate and save visualizations of model predictions"""
    print(f"Generating visualizations for {num_samples} samples...")
    
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'uncertainty_maps'), exist_ok=True)
    
    # Get samples to visualize
    with torch.no_grad():
        all_samples = []
        for batch in tqdm(dataloader, desc="Collecting samples", ncols=150, leave=True):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            filenames = batch['filename']
            
            # Forward pass
            outputs = model(images)
            probs = F.softmax(outputs['final'], dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Create uncertainty maps
            uncertainty_maps = [create_uncertainty_map(prob.cpu().numpy()) for prob in probs]
            
            # Add to samples
            for i in range(images.shape[0]):
                all_samples.append({
                    'image': images[i].cpu(),
                    'mask': masks[i].cpu(),
                    'pred': preds[i].cpu(),
                    'filename': filenames[i],
                    'uncertainty': uncertainty_maps[i]
                })
                
            # Break if we have enough samples
            if len(all_samples) >= num_samples:
                break
    
    # Visualize samples
    print(f"Creating and saving visualization images...")
    for i, sample in enumerate(tqdm(all_samples[:num_samples], desc="Creating visualizations", ncols=150)):
        # Visualize prediction
        fig = visualize_prediction(
            sample['image'], 
            sample['mask'], 
            sample['pred'],
            class_colors
        )
        
        # Save figure
        filename = sample['filename'].split('.')[0]
        vis_path = os.path.join(output_dir, 'visualizations', f"{filename}_vis.png")
        fig.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save uncertainty map
        plt.figure(figsize=(10, 8))
        plt.imshow(sample['uncertainty'], cmap='hot')
        plt.colorbar(label='Uncertainty (1 - max probability)')
        plt.title(f'Uncertainty Map: {filename}')
        plt.axis('off')
        
        unc_path = os.path.join(output_dir, 'uncertainty_maps', f"{filename}_uncertainty.png")
        plt.savefig(unc_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {num_samples} visualizations to {output_dir}/visualizations/")
    print(f"Saved {num_samples} uncertainty maps to {output_dir}/uncertainty_maps/")


def evaluate_model(args):
    """Main evaluation function"""
    # Setup
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create test dataloader
    if args.use_test:
        test_loader = create_test_dataloader(args.data_root, args.batch_size, args.patch_size)
        print(f"Created test dataloader with {len(test_loader.dataset)} samples")
    else:
        test_loader = create_val_dataloader(args.data_root, args.batch_size, args.patch_size)
        print(f"Created val dataloader with {len(test_loader.dataset)} samples")
    
    # Load model
    model = load_model(args.checkpoint, device, args.hidden_dim, args.gnn_layers, args.model_type)
    print(f"Loaded model from {args.checkpoint}")
    
    # Evaluate model
    all_true_masks = []
    all_pred_masks = []
    
    print(f"Running inference on {len(test_loader.dataset)} test images...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running inference", ncols=150, leave=True):
            # Get data
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            logits = outputs['final']
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Add to lists
            all_true_masks.append(masks.cpu().numpy())
            all_pred_masks.append(preds.cpu().numpy())
    
    # Concatenate all masks
    print("Processing results...")
    all_true_masks = np.concatenate(all_true_masks, axis=0)
    all_pred_masks = np.concatenate(all_pred_masks, axis=0)
    
    # Calculate metrics
    metrics = calculate_class_metrics(all_true_masks, all_pred_masks, NUM_CLASSES)
    
    # Print metrics
    print_metrics(metrics, CLASS_NAMES)
    
    # Save metrics to file
    print("Saving metrics to file...")
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Test dataset: {args.data_root}/X_test\n\n")
        
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"  Mean IoU:  {metrics['mean_iou']:.4f}\n")
        f.write(f"  Mean F1:   {metrics['mean_f1']:.4f}\n")
        f.write(f"  Dice:      {metrics['dice']:.4f}\n\n")
        
        f.write("Per-Class Metrics:\n")
        headers = ["Class", "Accuracy", "IoU", "F1", "Precision", "Recall", "Support"]
        f.write(f"  {headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10} {headers[5]:<10} {headers[6]:<10}\n")
        f.write("  " + "-"*75 + "\n")
        
        for c in sorted(metrics['class_accuracy'].keys()):
            class_name = CLASS_NAMES[c]
            acc = metrics['class_accuracy'][c]
            iou = metrics['class_iou'][c]
            f1 = metrics['class_f1'][c]
            prec = metrics['class_precision'][c]
            rec = metrics['class_recall'][c]
            support = metrics['class_support'][c]
            
            f.write(f"  {class_name:<15} {acc:<10.4f} {iou:<10.4f} {f1:<10.4f} {prec:<10.4f} {rec:<10.4f} {support:<10}\n")
    
    print(f"Metrics saved to {metrics_path}")
    
    # Save confusion matrix
    save_confusion_matrix(all_true_masks, all_pred_masks, CLASS_NAMES, args.output_dir)
    
    # Save visualizations
    save_visualizations(model, test_loader, device, args.output_dir, args.visualize_samples, CLASS_NAMES, CLASS_COLORS)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)
    