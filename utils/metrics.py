"""
Evaluation metrics for segmentation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import JaccardIndex, ConfusionMatrix
from torchmetrics.segmentation import DiceScore
import matplotlib.pyplot as plt
import io
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def calculate_class_weights(masks, num_classes):
    """
    Calculate balanced class weights for a batch of masks
    Similar to the original SegmentationGen._calculateWeights method
    
    Args:
        masks: Batch of masks with shape [B, H, W]
        num_classes: Number of classes
    
    Returns:
        weights: Tensor of class weights with shape [num_classes]
    """
    device = masks.device
    masks = masks.cpu().numpy()
    
    # Count pixels per class
    class_counts = np.zeros(num_classes)
    batch_size = masks.shape[0]
    
    # Count pixels for each class across the batch
    for b in range(batch_size):
        mask = masks[b]
        for c in range(num_classes):
            class_counts[c] += np.sum(mask == c)
    
    # Adjust for absent classes
    present_classes = []
    absent_classes = []
    y = []
    
    for i in range(num_classes):
        if class_counts[i] == 0:
            absent_classes.append(i)
        else:
            present_classes.append(i)
            y.extend([i] * int(class_counts[i]))
    
    # Calculate balanced weights
    if len(present_classes) > 0:
        weights = compute_class_weight("balanced", classes=np.array(present_classes), y=np.array(y))
        # Add zeros for absent classes
        for c in absent_classes:
            weights = np.insert(weights, c, 0)
    else:
        weights = np.ones(num_classes)
    
    return torch.tensor(weights, dtype=torch.float32, device=device)


class WeightedCrossEntropyLoss(torch.nn.Module):
    """
    Weighted cross entropy loss for segmentation
    Similar to the original SegmentationGen approach
    """
    def __init__(self, num_classes, weight_mod=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_mod = weight_mod  # Optional dict to modify specific class weights
        
    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions [B, C, H, W]
            targets: Ground truth masks [B, H, W]
            
        Returns:
            loss: Weighted cross entropy loss
        """
        # Calculate class weights
        weights = calculate_class_weights(targets, self.num_classes)
        
        # Apply weight modifiers if specified
        if self.weight_mod is not None:
            for class_idx, modifier in self.weight_mod.items():
                weights[class_idx] *= modifier
        
        # Create sample weights - weights for each pixel based on its class
        B, H, W = targets.shape
        sample_weights = torch.zeros((B, H, W), device=targets.device)
        
        for c in range(self.num_classes):
            class_mask = (targets == c)
            sample_weights[class_mask] = weights[c]
        
        # Apply standard cross entropy with sample weights
        loss = F.cross_entropy(logits, targets, reduction='none')
        weighted_loss = loss * sample_weights
        
        return weighted_loss.mean()


class MetricTracker:
    """
    Tracks metrics during training and validation.
    """
    def __init__(self, num_classes, device=None):
        self.num_classes = num_classes
        self.device = device or torch.device('cpu')
        self.reset()
        
        # Initialize metrics
        self.jaccard = JaccardIndex(task="multiclass", num_classes=num_classes).to(self.device)
        self.dice = DiceScore(num_classes=num_classes, average='macro').to(self.device)
        self.conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(self.device)
        
    def reset(self):
        """Reset accumulated metrics"""
        self.losses = []
        self.iou_scores = []
        self.dice_scores = []
        self.confusion_matrices = []
        self.last_batch_metrics = {
            'loss': 0.0,
            'iou': 0.0,
            'dice': 0.0
        }
        
    def update(self, logits, masks, loss):
        """
        Update metrics with new batch
        
        Args:
            logits: Model predictions (B, C, H, W)
            masks: Ground truth masks (B, H, W)
            loss: Loss value
        """
        # Ensure metrics are on the same device as input tensors
        device = logits.device
        if self.device != device:
            self.device = device
            self.jaccard = self.jaccard.to(device)
            self.dice = self.dice.to(device)
            self.conf_matrix = self.conf_matrix.to(device)
        
        if isinstance(loss, torch.Tensor):
            loss_value = loss.item()
        else:
            loss_value = loss
            
        self.losses.append(loss_value)
            
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Calculate metrics
        iou_value = self.jaccard(preds, masks).item()
        dice_value = self.dice(
            F.one_hot(preds, num_classes=self.num_classes).permute(0, 3, 1, 2), 
            F.one_hot(masks, num_classes=self.num_classes).permute(0, 3, 1, 2)
        ).item()
        
        self.iou_scores.append(iou_value)
        self.dice_scores.append(dice_value)
        
        # Add to confusion matrix
        self.confusion_matrices.append(self.conf_matrix(preds, masks).cpu().numpy())
        
        # Store last batch metrics for progress bar updates
        self.last_batch_metrics = {
            'loss': loss_value,
            'iou': iou_value,
            'dice': dice_value
        }
            
    def get_metrics(self):
        """Get average metrics"""
        avg_loss = np.mean(self.losses) if self.losses else 0
        avg_iou = np.mean(self.iou_scores) if self.iou_scores else 0
        avg_dice = np.mean(self.dice_scores) if self.dice_scores else 0
        
        # Sum confusion matrices
        if self.confusion_matrices:
            conf_mat = np.sum(self.confusion_matrices, axis=0)
        else:
            conf_mat = np.zeros((self.num_classes, self.num_classes))
            
        return {
            'loss': avg_loss,
            'iou': avg_iou,
            'dice': avg_dice,
            'confusion_matrix': conf_mat
        }
    
    def get_last_batch_metrics(self):
        """Get metrics from the last processed batch"""
        return self.last_batch_metrics
        
    def print_metrics(self, phase='train'):
        """Print metrics to console"""
        metrics = self.get_metrics()
        print(f"{phase.capitalize()} Loss: {metrics['loss']:.4f}, IOU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}")
        
    def plot_confusion_matrix(self, class_names):
        """
        Plots and returns confusion matrix as PIL Image
        
        Args:
            class_names: List of class names
            
        Returns:
            PIL Image of the confusion matrix
        """
        metrics = self.get_metrics()
        cm = metrics['confusion_matrix']
        
        # Normalize the confusion matrix (row-wise for recall)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Normalized Confusion Matrix (Recall)')
        plt.colorbar()
        
        # Add labels
        num_classes = len(class_names)
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm_norm.max() / 2.0
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                         ha="center", va="center",
                         color="white" if cm_norm[i, j] > thresh else "black")
                
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save figure to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        # Convert to PIL Image
        image = Image.open(buf)
        return image
    