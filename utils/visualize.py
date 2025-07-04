"""
Visualization utilities for tissue segmentation.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
from PIL import Image
import io


def apply_color_map(mask, colors):
    """
    Apply color map to segmentation mask
    
    Args:
        mask: Segmentation mask (H, W) with class indices
        colors: List of RGB colors for each class
        
    Returns:
        colored_mask: RGB mask (H, W, 3)
    """
    # Create color map
    cmap = ListedColormap([np.array(color) / 255 for color in colors])
    
    # Apply colormap
    colored_mask = cmap(mask)
    
    # Remove alpha channel if present
    if colored_mask.shape[-1] == 4:
        colored_mask = colored_mask[..., :3]
        
    return colored_mask


def visualize_prediction(image, true_mask, pred_mask, colors, alpha=0.5):
    """
    Visualize image, ground truth and prediction
    
    Args:
        image: Input image (C, H, W) or (H, W, C)
        true_mask: Ground truth mask (H, W)
        pred_mask: Predicted mask (H, W)
        colors: List of RGB colors for each class
        alpha: Transparency for overlay
    
    Returns:
        figure: Matplotlib figure
    """
    # Convert tensors to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
        # Denormalize if needed
        if image.min() < 0:
            image = (image + 1) / 2.0
            
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()
    
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    
    # Create colored masks
    true_colored = apply_color_map(true_mask, colors)
    pred_colored = apply_color_map(pred_mask, colors)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot ground truth
    axes[1].imshow(true_colored)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Plot prediction
    axes[2].imshow(pred_colored)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def plot_to_image(figure):
    """
    Convert a Matplotlib figure to a PIL Image
    
    Args:
        figure: Matplotlib figure
        
    Returns:
        image: PIL Image
    """
    # Save figure to a buffer
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    
    # Convert to PIL Image
    image = Image.open(buf)
    return image


def create_uncertainty_map(prob_map):
    """
    Create uncertainty map from probability map
    
    Args:
        prob_map: Probability map (C, H, W)
        
    Returns:
        uncertainty_map: Uncertainty map (H, W)
    """
    if isinstance(prob_map, torch.Tensor):
        prob_map = prob_map.cpu().numpy()
        
    # Calculate uncertainty as 1 - max probability
    max_prob = np.max(prob_map, axis=0)
    uncertainty = 1 - max_prob
    
    return uncertainty
