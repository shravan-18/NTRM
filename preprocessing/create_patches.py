"""
Image Patch Generator
---------------------
Creates training patches from histopathological images and corresponding masks.
Supports non-overlapping and overlapping tiling strategies.

Usage:
    python create_patches.py --dir ./data/ --dim 512 [--overlap]
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create training set of patches from images")
    parser.add_argument("--dir", type=str, default="./data/", 
                        help="Path to data directory")
    parser.add_argument("--dim", type=int, default=512, 
                        help="Patch size (default: 512)")
    parser.add_argument("--overlap", dest="overlap", action="store_true", 
                        help="Enable overlapping tiles")
    parser.set_defaults(overlap=False)
    return parser.parse_args()


def setup_directories(base_dir: str, dim: int, overlap: bool) -> Tuple[str, str, str]:
    """Setup input/output directories."""
    # Create output folder name based on parameters
    patch_dir_name = f"Patches_{'Overlapped' if overlap else ''}_{dim}"
    patch_dir = os.path.join(base_dir, patch_dir_name)
    
    # Create output directory if it doesn't exist
    Path(patch_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup I/O directories
    image_in = os.path.join(base_dir, "Images")
    mask_in = os.path.join(base_dir, "Masks")
    
    return patch_dir, image_in, mask_in


def prepare_image_folders(patch_dir: str, image_name: str) -> str:
    """Create image-specific output folders."""
    # Strip extension to get base filename
    fname = image_name.split(".")[0]
    
    # Create folder structure
    folder = os.path.join(patch_dir, fname)
    Path(folder).mkdir(parents=True, exist_ok=True)
    
    # Create X and y subdirectories
    for sub_folder in ["X", "y"]:
        Path(os.path.join(folder, sub_folder)).mkdir(parents=True, exist_ok=True)
    
    return fname, folder


def process_overlapping_tiles(image: np.ndarray, mask: np.ndarray, 
                             dim: int, folder: str, fname: str) -> int:
    """Process image using overlapping tiling strategy."""
    h, w = image.shape[0], image.shape[1]
    
    # Compute steps and overlap amounts
    w_steps = w // dim
    w_overlap = (dim - (w % dim)) // w_steps if w_steps > 0 else 0
    h_steps = h // dim
    h_overlap = (dim - (h % dim)) // h_steps if h_steps > 0 else 0
    
    # Initial positions
    count = 1
    h_x, h_y = 0, dim
    
    # Loop through all tile positions
    for i in range(h_steps + 1):
        w_x, w_y = 0, dim
        for j in range(w_steps + 1):
            # Extract patches
            image_patch = image[h_x:h_y, w_x:w_y, :]
            mask_patch = mask[h_x:h_y, w_x:w_y, :]
            
            # Skip if patch dimensions don't match target or no mask content
            if (image_patch.shape[0] < dim or 
                image_patch.shape[1] < dim or 
                np.sum(mask_patch) == 0):
                w_x += dim - w_overlap
                w_y += dim - w_overlap
                continue
                
            # Generate filenames
            count = save_patches(image_patch, mask_patch, folder, fname, count)
            
            # Update column positions
            w_x += dim - w_overlap
            w_y += dim - w_overlap
        
        # Update row positions
        h_x += dim - h_overlap
        h_y += dim - h_overlap
    
    return count


def process_non_overlapping_tiles(image: np.ndarray, mask: np.ndarray, 
                                 dim: int, folder: str, fname: str) -> int:
    """Process image using non-overlapping tiling strategy."""
    # Calculate number of steps to tile image
    row_steps = (image.shape[0] // dim) + 1
    col_steps = (image.shape[1] // dim) + 1
    
    count = 1
    row_pos = 0
    
    # Loop through all tile positions
    for r in range(row_steps):
        col_pos = 0
        for c in range(col_steps):
            # Extract patches
            image_patch = image[row_pos:row_pos+dim, col_pos:col_pos+dim, :]
            mask_patch = mask[row_pos:row_pos+dim, col_pos:col_pos+dim, :]
            
            # Skip if patch dimensions don't match target or no mask content
            if (image_patch.shape[0] < dim or 
                image_patch.shape[1] < dim or 
                np.sum(mask_patch) == 0):
                col_pos += dim
                continue
                
            # Generate filenames
            count = save_patches(image_patch, mask_patch, folder, fname, count)
            
            # Move to next column
            col_pos += dim
            
        # Move to next row
        row_pos += dim
    
    return count


def save_patches(image_patch: np.ndarray, mask_patch: np.ndarray, 
                folder: str, fname: str, count: int) -> int:
    """Save image and mask patches to disk."""
    # Generate filenames
    image_name = f"X/{fname}_{count:04d}.png"
    image_path = os.path.join(folder, image_name)
    
    mask_name = f"y/{fname}_{count:04d}.png"
    mask_path = os.path.join(folder, mask_name)
    
    print(f"Saving... {image_path} {mask_path}")
    
    # Write files to disk
    cv2.imwrite(image_path, image_patch)
    cv2.imwrite(mask_path, mask_patch)
    
    return count + 1


def main():
    """Main function to orchestrate patch generation."""
    # Parse arguments
    args = parse_args()
    
    # Setup directories
    patch_dir, image_in, mask_in = setup_directories(args.dir, args.dim, args.overlap)
    
    # Get files in dataset
    files = os.listdir(image_in)
    
    # Process each file
    for idx, file in enumerate(files, 1):
        print(f"Processing file {idx} of {len(files)}")
        
        # Create output folder structure
        fname, folder = prepare_image_folders(patch_dir, file)
        
        # Load image and mask
        image = cv2.imread(os.path.join(image_in, f"{fname}.tif"))
        mask = cv2.imread(os.path.join(mask_in, f"{fname}.png"))
        
        if image is None or mask is None:
            print(f"Error loading images for {fname}. Skipping.")
            continue
            
        # Process tiles based on strategy
        if args.overlap:
            process_overlapping_tiles(image, mask, args.dim, folder, fname)
        else:
            process_non_overlapping_tiles(image, mask, args.dim, folder, fname)
    
    print("Done.")


if __name__ == "__main__":
    main()
