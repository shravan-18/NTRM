"""
Image Augmentation for Histopathological Image Segmentation
-------------------------------------------------
Augments training data by applying transformations (flips and rotations)
to images containing specific tissue types.

This script selectively augments images containing underrepresented tissue classes
to address class imbalance in histopathological image segmentation datasets.
"""

import os
import argparse
import numpy as np
import skimage.io as io
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Augment training data for segmentation tasks")
    parser.add_argument("--x_dir", type=str, required=True, 
                        help="Directory containing input images")
    parser.add_argument("--y_dir", type=str, required=True, 
                        help="Directory containing segmentation masks")
    parser.add_argument("--target_classes", type=str, nargs="+", 
                        default=["BCC", "SCC", "IEC", "FOL"],
                        help="List of target class names to augment")
    parser.add_argument("--augment_flips", action="store_true", default=True,
                        help="Apply flips (LR, UD) for augmentation")
    parser.add_argument("--augment_rotations", action="store_true", default=True,
                        help="Apply rotations (90°, 180°, 270°) for augmentation")
    return parser.parse_args()


def get_color_dict() -> Dict[str, List[int]]:
    """Return color mapping for tissue classes."""
    return {
        "EPI":  [73, 0, 106],
        "GLD":  [108, 0, 115],
        "INF":  [145, 1, 122],
        "RET":  [181, 9, 130],
        "FOL":  [216, 47, 148],
        "PAP":  [236, 85, 157],
        "HYP":  [254, 246, 242],
        "KER":  [248, 123, 168],
        "BKG":  [0, 0, 0],
        "BCC":  [127, 255, 255],
        "SCC":  [127, 255, 142],
        "IEC":  [255, 127, 127]
    }


def contains_target_class(mask: np.ndarray, class_color: List[int]) -> bool:
    """Check if mask contains the target class."""
    return np.any(np.all(mask == tuple(class_color), axis=-1))


def perform_augmentation(image: np.ndarray, mask: np.ndarray, 
                         file_name: str, x_dir: str, y_dir: str,
                         do_flips: bool = True, do_rotations: bool = True) -> int:
    """Perform data augmentation and save results.
    
    Args:
        image: Input image
        mask: Segmentation mask
        file_name: Original file name
        x_dir: Output directory for augmented images
        y_dir: Output directory for augmented masks
        do_flips: Whether to perform flip augmentations
        do_rotations: Whether to perform rotation augmentations
        
    Returns:
        Number of augmented images created
    """
    count = 0
    name_base = Path(file_name).stem
    
    # List of augmentation operations
    augmentations = []
    
    if do_flips:
        flip_ops = [
            ("LR", np.fliplr),
            ("UD", np.flipud)
        ]
        augmentations.extend(flip_ops)
    
    # Apply augmentations
    for aug_name, aug_func in augmentations:
        # Apply the transformation
        mask_aug = aug_func(mask)
        image_aug = aug_func(image)
        
        # Apply rotations if requested
        if do_rotations:
            rotations = [
                ("0", 0),
                ("90", 1), 
                ("180", 2), 
                ("270", 3)
            ]
            
            for rot_name, k in rotations:
                # Apply rotation
                mask_out = np.rot90(mask_aug, k)
                image_out = np.rot90(image_aug, k)
                
                # Generate output filename
                out_name = f"{name_base}_{aug_name}_{rot_name}.png"
                
                # Save augmented images
                io.imsave(os.path.join(y_dir, out_name), mask_out, check_contrast=False)
                io.imsave(os.path.join(x_dir, out_name), image_out, check_contrast=False)
                count += 1
        else:
            # Save just the flipped images without rotation
            out_name = f"{name_base}_{aug_name}.png"
            io.imsave(os.path.join(y_dir, out_name), mask_aug, check_contrast=False)
            io.imsave(os.path.join(x_dir, out_name), image_aug, check_contrast=False)
            count += 1
            
    return count


def main():
    """Main function to orchestrate data augmentation."""
    # Parse arguments
    args = parse_args()
    
    # Get color mapping
    color_dict = get_color_dict()
    
    # Get list of files
    files = os.listdir(args.x_dir)
    
    print(f"Found {len(files)} original images. Starting augmentation...")
    
    # Track statistics
    total_augmented = 0
    augmented_files = 0
    
    # Process each file
    for file in tqdm(files, desc="Augmenting"):
        # Load mask to check for target classes
        mask_path = os.path.join(args.y_dir, file)
        if not os.path.exists(mask_path):
            continue
            
        mask = io.imread(mask_path)
        
        # Check if any target class is present
        for class_name in args.target_classes:
            if class_name in color_dict and contains_target_class(mask, color_dict[class_name]):
                # Load the image
                image_path = os.path.join(args.x_dir, file)
                image = io.imread(image_path)
                
                # Perform augmentation
                num_created = perform_augmentation(
                    image, mask, file, args.x_dir, args.y_dir,
                    do_flips=args.augment_flips,
                    do_rotations=args.augment_rotations
                )
                
                total_augmented += num_created
                augmented_files += 1
                
                # Once we find a target class, no need to check other classes
                break
    
    print("\nAugmentation Summary:")
    print(f"Original images processed: {len(files)}")
    print(f"Images containing target classes: {augmented_files}")
    print(f"New augmented images created: {total_augmented}")
    print(f"Total dataset size: {len(files) + total_augmented} images")


if __name__ == "__main__":
    main()
