"""
Training Set Generator
---------------------
Creates training, validation, and test sets for histopathological image segmentation
while respecting patient-based data splits to prevent data leakage.

Usage:
    python training_set_generator.py -n 10 --base_dir ./data/ --split 0.8 --dim 512
"""

import os
import sys
import argparse
import numpy as np
import platform
import shutil
from pathlib import Path
from typing import List, Dict, Tuple


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create training set of patches from n images")
    parser.add_argument("-n", type=int, default=290, 
                        help="Number of images in training set")
    parser.add_argument("--base_dir", type=str, default="./data/", 
                        help="Path to data directory")
    parser.add_argument("--split", type=float, default=0.8, 
                        help="Train/validation split ratio (0-1)")
    parser.add_argument("--dim", type=int, default=512, 
                        help="Dimension of patches")
    parser.add_argument("--copy", action="store_true",
                        help="Force copying files instead of symlinking")
    return parser.parse_args()


def get_patient_groups() -> List[List[str]]:
    """Return groups of images from the same patient to prevent data leakage."""
    return [
        ['SCC_1', 'SCC_2'],
        ['IEC_2', 'IEC_3'],
        ['BCC_1', 'BCC_2'],
        ['BCC_5', 'BCC_6'],
        ['SCC_5', 'SCC_6'],
        ['BCC_7', 'BCC_8'],
        ['BCC_11', 'BCC_12'],
        ['SCC_10', 'SCC_11'],
        ['BCC_19', 'BCC_20', 'BCC_21'],
        ['BCC_23', 'BCC_24'],
        ['BCC_25', 'BCC_26'],
        ['BCC_27', 'BCC_28'],
        ['IEC_4', 'IEC_5'],
        ['IEC_6', 'IEC_7'],
        ['IEC_8', 'IEC_9'],
        ['IEC_10', 'IEC_11'],
        ['BCC_33', 'BCC_34'],
        ['IEC_13', 'IEC_14'],
        ['BCC_37', 'BCC_38'],
        ['BCC_40', 'BCC_41'],
        ['IEC_19', 'IEC_20'],
        ['IEC_22', 'IEC_23'],
        ['IEC_27', 'IEC_28'],
        ['IEC_29', 'IEC_30'],
        ['BCC_46', 'BCC_47'],
        ['SCC_15', 'SCC_16'],
        ['IEC_34', 'IEC_35'],
        ['BCC_55', 'BCC_56'],
        ['IEC_43', 'IEC_44'],
        ['IEC_51', 'IEC_52'],
        ['BCC_60', 'BCC_61'],
        ['SCC_17', 'SCC_18'],
        ['BCC_83', 'BCC_84'],
        ['SCC_34', 'SCC_35']
    ]


def setup_directories(base_dir: str, split: float, dim: int) -> Tuple[str, str]:
    """Setup input/output directories."""
    # Input directory (patches)
    patch_dir = os.path.join(base_dir, f"Patches_Overlapped_{dim}")
    
    # Output directory for training data
    training_dir = os.path.join(base_dir, f"TrainingData_{int(split*100)}")
    Path(training_dir).mkdir(parents=True, exist_ok=True)
    
    return patch_dir, training_dir


def create_dataset_structure(training_dir: str, num_files: int) -> str:
    """Create dataset directory structure."""
    # Create dataset directory
    set_name = f"Data_{num_files}"
    set_dir = os.path.join(training_dir, set_name)
    Path(set_dir).mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for train/val/test splits
    for subdir in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        Path(os.path.join(set_dir, subdir)).mkdir(parents=True, exist_ok=True)
    
    return set_dir


def split_data(files: List[str], patient_groups: List[List[str]], 
               split: float, num_files: int) -> Dict[str, List[str]]:
    """Split data into training, validation and test sets."""
    # Remove patient groups temporarily to avoid data leakage
    filtered_files = files.copy()
    for group in patient_groups:
        for file in group:
            if file in filtered_files:
                filtered_files.remove(file)
    
    # Calculate split positions
    val_pos = int(split * len(filtered_files))
    test_pos = val_pos + (len(filtered_files) - val_pos) // 2
    
    # Split the files
    data_splits = {
        "train": filtered_files[:val_pos],
        "val": filtered_files[val_pos:test_pos],
        "test": filtered_files[test_pos:]
    }
    
    # Add patient groups back if we have enough files
    if num_files > 30:
        # Shuffle the patient groups
        np.random.shuffle(patient_groups)
        
        # Calculate split positions for patient groups
        val_pos = int(split * len(patient_groups))
        test_pos = val_pos + (len(patient_groups) - val_pos) // 2
        
        # Add patient groups to the respective splits
        for group in patient_groups[:val_pos]:
            data_splits["train"].extend(group)
        
        for group in patient_groups[val_pos:test_pos]:
            data_splits["val"].extend(group)
            
        for group in patient_groups[test_pos:]:
            data_splits["test"].extend(group)
    
    return data_splits


def link_or_copy_file(src: str, dst: str, use_copy: bool = False) -> None:
    """Create a symlink or copy a file depending on platform and permissions."""
    if os.path.exists(dst):
        return  # Skip if destination already exists
        
    try:
        if use_copy:
            shutil.copy2(src, dst)  # Copy with metadata
        else:
            os.symlink(src, dst)  # Try to create a symlink
    except (OSError, PermissionError):
        # Fall back to copying if symlink fails
        print(f"Warning: Could not create symlink. Copying file instead: {dst}")
        shutil.copy2(src, dst)


def create_dataset_links(set_dir: str, patch_dir: str, data_splits: Dict[str, List[str]], 
                         force_copy: bool = False) -> None:
    """Create symlinks or copies for patches in the dataset structure."""
    # Determine if we're running on Windows
    is_windows = platform.system() == "Windows"
    use_copy = force_copy or is_windows
    
    if use_copy:
        print("Using file copying instead of symbolic links")
    
    # Mapping between split names and directory names
    dir_mapping = {
        "train": {"img": "X_train", "mask": "y_train", "file_list": "train_files.txt"},
        "val": {"img": "X_val", "mask": "y_val", "file_list": "validation_files.txt"},
        "test": {"img": "X_test", "mask": "y_test", "file_list": "test_files.txt"}
    }
    
    # Process each split
    for split_name, files in data_splits.items():
        # Create file list
        file_list_path = os.path.join(set_dir, dir_mapping[split_name]["file_list"])
        with open(file_list_path, 'w') as f:
            f.write('\n'.join(files))
        
        # Create symlinks or copies for each file
        files_processed = 0
        for file in files:
            file_patch_dir = os.path.join(patch_dir, file)
            
            # Skip if directory doesn't exist
            if not os.path.exists(file_patch_dir):
                print(f"Warning: Directory not found: {file_patch_dir}")
                continue
                
            x_dir = os.path.join(file_patch_dir, "X")
            if not os.path.exists(x_dir):
                print(f"Warning: X directory not found: {x_dir}")
                continue
                
            patch_files = os.listdir(x_dir)
            
            for patch in patch_files:
                # Source paths
                img_patch = os.path.join(patch_dir, file, "X", patch)
                mask_patch = os.path.join(patch_dir, file, "y", patch)
                
                # Check if mask exists
                if not os.path.exists(mask_patch):
                    continue
                
                # Destination paths
                img_out = os.path.join(set_dir, dir_mapping[split_name]["img"], patch)
                mask_out = os.path.join(set_dir, dir_mapping[split_name]["mask"], patch)
                
                # Create symlinks or copies
                link_or_copy_file(img_patch, img_out, use_copy)
                link_or_copy_file(mask_patch, mask_out, use_copy)
                files_processed += 1
            
            # Print progress
            if files_processed % 100 == 0 and files_processed > 0:
                print(f"Processed {files_processed} files...")


def main():
    """Main function to orchestrate training set generation."""
    # Parse arguments
    args = parse_args()
    
    # Setup directories
    patch_dir, training_dir = setup_directories(args.base_dir, args.split, args.dim)
    
    # Create dataset structure
    set_dir = create_dataset_structure(training_dir, args.n)
    
    # Get files and shuffle
    try:
        available_files = os.listdir(patch_dir)[:args.n]
    except FileNotFoundError:
        print(f"Error: Patch directory not found: {patch_dir}")
        return
        
    if not available_files:
        print(f"Error: No files found in patch directory: {patch_dir}")
        return
    
    np.random.shuffle(available_files)
    
    # Get patient groups to prevent data leakage
    patient_groups = get_patient_groups()
    
    # Split data into train/val/test
    data_splits = split_data(available_files, patient_groups, args.split, args.n)
    
    # Create dataset links or copies
    create_dataset_links(set_dir, patch_dir, data_splits, args.copy)
    
    # Print summary
    print(f"\nDataset created successfully at {set_dir}")
    print(f"Training: {len(data_splits['train'])} files")
    print(f"Validation: {len(data_splits['val'])} files")
    print(f"Test: {len(data_splits['test'])} files")


if __name__ == "__main__":
    main()
