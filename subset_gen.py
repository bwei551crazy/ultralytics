"""
This file contains code for creating a subset of a dataset, to tackle the challenge of imbalanced class distribution object 
detection dataset. 

"""

import os
import shutil
import random
from tqdm import tqdm
from pathlib import Path

"""
Create a subset of a dataset containing at least one of the underrepresented classes

Args:
    orig_img_dir (str): Path to the original images directory
    orig_lbl_dir (str): Path to the original labels directory
    sav_img_dir (str): Path to the directory where subset images will be saved
    sav_lbl_dir (str): Path to the directory where subset labels will be saved
    under_rep_classes (list): List of underrepresented class IDs to include in the subset
    copy_ratio (float): Ratio of images containing underrepresented classes to copy (default is 1.0, meaning all such images are copied)

"""

def create_subset(orig_img_dir, orig_lbl_dir, sav_img_dir, sav_lbl_dir, 
                                   class_ratios):
    """
    Create a subset with specified duplication ratios for each class
    Each class is processed independently to ensure exact ratios
    
    Args:
        class_ratios: Dictionary with {class_index: ratio} e.g., {0: 0.5, 1: 2.0, 2: 1.5}
    """
    
    os.makedirs(sav_img_dir, exist_ok=True)
    os.makedirs(sav_lbl_dir, exist_ok=True)

    lbl_files = list(Path(orig_lbl_dir).glob('*.txt'))
    print(f"Found {len(lbl_files)} label files")

    # Step 1: Find files that contain the classes we want to process
    target_classes = set(class_ratios.keys())
    class_files = {class_idx: [] for class_idx in target_classes}
    
    for lbl_file in lbl_files:
        try: 
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
                
            file_classes = set()
            for line in lines:
                if line.strip():
                    try:
                        class_idx = int(line.split()[0])
                        file_classes.add(class_idx)
                    except (ValueError, IndexError):
                        print(f"Warning: Malformed line in {lbl_file}: {line.strip()}")
                        continue
            
            # Only add files that contain our target classes
            for class_idx in target_classes:
                if class_idx in file_classes:
                    class_files[class_idx].append(lbl_file)
                
        except Exception as e:
            print(f"Error reading {lbl_file}: {e}")
    
    print(f"Found files for target classes:")
    for class_idx, files in class_files.items():
        print(f"  Class {class_idx}: {len(files)} files")

    # Step 2: Process each class completely independently
    files_to_copy = []
    
    for class_idx, files in class_files.items():
        ratio = class_ratios[class_idx]
        
        if ratio <= 0:
            continue
            
        print(f"\nProcessing class {class_idx} with ratio {ratio}")
        
        if ratio <= 1.0:
            # Under-sampling: take a subset
            random.shuffle(files)
            num_files = max(1, int(len(files) * ratio))
            selected_files = files[:num_files]
            print(f"  Selecting {num_files} out of {len(files)} files")
            
            # Add each file once
            for file in selected_files:
                files_to_copy.append((file, 1, class_idx))
                
        else:
            # Over-sampling: duplicate ALL files for this class
            base_copies = int(ratio)
            fractional_part = ratio - base_copies
            
            print(f"  Duplicating {len(files)} files:")
            print(f"    Base copies: {base_copies}")
            print(f"    Fractional: {fractional_part:.2f}")
            
            total_copies_for_class = 0
            for file in files:
                copies = base_copies
                # Handle fractional part probabilistically
                if random.random() < fractional_part:
                    copies += 1
                
                files_to_copy.append((file, copies, class_idx))
                total_copies_for_class += copies
            
            print(f"    Total copies for class {class_idx}: {total_copies_for_class}")

    print(f"\nTotal file-copy operations: {len(files_to_copy)}")
    
    # Step 3: Copy files with duplication
    copied_cnt = 0
    img_cnt = 0
    skipped_cnt = 0
    class_copy_count = {class_idx: 0 for class_idx in target_classes}
    
    for file_path, num_copies, class_idx in tqdm(files_to_copy):
        base_name = file_path.stem
        
        for copy_num in range(num_copies):
            try:
                img_copied = False
                
                # Create unique filename for each copy
                if num_copies > 1:
                    new_base_name = f"{base_name}_c{class_idx}_copy{copy_num+1}"
                else:
                    new_base_name = f"{base_name}_c{class_idx}"
                
                source_lbl = file_path
                dest_lbl = Path(sav_lbl_dir) / f"{new_base_name}.txt"

                source_img = Path(orig_img_dir) / f"{base_name}.jpg"    
                if source_img.exists():
                    dest_img = Path(sav_img_dir) / f"{new_base_name}.jpg"
                    shutil.copy2(source_img, dest_img)
                    img_copied = True
                    img_cnt += 1
                
                if img_copied: 
                    shutil.copy2(source_lbl, dest_lbl)
                    copied_cnt += 1
                    class_copy_count[class_idx] += 1
                else:
                    print(f"No image found for {base_name}, so skipped")
                    skipped_cnt += 1
                    
            except Exception as e:
                print(f"Error copying files for {base_name} (copy {copy_num+1}): {e}")
                skipped_cnt += 1
                continue

    print(f"\n=== FINAL RESULTS ===")
    print(f"Successfully copied {copied_cnt} image-label pairs")
    print(f"Total images copied: {img_cnt}")
    print(f"Skipped {skipped_cnt} files due to errors or missing images")
    
    print(f"\nClass distribution after duplication:")
    for class_idx in target_classes:
        original_count = len(class_files[class_idx])
        ratio = class_ratios[class_idx]
        expected_count = original_count * ratio if ratio > 1.0 else original_count * ratio
        actual_count = class_copy_count[class_idx]
        
        print(f"  Class {class_idx}:")
        print(f"    Original: {original_count} files")
        print(f"    Ratio: {ratio}")
        print(f"    Expected: {expected_count:.0f} copies")
        print(f"    Actual: {actual_count} copies")
        print(f"    Achievement: {(actual_count/expected_count*100):.1f}% of target")

if __name__ == "__main__":
    
    ORIGNAL_IMG_DIR = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/BDD100k_yolo/images/val"
    ORIGNAL_LBL_DIR = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/BDD100k_yolo/labels/val"
    SAVE_IMG_DIR = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/BDD100k_yolo_subset/images/val"
    SAVE_LBL_DIR = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/BDD100k_yolo_subset/labels/val"

    # Change this to your underrepresented class IDs
    #Left: Class ID, Right: Duplication amount
    UNDER_REP_CLASSES = {
        1:  1.72,
        3:  3.82,
        6:  71.4,
        80: 2.0
    } 

    print("creating subset...")
    create_subset(
        ORIGNAL_IMG_DIR, 
        ORIGNAL_LBL_DIR, 
        SAVE_IMG_DIR, 
        SAVE_LBL_DIR, 
        UNDER_REP_CLASSES
    )