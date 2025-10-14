import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_dataset(image_train_dir, label_train_dir, image_val_dir, label_val_dir, split_ratio=0.8):
    """
    Split train dataset into train and validation sets by VIDEO ID
    
    Args:
        image_train_dir: Path to original image train directory
        label_train_dir: Path to original label train directory  
        image_val_dir: Path to new image validation directory
        label_val_dir: Path to new label validation directory
        split_ratio: Ratio of data to keep in train set (default: 0.8 = 80%)
    """
    
    # Create validation directories if they don't exist
    os.makedirs(image_val_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_train_dir) if os.path.isfile(os.path.join(image_train_dir, f))]
    
    # Extract unique video IDs from filenames
    # Assuming format like: "MVI_20011_img00001.jpg" or "MVI_20011_frame_0001.jpg"
    video_ids = set()
    for img_file in image_files:
        # Extract video ID - this depends on your filename format
        # Method 1: Split by common separators and take the first part
        parts = img_file.split('_')
        if len(parts) >= 2:
            # This gets "MVI_20011" from "MVI_20011_img00001.jpg"
            video_id = '_'.join(parts[:2])  # Adjust this based on your actual filename format
            video_ids.add(video_id)
    
    # Convert to list and shuffle
    video_ids = list(video_ids)
    random.shuffle(video_ids)
    
    print(f"Found {len(video_ids)} unique video sequences: {video_ids}")
    
    # Calculate split index
    split_index = int(len(video_ids) * split_ratio)
    
    # Split video IDs into train and validation sets
    train_video_ids = set(video_ids[:split_index])
    val_video_ids = set(video_ids[split_index:])
    
    print(f"Train videos: {train_video_ids}")
    print(f"Validation videos: {val_video_ids}")
    
    # Now process files based on video ID membership
    train_count = 0
    val_count = 0
    
    for img_file in tqdm(image_files):
        # Extract video ID from this image (same method as above)
        parts = img_file.split('_')
        if len(parts) >= 2:
            video_id = '_'.join(parts[:2])  # Adjust this based on your actual filename format
            
            # Get base name for label matching
            img_base = os.path.splitext(img_file)[0]
            label_file = img_base + ".txt"
            label_path = os.path.join(label_train_dir, label_file)
            
            # Skip if label doesn't exist
            if not os.path.exists(label_path):
                print(f"Warning: Label file {label_file} not found, skipping {img_file}")
                continue
            
            # Determine if this file goes to validation set
            if video_id in val_video_ids:
                # Move to validation directories
                img_src = os.path.join(image_train_dir, img_file)
                img_dst = os.path.join(image_val_dir, img_file)
                label_dst = os.path.join(label_val_dir, label_file)
                
                shutil.move(img_src, img_dst)
                shutil.move(label_path, label_dst)
                val_count += 1
            else:
                # File stays in train directory
                train_count += 1
    print(f"Train videos: {train_video_ids}")
    print(f"Validation videos: {val_video_ids}")
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Successfully moved {val_count} samples to validation set")

def main():
    # Define paths
    base_dir = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/ua-detrac (original)"  # Current directory, adjust if needed
    
    image_train_dir = os.path.join(base_dir, "images", "train")

    label_train_dir = os.path.join(base_dir, "labels", "train")
    
    image_val_dir = os.path.join(base_dir, "images", "val")
    label_val_dir = os.path.join(base_dir, "labels", "val")
    
    # Verify that train directories exist
    if not os.path.exists(image_train_dir):
        print(f"Error: Image train directory not found: {image_train_dir}")
        return
    
    if not os.path.exists(label_train_dir):
        print(f"Error: Label train directory not found: {label_train_dir}")
        return
    
    # Perform the split
    split_dataset(image_train_dir, label_train_dir, image_val_dir, label_val_dir, split_ratio=0.8)

if __name__ == "__main__":
    main()