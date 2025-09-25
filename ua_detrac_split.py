import os
import shutil
import random
from pathlib import Path

def split_dataset(image_train_dir, label_train_dir, image_val_dir, label_val_dir, split_ratio=0.8):
    """
    Split train dataset into train and validation sets
    
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
    
    # Get list of image files (without extensions to match with labels)
    image_files = [f for f in os.listdir(image_train_dir) if os.path.isfile(os.path.join(image_train_dir, f))]

    # Remove extensions to get base names for matching with labels
    base_names = [os.path.splitext(f)[0] for f in image_files]  

    
    # Get corresponding label files (assuming same base names with .txt extension)
    label_files = [f + ".txt" for f in base_names]

    print(label_files)
    
    #Filter out label files that don't actually exist
    valid_pairs = []
    for img_base, label_file in zip(base_names, label_files):
        label_path = os.path.join(label_train_dir, label_file)
        if os.path.exists(label_path):
            valid_pairs.append((img_base, label_file))
        else:
            print(f"Warning: Label file {label_file} not found, skipping {img_base}")
    
    # # Shuffle the pairs randomly
    # random.shuffle(valid_pairs)
    
    # # Calculate split index
    # split_index = int(len(valid_pairs) * split_ratio)
    
    # # Split into train and validation sets
    # train_pairs = valid_pairs[:split_index]
    # val_pairs = valid_pairs[split_index:]
    
    # print(f"Total samples: {len(valid_pairs)}")
    # print(f"Train samples: {len(train_pairs)}")
    # print(f"Validation samples: {len(val_pairs)}")
    
    # # Move validation files to their new directories
    # moved_count = 0
    # for img_base, label_file in val_pairs:
    #     # Find the original image file with correct extension
    #     matching_images = [f for f in image_files if os.path.splitext(f)[0] == img_base]
        
    #     if matching_images:
    #         img_file = matching_images[0]
    #         img_src = os.path.join(image_train_dir, img_file)
    #         img_dst = os.path.join(image_val_dir, img_file)
            
    #         label_src = os.path.join(label_train_dir, label_file)
    #         label_dst = os.path.join(label_val_dir, label_file)
            
    #         # Move files
    #         shutil.move(img_src, img_dst)
    #         shutil.move(label_src, label_dst)
    #         moved_count += 1
    
    # print(f"Successfully moved {moved_count} samples to validation set")

def main():
    # Define paths
    base_dir = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/ua-detrac"  # Current directory, adjust if needed
    
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