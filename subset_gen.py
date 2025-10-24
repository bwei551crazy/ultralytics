import os
import random
import shutil
import argparse

def split_labels_and_images(labels_dir, images_dir, split_ratio=0.8, seed=42):
    """
    Split labels and corresponding images with the same split.
    
    Args:
        labels_dir: Path to directory containing label files (.txt)
        images_dir: Path to directory containing image files
        split_ratio: Ratio for train split (e.g., 0.8 = 80% train, 20% val)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Get all label files
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    print(f"Found {len(label_files)} label files")
    
    # Find corresponding images for each label file
    valid_pairs = []
    for label_file in label_files:
        base_name = os.path.splitext(label_file)[0]
        
        # Look for corresponding image with common extensions
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            image_file = base_name + ext
            image_path = os.path.join(images_dir, image_file)
            if os.path.exists(image_path):
                valid_pairs.append((label_file, image_file))
                break
        else:
            print(f"Warning: No image found for {label_file}")
    
    print(f"Found {len(valid_pairs)} valid label-image pairs")
    
    # Shuffle the pairs
    random.shuffle(valid_pairs)
    
    # Split the pairs
    split_index = int(len(valid_pairs) * split_ratio)
    train_pairs = valid_pairs[:split_index]
    val_pairs = valid_pairs[split_index:]
    
    print(f"Train split: {len(train_pairs)} pairs")
    print(f"Val split: {len(val_pairs)} pairs")
    
    # Create output directories
    train_labels_dir = os.path.join(labels_dir, 'train')
    train_images_dir = os.path.join(images_dir, 'train')
    val_labels_dir = os.path.join(labels_dir, 'val')
    val_images_dir = os.path.join(images_dir, 'val')
    
    for dir_path in [train_labels_dir, train_images_dir, val_labels_dir, val_images_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Copy files to train directories
    for label_file, image_file in train_pairs:
        shutil.copy2(os.path.join(labels_dir, label_file), train_labels_dir)
        shutil.copy2(os.path.join(images_dir, image_file), train_images_dir)
    
    # Copy files to val directories
    for label_file, image_file in val_pairs:
        shutil.copy2(os.path.join(labels_dir, label_file), val_labels_dir)
        shutil.copy2(os.path.join(images_dir, image_file), val_images_dir)
    
    print(f"\nCreated:")
    print(f"  {train_labels_dir}")
    print(f"  {train_images_dir}")
    print(f"  {val_labels_dir}")
    print(f"  {val_images_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split labels and images with the same ratio')
    parser.add_argument('--labels-dir', required=True, help='Path to labels directory')
    parser.add_argument('--images-dir', required=True, help='Path to images directory')
    parser.add_argument('--split-ratio', type=float, default=0.8, help='Train split ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    split_labels_and_images(args.labels_dir, args.images_dir, args.split_ratio, args.seed)