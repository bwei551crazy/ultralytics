import os
import shutil
import random
import cv2
import numpy as np
from collections import Counter, defaultdict
import albumentations as A
import argparse

class SmartOversampler:
    def __init__(self, dataset_path, target_classes=[1, 3, 6, 80]):
        self.dataset_path = dataset_path
        self.target_classes = target_classes
        self.setup_augmentations()
        
    def setup_augmentations(self):
        """Setup augmentation strategies - using ONLY color/quality transforms"""
        
        # Use ONLY non-geometric augmentations to avoid coordinate issues
        self.augmentation_color = A.Compose([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.ChannelShuffle(p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ])
        
        self.augmentation_weather = A.Compose([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
            A.RandomRain(p=0.2),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.2),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.MedianBlur(blur_limit=5, p=0.2),
        ])
        
        self.all_augmentations = [
            self.augmentation_color,
            self.augmentation_weather,
        ]
    
    def filter_target_files(self, split='train'):
        """Find all files containing target classes and copy them to a filtered folder"""
        original_label_dir = os.path.join(self.dataset_path, 'labels', split)
        original_image_dir = os.path.join(self.dataset_path, 'images', split)
        
        # Create filtered dataset folder
        filtered_path = os.path.join(self.dataset_path, f'filtered_{split}')
        filtered_label_dir = os.path.join(filtered_path, 'labels')
        filtered_image_dir = os.path.join(filtered_path, 'images')
        
        os.makedirs(filtered_label_dir, exist_ok=True)
        os.makedirs(filtered_image_dir, exist_ok=True)
        
        target_files = []
        class_counts = Counter()
        
        print("Filtering files containing target classes...")
        
        for label_file in os.listdir(original_label_dir):
            if label_file.endswith('.txt'):
                label_path = os.path.join(original_label_dir, label_file)
                
                # Check if this file contains any target classes
                with open(label_path, 'r') as f:
                    has_target_class = False
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if class_id in self.target_classes:
                                has_target_class = True
                                class_counts[class_id] += 1
                
                if has_target_class:
                    target_files.append(label_file)
                    
                    # Copy label file
                    shutil.copy2(label_path, os.path.join(filtered_label_dir, label_file))
                    
                    # Find and copy corresponding image file
                    base_name = label_file.replace('.txt', '')
                    for ext in ['.jpg', '.jpeg', '.png']:
                        image_path = os.path.join(original_image_dir, base_name + ext)
                        if os.path.exists(image_path):
                            shutil.copy2(image_path, os.path.join(filtered_image_dir, base_name + ext))
                            break
        
        print(f"Found {len(target_files)} files containing target classes")
        print("Class distribution in filtered set:")
        for class_id in self.target_classes:
            print(f"Class {class_id}: {class_counts[class_id]} instances")
        
        return filtered_path, target_files, class_counts
    
    def calculate_oversampling_factors(self, class_counts):
        """Calculate how many copies to make based on class frequency"""
        max_count = max(class_counts.values()) if class_counts else 1
        oversampling_factors = {}
        
        for class_id in self.target_classes:
            count = class_counts.get(class_id, 0)
            if count == 0:
                oversampling_factors[class_id] = 0
                continue
                
            ratio = max_count / count
            factor = min(int(np.log2(ratio) * 2), 8)  # Cap at 8 copies
            factor = max(factor, 1)
            
            oversampling_factors[class_id] = factor
            print(f"Class {class_id}: {count} instances -> {factor} copies per file")
        
        return oversampling_factors
    
    def apply_augmentation(self, image, augmentation):
        """Apply augmentation to image only (no bbox coordinate issues)"""
        try:
            augmented = augmentation(image=image)
            return augmented['image']
        except Exception as e:
            print(f"Augmentation failed: {e}, using original")
            return image
    
    def oversample_filtered_dataset(self, split='train'):
        """Main oversampling function using filtered dataset"""
        filtered_path, target_files, class_counts = self.filter_target_files(split)
        
        filtered_label_dir = os.path.join(filtered_path, 'labels')
        filtered_image_dir = os.path.join(filtered_path, 'images')
        
        # Create augmented dataset folder
        augmented_path = os.path.join(self.dataset_path, f'augmented_{split}')
        augmented_label_dir = os.path.join(augmented_path, 'labels')
        augmented_image_dir = os.path.join(augmented_path, 'images')
        
        os.makedirs(augmented_label_dir, exist_ok=True)
        os.makedirs(augmented_image_dir, exist_ok=True)
        
        oversampling_factors = self.calculate_oversampling_factors(class_counts)
        
        print(f"\nStarting oversampling with {len(target_files)} filtered files...")
        
        total_new_files = 0
        
        for file_idx, label_file in enumerate(target_files):
            if file_idx % 100 == 0:
                print(f"Processing file {file_idx}/{len(target_files)}")
            
            # Determine max oversampling factor for this file
            label_path = os.path.join(filtered_label_dir, label_file)
            file_classes = set()
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        file_classes.add(class_id)
            
            max_factor = 0
            for class_id in file_classes:
                if class_id in self.target_classes:
                    max_factor = max(max_factor, oversampling_factors[class_id])
            
            if max_factor == 0:
                continue
            
            # Load image
            base_name = label_file.replace('.txt', '')
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_path = os.path.join(filtered_image_dir, base_name + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path is None:
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Copy original files to augmented dataset first
            shutil.copy2(label_path, os.path.join(augmented_label_dir, label_file))
            shutil.copy2(image_path, os.path.join(augmented_image_dir, base_name + os.path.splitext(image_path)[1]))
            
            # Create augmented copies
            for copy_idx in range(max_factor):
                # Apply random augmentation to image only
                aug = random.choice(self.all_augmentations)
                aug_image = self.apply_augmentation(image, aug)
                
                # Create new filenames
                new_base_name = f"{base_name}_aug{copy_idx:02d}"
                new_image_name = new_base_name + os.path.splitext(image_path)[1]
                new_label_name = new_base_name + '.txt'
                
                # Save augmented image
                new_image_path = os.path.join(augmented_image_dir, new_image_name)
                cv2.imwrite(new_image_path, aug_image)
                
                # Copy original labels (same bboxes, different image)
                # Since we're only doing color/quality augmentations, bboxes don't change
                shutil.copy2(label_path, os.path.join(augmented_label_dir, new_label_name))
                
                total_new_files += 1
        
        print(f"\nOversampling completed!")
        print(f"Created {total_new_files} new augmented samples")
        print(f"Filtered dataset: {filtered_path}")
        print(f"Augmented dataset: {augmented_path}")
        
        return augmented_path

def main():
    parser = argparse.ArgumentParser(description='Smart oversampling for BDD100k dataset')
    parser.add_argument('--dataset_path', type=str, required=True, 
                       help='Path to BDD100k dataset directory')
    parser.add_argument('--split', type=str, default='train', 
                       help='Dataset split to process (train/val)')
    parser.add_argument('--target_classes', type=int, nargs='+', 
                       default=[1, 3, 6, 80], 
                       help='Target class IDs to oversample')
    
    args = parser.parse_args()
    
    # Initialize oversampler
    oversampler = SmartOversampler(
        dataset_path=args.dataset_path,
        target_classes=args.target_classes
    )
    
    # Run oversampling
    augmented_path = oversampler.oversample_filtered_dataset(split=args.split)
    print(f"\nUse this path for training: {augmented_path}")

if __name__ == "__main__":
    main()