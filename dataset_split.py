import os
import shutil
import random
import argparse

def create_fine_tune_dataset(label_folder, image_folder, output_folder, percentage, image_extensions=None):
    """
    Create a fine-tuning dataset by copying a specified percentage of labels and their corresponding images.
    
    Args:
        label_folder (str): Path to the folder containing label files
        image_folder (str): Path to the folder containing image files
        output_folder (str): Path where the fine-tuning dataset will be created
        percentage (float): Percentage of data to copy (0.0 to 1.0)
        image_extensions (list): List of image file extensions to look for
    """
    
    # Default image extensions
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Create output directories
    labels_output = os.path.join(output_folder, 'labels')
    images_output = os.path.join(output_folder, 'images')
    
    os.makedirs(labels_output, exist_ok=True)
    os.makedirs(images_output, exist_ok=True)
    
    # Get all label files
    label_files = [f for f in os.listdir(label_folder) 
                  if os.path.isfile(os.path.join(label_folder, f))]
    
    # Filter for common label file extensions
    label_extensions = ['.txt', '.xml', '.json']  # Add more if needed
    label_files = [f for f in label_files 
                  if any(f.lower().endswith(ext) for ext in label_extensions)]
    
    if not label_files:
        print(f"No label files found in {label_folder}")
        return
    
    print(f"Found {len(label_files)} label files")
    
    # Calculate number of files to copy
    num_files_to_copy = int(len(label_files) * percentage)
    print(f"Copying {num_files_to_copy} files ({percentage*100:.1f}%)")
    
    # Randomly select files
    selected_files = random.sample(label_files, num_files_to_copy)
    
    # Counter for successful copies
    copied_count = 0
    missing_images = []
    
    # Copy selected files
    for label_file in selected_files:
        # Get filename without extension
        filename = os.path.splitext(label_file)[0]
        
        # Find corresponding image file
        image_file = None
        for ext in image_extensions:
            potential_files = [
                f for f in os.listdir(image_folder) 
                if f.startswith(filename) and f.lower().endswith(ext)
            ]
            if potential_files:
                image_file = potential_files[0]
                break
        
        if image_file:
            # Copy label file
            src_label = os.path.join(label_folder, label_file)
            dst_label = os.path.join(labels_output, label_file)
            shutil.copy2(src_label, dst_label)
            
            # Copy image file
            src_image = os.path.join(image_folder, image_file)
            dst_image = os.path.join(images_output, image_file)
            shutil.copy2(src_image, dst_image)
            
            copied_count += 1
        else:
            missing_images.append(filename)
    
    # Print summary
    print(f"\nSuccessfully copied {copied_count} label-image pairs")
    if missing_images:
        print(f"Warning: Could not find images for {len(missing_images)} labels:")
        for missing in missing_images[:5]:  # Show first 5 missing
            print(f"  - {missing}")
        if len(missing_images) > 5:
            print(f"  ... and {len(missing_images) - 5} more")

if __name__ == "__main__":
     # Configuration - Modify these paths and percentage as needed

    LABEL_FOLDER = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/BDD100k_yolo/filtered_val/labels"  # Change this to your label folder path
    IMAGE_FOLDER = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/BDD100k_yolo/filtered_val/images"  # Change this to your image folder path
    OUTPUT_FOLDER = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/BDD100k_yolo/fine_tune_dataset"   # Output folder name
    PERCENTAGE = 0.2  # 20% of the data
    
    create_fine_tune_dataset(LABEL_FOLDER, IMAGE_FOLDER, OUTPUT_FOLDER, PERCENTAGE)
