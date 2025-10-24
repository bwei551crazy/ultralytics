import os
import shutil
import random

def create_remaining_contents_folder(partial_folder, all_folder, output_folder):
    """
    Create a folder containing files that are in the all_folder but not in the partial_folder.
    
    Args:
        partial_folder (str): Path to the folder containing partial contents
        all_folder (str): Path to the folder containing all contents
        output_folder (str): Path where the remaining contents will be copied
    """
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get list of files in both folders
    try:
        partial_files = set(os.listdir(partial_folder))
        all_files = set(os.listdir(all_folder))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Find files that are in 'all' but not in 'partial'
    remaining_files = all_files - partial_files
    
    if not remaining_files:
        print("No remaining files found. The partial folder already contains all files.")
        return
    
    # Copy remaining files to output folder
    copied_count = 0
    for file_name in remaining_files:
        source_path = os.path.join(all_folder, file_name)
        dest_path = os.path.join(output_folder, file_name)
        
        try:
            if os.path.isfile(source_path):
                shutil.copy2(source_path, dest_path)
                copied_count += 1
            elif os.path.isdir(source_path):
                shutil.copytree(source_path, dest_path)
                copied_count += 1
        except Exception as e:
            print(f"Error copying {file_name}: {e}")
    
    print(f"Successfully copied {copied_count} items to {output_folder}")


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

    LABEL_FOLDER = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/BDD100k_yolo/80%BDD100k_val/labels"  # Change this to your label folder path
    IMAGE_FOLDER = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/BDD100k_yolo/80%BDD100k_val/images"  # Change this to your image folder path
    OUTPUT_FOLDER = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/BDD100k_yolo/80%BDD100k_val/split"   # Output folder name
    PERCENTAGE = 0.05  # 5% of the data

    # PARTIAL_DIR_IMG = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/BDD100k_yolo/20% BDD100k_val/images_orig"
    # PARTIAL_DIR_LBL = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/BDD100k_yolo/20% BDD100k_val/labels_orig"
    # OUTPUT_DIR = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/BDD100k_yolo/80%BDD100k_val/labels"

    
    create_fine_tune_dataset(LABEL_FOLDER, IMAGE_FOLDER, OUTPUT_FOLDER, PERCENTAGE)

    # create_remaining_contents_folder(PARTIAL_DIR_LBL, LABEL_FOLDER, OUTPUT_DIR)
