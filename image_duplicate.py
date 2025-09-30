import os
import shutil
from collections import Counter
from tqdm import tqdm

img_dir = "datasets/ua-detrac (original)/images/train"
label_dir = "datasets/ua-detrac (original)/labels/train"

class_name = ['truck','car','van','bus']

balance_ratio = 0.5

def get_label_classes(label_file):
    classes = []
    with open(label_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            cls_id = int(line.split()[0])
            if cls_id < len(class_name):
                classes.append(class_name[cls_id])
    return classes

def count_classes():
    counts = Counter()
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        label_path = os.path.join(label_dir, label_file)
        for cls in get_label_classes(label_path):
            counts[cls] += 1
    return counts

def oversample():
    counts = count_classes()
    #print("Class distribution before oversampling:", counts)

    max_class = "car"

    max_count = counts[max_class]

    target = int(max_count * balance_ratio)

    print(f"target for minorities = {target} (50% of {max_class})")

    #print("Max class count:", max_count)

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        label_path = os.path.join(label_dir, label_file)
        img_name = os.path.splitext(label_file)[0]
        #print(f"image name {img_name}")

        img_path = None

        for ext in ['.jpg', '.jpeg']:
            cand = os.path.join(img_dir, img_name + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            print(f"Image for label {label_file} not found, skipping.")
            continue
        
        cls_in_img = get_label_classes(label_path)
        if not cls_in_img:
            continue
        
        if set(cls_in_img) == {"car"}:
            continue

        oversample_factor = 1
        for cls in set(cls_in_img):
            
            if cls == "car":
                continue
            
            if counts[cls] < target:
                factor = target // counts[cls]
                if factor > oversample_factor:
                    oversample_factor = factor
        
        if oversample_factor > 1:
            for i in range(1, oversample_factor):
                new_img = os.path.join(img_dir, f"{img_name}_dup{i}{os.path.splitext(img_path)[1]}")
                new_lbl = os.path.join(label_dir, f"{img_name}_dup{i}.txt")
                if not os.path.exists(new_img):
                    shutil.copy(img_path, new_img)
                    shutil.copy(label_path, new_lbl)
            print(f"Oversampled {img_name} by factor x{oversample_factor}")
    
    new_counts = count_classes()
    print("Class distribution after oversampling:", new_counts)

def cleanup_duplicates():
    delete_files = 0
    for folder in [img_dir, label_dir]:
        for file in tqdm(os.listdir(folder)):
            if "_dup" in file:
                file_path = os.path.join(folder, file)
                os.remove(file_path)
                delete_files += 1
                #print(f"Deleted {delete_files} duplicate files.")
    print(f"Cleanup complete. Deleted {delete_files} files.")

if __name__ == "__main__":
    #oversample()
    #cleanup_duplicates()
    print(count_classes())

