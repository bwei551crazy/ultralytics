"""
File for switching labels in datasets around with capability to check for duplicates too. 

"""

import os
import glob
import json
from pathlib import Path
from tqdm import tqdm

#Map relevant BD100k categories to COCO categories
#LEFT: BDD100k CLASS ID
#RIGHT: COCO CLASS ID
#each lane marking class isn't available in coco dataset
#rider class from bdd100k also not available in coco dataset
bdd100k_to_coco = {
    # 0: 1,
    # 1: 5,
    # 4: 0,
    # 5: 80,
    # 6: 9,
    # 7: 11,
    # 8: 6,
    # 9: 7

    1: 0,
    5: 1,
    0: 4,
    80: 5,
    9: 6,
    11: 7,
    6: 8,
    7: 9
}

#VISDRONE DATASET to COCO dataset
#LEFT: VISDRONE CLASS ID
#RIGHT: COCO CLASS ID

visdrone_to_coco = {
    #0: 0,  #pedestrian (people standing/walking)
    81: 1,   #people (in sitting)
    1: 2,   #bicycle
    2: 3,   #car
    80: 4,  #van 
    7: 5,   #truck  
    83: 6,   #tricycle (the pedalling kind of tricycles. Can have that huge area at the back for storage)
    82: 7,   #Motored vers of the tricycle above. Typically will be more encased
    5: 8,   #bus    
    3: 9    #motor (coco labels motor as motorcycle)
}

#function to remap the labels of given dataset to the labels used in coco dataset

def remap_labels(label_dir, map_dic):
    """Remap UA-DETRAC labels to COCO class IDs"""
    for label_file in tqdm(os.listdir(label_dir)):
        if label_file.endswith('.txt'):
            label_path = os.path.join(label_dir, label_file)
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_class = int(parts[0])
                    if old_class in map_dic:
                        new_class = map_dic[old_class]
                        parts[0] = str(new_class)
                        new_lines.append(' '.join(parts) + '\n')
                    else:
                        new_lines.append(' '.join(parts) + '\n')
            
            # Write back modified labels
            with open(label_path, 'w') as f:
                f.writelines(new_lines)

#Used to help check whether a certain class id exist as a label in a dataset
def find_class_id_in_labels(folder_path, target_class_id):
    """
    Search through all .txt files in a folder for a specific class ID
    """
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    files_with_class = []
    
    for file_path in txt_files:
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                # COCO format: class_id x_center y_center width height
                parts = line.strip().split()
                if parts and parts[0] == str(target_class_id):
                    files_with_class.append({
                        'file': file_path,
                        'line_number': line_num,
                        'line_content': line.strip()
                    })
    
    return files_with_class

def has_duplicate_dicts(dict_list):
    seen = set()
    for d in dict_list:
        # Convert dict to JSON string for hashing
        dict_str = json.dumps(d, sort_keys=True)
        if dict_str in seen:
            return True
        seen.add(dict_str)
    return False

if __name__ == "__main__":

    folder_path = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/BDD100k_yolo/labels/val"
    target_class_id = [0,1,2,3,4,5,6,7,8,9]  # Change this to your desired class ID

    for i in target_class_id:

        print(f"Searching for class ID {i} in labels...")
        results = find_class_id_in_labels(folder_path, i)
        print(f"Found class ID {i} {len(results)} times within the labels ")
    
    #print(f"Found duplicates: {has_duplicate_dicts(results)}")

    # #Put absolute path to which image folder you want to use for remapping labels
    # label_directory = '/home/yanjiaqi/own_ultralytics/ultralytics/datasets/BDD100k_yolo/labels/train'
    
    # #CHANGE THE MAPPING PARAMETER BEFORE EXECUTING THE FILE!!!!!!
    # remap_labels(label_directory, bdd100k_to_coco)

   