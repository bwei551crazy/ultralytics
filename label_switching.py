#File for switching mismatch labels in used dataset to the labels used for COCO dataset

import os
from pathlib import Path

#-------------------------class mapping to COCO dataset-------------------------------------#

#UC-DETRAC DATASET to COCO dataset
#LEFT: UA_DETRAC CLASS ID
#RIGHT: COCO CLASS ID

ua_detrac_to_coco = {
    0: 1,   # Truck  
    1: 0,   # Car
    2: 2,  # Van    #cause Van doesn't exist in COCO
    3: 3    # Bus
}

#VISDRONE DATASET to COCO dataset
#LEFT: VISDRONE CLASS ID
#RIGHT: COCO CLASS ID

visdrone_to_coco = {
    0: 0,  #pedestrian (people standing/walking)
    1: 0,   #people (in sitting)
    2: 1,   #bicycle
    3: 2,   #car
    4: 80,  #van 
    5: 7,   #truck  
    6: 1,   #tricycle (coco counts tricycle as bicycle)
    7: 1,   #awning-tricycle (coco counts tricycle as bicycle)
    8: 5,   #bus    
    9: 3    #motor (coco labels motor as motorcycle)
}

#function to remap the labels of given dataset to the labels used in coco dataset

def remap_ua_detrac_labels(label_dir, save_dir_name, mapping_dic):
    
    label_dir = Path(label_dir)
    #Grabs the parent directory of label_dir and appends the new folder name to it
    save_dir = label_dir.parent / save_dir_name
    save_dir.mkdir(parents = True, exist_ok=True) #creates the new directory if it doesn't exist

    ua_to_coco = mapping_dic
    
    for label_file in label_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            if class_id in ua_to_coco:
                new_class_id = ua_to_coco[class_id]
                new_line = f"{new_class_id} " + " ".join(parts[1:]) + "\n"
                new_lines.append(new_line)
            else:
                print(f"Warning: Class ID {class_id} not found in mapping dictionary.")

        save_path = save_dir / label_file.name
        with open(save_path, 'w') as f:
            f.writelines(new_lines)

#switch 3rd parameter with the appropriate mapping dictionary if using different dataset

if __name__ == "__main__":

    #Change the path correspondingly to where the dataset is store (must be the absolute path)   
    label_directory = '/home/yanjiaqi/own_ultralytics/ultralytics/datasets/ua-detrac (original)/content/UA-DETRAC/DETRAC_Upload/labels/train_original'
    
    # Modify name of the folder to save the remapped labels
    save_directory_name = 'train_coco'  
    remap_ua_detrac_labels(label_directory, save_directory_name, ua_detrac_to_coco)

   