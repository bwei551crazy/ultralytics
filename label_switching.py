#File for switching labels in datasets around

import os
from pathlib import Path

#original
#0: truck
#1: car
#2: van
#3: bus

#UA-DETRAC dataset switchroo
#LEFT: UA-DETRAC CLASS ID Original
#RIGHT: New UA-DETRAC CLASS ID

ua_to_coco = {

    # Skip classes 6,7,8 as they don't have good COCO equivalents
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

def remap_labels(label_dir):
    """Remap UA-DETRAC labels to COCO class IDs"""
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(label_dir, label_file)
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_class = int(parts[0])
                    if old_class in ua_to_coco:
                        new_class = ua_to_coco[old_class]
                        parts[0] = str(new_class)
                        new_lines.append(' '.join(parts) + '\n')
            
            # Write back modified labels
            with open(label_path, 'w') as f:
                f.writelines(new_lines)

#switch 3rd parameter with the appropriate mapping dictionary if using different dataset

if __name__ == "__main__":

    #Change the path correspondingly to where the dataset is store (must be the absolute path)   
    label_directory = '/home/yanjiaqi/own_ultralytics/ultralytics/datasets/ua-detrac/UA-DETRAC_UPD_ANN/labels/train'
    
    remap_labels(label_directory)

   