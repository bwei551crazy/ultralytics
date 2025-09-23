#File for switching mismatch labels in used dataset to the labels used for COCO dataset

import os
from pathlib import Path

#class mapping for ua_detrac dataset to COCO dataset

#LEFT: UA_DETRAC CLASS ID
#RIGHT: COCO CLASS ID

ua_detrac_to_coco = {
    0: 7,   # Truck  
    1: 2,   # Car
    2: 80,  # Van    #cause Van doesn't exist in COCO
    3: 5    # Bus
}

def remap_ua_detrac_labels(label_dir, save_dir, mapping_dic):
    label_dir = Path(label_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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

if __name__ == "__main__":
    label_directory = 'datasets/uc-detrac (original)/content/UA-DETRAC/DETRAC_Upload/labels/train'
    save_directory = 'datasets/uc-detrac (original)/content/UA-DETRAC/DETRAC_Upload/labels/train_coco_map'
    remap_ua_detrac_labels(label_directory, save_directory, ua_detrac_to_coco)