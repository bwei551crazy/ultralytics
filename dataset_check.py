#visually inspect the bounding box of the dataset

import os
import cv2
import glob

def plot_bounding_boxes(img_dir, label_dir, output_dir = None):
    os.makedirs(output_dir, exist_ok=True)
    img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))
    
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
        
        if not os.path.exists(label_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image {img_path}")
            continue

        h, w, _ = img.shape
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            box_width = float(parts[3]) * w
            box_height = float(parts[4]) * h
            
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, str(class_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, img)
        else:
            cv2.imshow('Image with bounding boxes', img)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_directory = 'datasets/ua-detrac/images/train'
    label_directory = 'datasets/ua-detrac/labels/train'
    output_directory = 'datasets/ua-detrac/images/train_bbox'
    
    plot_bounding_boxes(img_directory, label_directory, output_directory)
