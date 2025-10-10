from ultralytics import YOLO
import torch
import math

def direct_index_weight_transfer(pretrained_path, bdd_final_order, coco_index_mapping, save_path):
    """
    Transfer weights using direct COCO-to-BDD index mappings
    No need for complete COCO class names!
    """
    model = YOLO(pretrained_path)
    
    # coco_index_mapping: Dictionary of {coco_index: bdd_index}
    # Example: {7: 0} means "COCO class 7 (truck) should become BDD class 0"
    
    print("Direct index mapping:")
    for coco_idx, bdd_idx in coco_index_mapping.items():
        print(f"COCO index {coco_idx} → BDD index {bdd_idx}")
    
    # Get detection head
    detect_head = model.model[-1]
    original_weight = detect_head.cv3[0].conv.weight.data.clone()
    original_bias = detect_head.cv3[0].conv.bias.data.clone()
    
    anchors_per_level = 3
    bdd_nc = len(bdd_final_order)
    new_output_channels = (bdd_nc + 5) * anchors_per_level
    
    # Initialize new weights
    new_weight = torch.randn(new_output_channels, original_weight.shape[1], 
                            original_weight.shape[2], original_weight.shape[3])
    new_bias = torch.randn(new_output_channels)
    
    torch.nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))
    
    # Copy regression and objectness weights (unchanged)
    for anchor in range(anchors_per_level):
        for coord in range(4):
            pos = coord * anchors_per_level + anchor
            new_weight[pos] = original_weight[pos]
            new_bias[pos] = original_bias[pos]
        pos = 4 * anchors_per_level + anchor
        new_weight[pos] = original_weight[pos]
        new_bias[pos] = original_bias[pos]
    
    # Transfer class weights using direct index mapping
    print("\nTransferring class weights:")
    for coco_index, bdd_index in coco_index_mapping.items():
        print(f"  COCO index {coco_index} → BDD index {bdd_index}")
        for anchor in range(anchors_per_level):
            source_pos = (coco_index + 5) * anchors_per_level + anchor
            dest_pos = (bdd_index + 5) * anchors_per_level + anchor
            new_weight[dest_pos] = original_weight[source_pos]
            new_bias[dest_pos] = original_bias[source_pos]
    
    # Update model
    detect_head.cv3[0].conv.out_channels = new_output_channels
    detect_head.cv3[0].conv.weight.data = new_weight
    detect_head.cv3[0].conv.bias.data = new_bias
    model.model.nc = bdd_nc
    model.model.names = {i: name for i, name in enumerate(bdd_final_order)}
    model.overrides['names'] = bdd_final_order
    
    model.save(save_path)
    return model

# Usage example:
# Define your BDD class order (can be any order you want)
bdd_final_order = [
    'truck',        # 0
    'car',          # 1  
    'bus',          # 2
    'person',       # 3
    'traffic light', # 4
    'bicycle',      # 5
    'motorcycle',   # 6
    'rider',        # 7 (new class - random init)
    'lane1',        # 8 (new class - random init)
    # ... etc
]

# Define direct index mappings based on COCO indices:
# Key: COCO index, Value: Your BDD index
coco_index_mapping = {
    7: 0,   # COCO truck (7) → Your truck (0)
    2: 1,   # COCO car (2) → Your car (1)
    5: 2,   # COCO bus (5) → Your bus (2)
    0: 3,   # COCO person (0) → Your person (3)
    9: 4,   # COCO traffic light (9) → Your traffic light (4)
    1: 5,   # COCO bicycle (1) → Your bicycle (5)
    3: 6,   # COCO motorcycle (3) → Your motorcycle (6)
    # Note: rider (7), lane1 (8) etc. are new classes - no mapping needed
}

model = direct_index_weight_transfer(
    'yolo11m.pt',
    bdd_final_order,
    coco_index_mapping,
    'yolo11m_bdd_direct_index.pt'
)