from ultralytics import YOLO
import torch


def custom_classifier(pretrained_weights, pretrained_bias, picked_class) :
    custom_weights = pretrained_weights[picked_class]
    #If want unbiased results, uncomment
    #custom_bias = torch.zeros(5)
    custom_bias = pretrained_bias[picked_class]

    return custom_weights, custom_bias

def yolo_transfer(model_dict, picked_class, orig_weights, orig_bias):
    #perform weight transfer
    original_weights = model_dict['whatever key'].clone()
    original_bias = model_dict['whatever key'].clone()

    custom_weights, custom_bias = custom_classifier(
                                    orig_weights, 
                                    orig_bias,
                                    picked_class= picked_class)   

    model_dict['whatever key'] = custom_weights
    model_dict['whatever key'] = custom_bias

    torch.save(model_dict, f"yolo11m_{len(picked_class)}.pt")

def rtdetr_transfer(model_dict, picked_class, orig_weights, orig_bias):
    #perform weight transfer
    original_weights = model_dict['model.classifier.weight'].clone()
    original_bias = model_dict['model.classifier.bias'].clone()

    custom_weights, custom_bias = custom_classifier(
                                    orig_weights, 
                                    orig_bias,
                                    picked_class= picked_class)   

    model_dict['model.classifier.weight'] = custom_weights
    model_dict['model.classifier.bias'] = custom_bias

    torch.save(model_dict, f"rtdetr-l_{len(picked_class)}.pt")


#Makes the classes into the following order:
#car, bus, truck
#Makes it to where the labels 0, 1, 2, 3, 4 are the classes listed above
picked_class = [7, 2, 2, 5]

model = YOLO('yolo11m.pt')
 
# Get the state dict
pretrained = model.model.state_dict()
