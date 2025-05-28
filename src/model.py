# src/model.py

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes=4):
    """
    Creates a Faster R-CNN model with a ResNet50 FPN backbone.
    By default, pretrained on COCO. We then replace the classifier head.
    
    num_classes = 4 because you have:
       1 background class (by convention in some frameworks)
       + 3 classes (person=1, bike=2, car=3)
    """
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one for our classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
