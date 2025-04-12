# models/resnet50.py
import torch
import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes=21, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

if __name__ == "__main__":
    model = get_resnet50(num_classes=21, pretrained=False)
    print(model)
