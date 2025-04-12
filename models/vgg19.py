# models/vgg19.py
import torch
import torch.nn as nn
from torchvision import models

def get_vgg19(num_classes=21, pretrained=True):
    model = models.vgg19(pretrained=pretrained)
    # Replace the classifier's last layer
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model

if __name__ == "__main__":
    model = get_vgg19(num_classes=21, pretrained=False)
    print(model)
