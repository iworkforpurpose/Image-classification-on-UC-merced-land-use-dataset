# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from config import NUM_CLASSES, DEVICE, LEARNING_RATE, WEIGHT_DECAY

from models.lenet import LeNet5
from models.alexnet import get_alexnet
from models.vgg19 import get_vgg19
from models.resnet50 import get_resnet50
from train import train_model, test_model

def main():
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Get list of class names for confusion matrix plot
    classes = train_loader.dataset.classes

    # Dictionary to hold models
    model_dict = {
        'LeNet5': LeNet5(num_classes=NUM_CLASSES),
        'AlexNet': get_alexnet(num_classes=NUM_CLASSES, pretrained=True),
        'VGG19': get_vgg19(num_classes=NUM_CLASSES, pretrained=True),
        'ResNet50': get_resnet50(num_classes=NUM_CLASSES, pretrained=True)
    }
    
    # For each model: set criterion, optimizer, train, and test
    for model_name, model in model_dict.items():
        print(f"===== Training {model_name} =====")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        model = train_model(model, train_loader, val_loader, criterion, optimizer, model_name)
        print(f"===== Testing {model_name} =====")
        test_model(model, test_loader, classes, criterion, model_name)
        
if __name__ == "__main__":
    main()
