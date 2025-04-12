# dataset.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

from config import DATA_ROOT, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TRAIN_VAL_SPLIT, VAL_SPLIT

def get_data_transforms():
    # Define transforms for training and evaluation
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # using ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, eval_transform

def load_datasets():
    train_transform, eval_transform = get_data_transforms()

    # Load the whole dataset using ImageFolder. 
    # It is assumed that DATA_ROOT folder is structured by class.
    full_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=eval_transform)
    
    # Get indices for train + val vs test split
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    split = int(np.floor(TRAIN_VAL_SPLIT * num_samples))  # e.g., 80% for train+val
    np.random.shuffle(indices)
    
    train_val_idx, test_idx = indices[:split], indices[split:]
    
    # Further split train_val into training and validation
    train_idx, val_idx = train_test_split(train_val_idx, test_size=VAL_SPLIT/ TRAIN_VAL_SPLIT, random_state=42)
    
    # Create subsets with proper transforms (we want augmentation only on training set)
    train_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=train_transform)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)
    
    return train_dataset, val_dataset, test_dataset

def get_dataloaders():
    train_dataset, val_dataset, test_dataset = load_datasets()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader
