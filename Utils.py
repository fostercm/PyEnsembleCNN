from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import torch
import random

class TestDataset(Dataset):
    def __init__(self, data_path, transform=None, target_count=None):

        # Store the directory path
        self.directory = data_path
        
        # Get list of all files
        self.files = os.listdir(self.directory)
        
        # Store transform
        self.transform = transform
        
        if target_count:
            self.files = random.sample(self.files, target_count)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        # Get filename
        filename = self.files[idx]
        
        # Construct full file path
        file_path = os.path.join(self.directory, filename)
        
        # Load image
        image = Image.open(file_path)
        
        # Apply transform if exists
        if self.transform:
            image = self.transform(image)
        
        return image

class TrainDataset(Dataset):
    def __init__(self, data_path, label_path, transform=None, target_count=None):

        # Read the CSV file
        self.labels = pd.read_csv(label_path)
        self.labels = torch.tensor((self.labels['level']))
        
        # Store the directory path
        self.directory = data_path
        
        # Get list of all files
        self.files = os.listdir(self.directory)
        
        # Store transform
        self.transform = transform
        
        # Get unique classes and their counts
        unique_classes, counts = torch.unique(self.labels, return_counts=True)
        self.class_counts = {cls.item(): count.item() for cls, count in zip(unique_classes, counts)}

        # Determine the target count for each class
        self.target_count = target_count or min(self.class_counts.values())

        # Generate balanced indices (if training)
        self.balanced_indices = self.get_balanced_indices()
    
    def get_balanced_indices(self):
        
        indices_per_class = {cls: (self.labels == cls).nonzero(as_tuple=True)[0] for cls in self.class_counts.keys()}
        balanced_indices = []

        for cls, indices in indices_per_class.items():
            sampled_indices = random.sample(indices.tolist(), self.target_count)
            balanced_indices.extend(sampled_indices)

        return balanced_indices
        
    
    def __len__(self):
        return len(self.balanced_indices)
    
    def __getitem__(self, idx):
        
        # Get index
        idx = self.balanced_indices[idx]
        
        # Get filename
        filename = self.files[idx]
        
        # Construct full file path
        file_path = os.path.join(self.directory, filename)
        
        # Load image
        image = Image.open(file_path)
        
        # Apply transform if exists
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]
    
def train(model, train_loader, val_loader, optimizer, criterion, device):
    
    model.train()
    train_loss = 0.0
    
    for i, data in enumerate(train_loader):
        
        # Get data from dataloader
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track loss
        train_loss += loss.item()
    
    val_loss = 0.0
    
    for i, data in enumerate(val_loader):
        
        # Get data from dataloader
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Track loss
        val_loss += loss.item()
        
    return train_loss / len(train_loader), val_loss / len(val_loader)

def process_data(data_path, label_path, batch_size, target_count=None):
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    train_dataset = TrainDataset(os.path.join(data_path,"train"), label_path, transform)
    test_dataset = TestDataset(os.path.join(data_path,"test"), transform, target_count)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader