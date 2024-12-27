from ensembles import EnsembleTemplate
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from matplotlib import pyplot as plt
from typing import List, Tuple
from IPython.display import clear_output

def train(model: EnsembleTemplate, 
          train_loader: DataLoader, 
          val_loader: DataLoader, 
          optimizer: Optimizer, 
          criterion: nn.Module, 
          epochs: int, 
          device: str, 
          verbose: bool=False) -> Tuple[int,EnsembleTemplate]:
    """
    Train the model
    """
    train_losses = []
    val_losses = []
    
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        # Train head
        train_loss = 0
        for data, target in train_loader:
            
            # Retrieve ata and send to device
            data, target = data.to(device), target.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update loss
            train_loss += loss.item()
            
        train_losses.append(train_loss / len(train_loader))
        
        # Validation
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                
                # Retrieve data and send to device
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Calculate loss
                loss = criterion(output, target)
                
                # Update loss
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))
    
        # Show loss curves
        if verbose:
            live_plot(train_losses, val_losses)
        
        # Save the best model
        if val_losses[-1] == min(val_losses):
            best_epoch = epoch
            best_model_weights = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_weights)
    
    return best_epoch, model


def live_plot(train_loss: List[float], val_loss: List[float]) -> None:
    
    # Clear previous plot
    clear_output(wait=True)
    
    # Plot loss curves
    plt.plot(train_loss, label='Training Loss', marker='o')
    plt.plot(val_loss, label='Validation Loss', marker='o')
    
    # Add labels and legend
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Display plot
    plt.grid()
    plt.show()