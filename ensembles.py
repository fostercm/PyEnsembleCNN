import torch
from torch import nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from utils.modify import get_last_conv_layer
from abc import ABC, abstractmethod
from typing import List

class EnsembleTemplate(ABC):
    
    def __init__(self, extractors: List[nn.Module], head: nn.Module) -> None:
        
        # Pretrained feature extractors and head
        self.extractors = extractors
        self.head = head
        
        # Freeze extractor weights
        for extractor in self.extractors:
            for param in extractor.parameters():
                param.requires_grad = False
    
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def checks(self) -> None:
        pass
    
    # Move model to device
    def to(self, device: str) -> None:
        # Move all extractors to device
        for extractor in self.extractors:
            extractor.to(device)
        
        # Move head to device
        self.head.to(device)
    
    def train(self) -> None:
        # Set model to train mode
        self.head.train()
    
    def parameters(self) -> List[nn.Parameter]:
        # Return head parameters
        return self.head.parameters()
    
    def state_dict(self) -> dict:
        # Return head state dict
        return self.head.state_dict()
    
    def load_state_dict(self, state_dict: dict) -> None:
        # Load head state dict
        self.head.load_state_dict(state_dict)

class AverageEnsemble(EnsembleTemplate):
    
    def __init__(self, extractors: List[nn.Module], head: nn.Module, averaging='equal', CAM=False) -> None:
        
        super(AverageEnsemble, self).__init__(extractors, head)
        
        # Get CAMs
        if CAM:
            self.CAMs = [GradCAM(extractor, [get_last_conv_layer(extractor)]) for extractor in self.extractors]
        
        # Extractor proportions for averaging
        if averaging == 'equal':
            self.proportions = torch.ones(len(extractors))
        elif averaging == 'trained':
            self.proportions = nn.Parameter(torch.randn(len(extractors)))
        else:
            raise ValueError('Averaging must be either "equal" or "trained"')
        self.proportion_softmax = nn.Softmax(dim=0)
        
        # Check model
        self.checks()
    
    def checks(self) -> None:
        # Check that the extractors output in the same dimension
        x = torch.randn(1, 3, 224, 224)
        if len(set([extractor(x).shape for extractor in self.extractors])) > 1:
            raise ValueError('Extractors output in different dimensions')
        
        # Check that the head input dimension matches the extractor output dimension
        first_layer = list(self.head.children())[0]
        if first_layer.in_features != self.extractors[0](x).shape[1]:
            raise ValueError('Head input dimension does not match extractor output dimension')
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features from each model
        extracted_features = torch.stack([extractor(x) for extractor in self.extractors])
        
        # Turn proportions into a probability distribution
        proportions = self.proportion_softmax(self.proportions)
        
        # Weighted average of extractor outputs
        x = torch.sum(extracted_features * proportions[:, None, None], dim=0)
        
        # Classifier/Regression head
        return self.head(x)
    
    def get_CAM(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        # Unfreeze extractor weights
        for extractor in self.extractors:
            for param in extractor.parameters():
                param.requires_grad = True
        
        # Extract CAMs from each model
        CAMs = torch.stack([torch.tensor(cam(x, [ClassifierOutputTarget(class_idx)])) for cam in self.CAMs])
        
        # Freeze extractor weights
        for extractor in self.extractors:
            for param in extractor.parameters():
                param.requires_grad = False
        
        # Turn proportions into a probability distribution
        proportions = self.proportion_softmax(self.proportions)
        
        # Weighted average of CAMs
        return torch.sum(CAMs * proportions[:, None, None], dim=0).detach()
    
    def to(self, device: str) -> None:
        super(AverageEnsemble, self).to(device)
        self.proportions = self.proportions.to(device)

class StackEnsemble(EnsembleTemplate):
        
    def __init__(self, extractors: List[nn.Module], head: nn.Module) -> None:
        super(StackEnsemble, self).__init__(extractors, head)
        
        # Check model
        self.checks()
    
    def checks(self) -> None:
        # Check that the head input dimension matches the sum of the extractor output dimensions
        x = torch.randn(1, 3, 224, 224)
        first_layer = list(self.head.children())[0]
        if first_layer.in_features != sum([extractor(x).shape[1] for extractor in self.extractors]):
            raise ValueError('Head input dimension does not match the sum of extractor output dimensions')
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features from each model
        extracted_features = torch.cat([extractor(x) for extractor in self.extractors], dim=1)
        
        # Classifier/Regression head
        return self.head(extracted_features)