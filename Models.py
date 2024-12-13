from abc import ABC, abstractmethod
from torchvision import models
import torch
from torch import nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Abstract class for models
class ExtractorTemplate(ABC):
    
    @abstractmethod
    def __call__(self, x):
        "Forward pass"
    
    @abstractmethod
    def getCAM(self, x, class_idx):
        "Get class activation map"

def get_last_conv_layer(model):
    last_conv_layer = None
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer
            
    if last_conv_layer is None:
        raise Exception("No 2D convolutional layers found in model")
    
    return last_conv_layer    

class AverageEnsemble(nn.Module):
    
    def __init__(self, extractors, classifier, CAM=False):
        
        super(AverageEnsemble, self).__init__()
        
        # Pretrained feature extractors
        self.extractors = extractors
        
        # Get CAMs if required
        if CAM:
            self.CAMs = [GradCAM(extractor, [get_last_conv_layer(extractor)]) for extractor in self.extractors]
        
        # Classifier
        self.classifier = classifier
        
        # Extractor proportions for averaging
        self.proportions = nn.Parameter(torch.randn(len(extractors)))
        
        # Softmax layers
        self.proportion_softmax = torch.nn.Softmax(dim=0)
        self.output_softmax = torch.nn.Softmax(dim=1)
    
    def checks(self):
        
        # Check that the extractors output in the same dimension
        x = torch.randn(1, 3, 224, 224)
        if len(set([extractor(x).shape for extractor in self.extractors])) > 1:
            raise ValueError('Extractors output in different dimensions')
        
        # Check that the classifier input dimension matches the extractor output dimension
        if self.classifier.in_features != self.extractors[0](x).shape[1]:
            raise ValueError('Classifier input dimension does not match extractor output dimension')
        
        # Freeze extractor weights
        for extractor in self.extractors:
            for param in extractor.parameters():
                param.requires_grad = False
    
    def __call__(self, x):
        
        # Extract features from each model
        extracted_features = torch.stack([extractor(x) for extractor in self.extractors])
        
        # Turn proportions into a probability distribution
        proportions = self.proportion_softmax(self.proportions)
        
        # Weighted average of extractor outputs
        x = torch.sum(extracted_features * proportions[:, None, None], dim=0)
        
        # Classifier head
        x = self.classifier(x)
        
        # Softmax output
        return self.output_softmax(x)
    
    def get_CAM(self, x, class_idx):
        
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
    
    def to(self, device):
        
        # Move model to device
        super().to(device)
        
        # Move all parameters to device
        for extractor in self.extractors:
            extractor.to(device)