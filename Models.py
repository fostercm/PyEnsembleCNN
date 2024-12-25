import torch
from torch import nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from Utils import get_last_conv_layer

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
        
        # Check model
        self.checks()
    
    def checks(self):
        # Check that the extractors output in the same dimension
        x = torch.randn(1, 3, 224, 224)
        if len(set([extractor(x).shape for extractor in self.extractors])) > 1:
            raise ValueError('Extractors output in different dimensions')
        
        # Check that the classifier input dimension matches the extractor output dimension
        first_layer = list(self.classifier.children())[0]
        if first_layer.in_features != self.extractors[0](x).shape[1]:
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
        
        # Move all extractors to device
        for extractor in self.extractors:
            extractor.to(device)
        
        # Move classifier to device
        self.classifier.to(device)

class StackEnsemble(nn.Module):
        
    def __init__(self, extractors, classifier):
        super(StackEnsemble, self).__init__()
        
        # Pretrained feature extractors
        self.extractors = extractors
        
        # Classifier
        self.classifier = classifier
        
        # Softmax layer
        self.output_softmax = torch.nn.Softmax(dim=1)
        
        # Check model
        self.checks()
    
    def checks(self):
        # Check that the classifier input dimension matches the sum of the extractor output dimensions
        x = torch.randn(1, 3, 224, 224)
        first_layer = list(self.classifier.children())[0]
        if first_layer.in_features != sum([extractor(x).shape[1] for extractor in self.extractors]):
            raise ValueError('Classifier input dimension does not match the sum of extractor output dimensions')
        
        # Freeze extractor weights
        for extractor in self.extractors:
            for param in extractor.parameters():
                param.requires_grad = False
    
    def __call__(self, x):
        # Extract features from each model
        extracted_features = torch.cat([extractor(x) for extractor in self.extractors], dim=1)
        
        # Classifier head
        x = self.classifier(extracted_features)
        
        # Softmax output
        return self.output_softmax(x)
    
    def get_CAM(self, x, class_idx):
        raise NotImplementedError('StackEnsemble does not support CAMs')
        
    def to(self, device):
        # Move model to device
        super().to(device)
        
        # Move all extractors to device
        for extractor in self.extractors:
            extractor.to(device)
        
        # Move classifier to device
        self.classifier.to(device)
        