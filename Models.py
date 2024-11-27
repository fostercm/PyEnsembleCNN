from abc import ABC, abstractmethod
from torchvision import models # type: ignore
import torch # type: ignore
from torch import nn # type: ignore

# Abstract class for models
class ExtractorTemplate(ABC):
    
    @abstractmethod
    def __call__(self, x):
        "Forward pass"


class ResNetExtractor(ExtractorTemplate):
    
    def __init__(self, name):
        
        # Inputted model name
        self.name = name
        
        # Get the model and its weights
        if name == 'resnet18':
            self.extractor = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        elif name == 'resnet34':
            self.extractor = models.resnet34(weights='ResNet34_Weights.DEFAULT')
        elif name == 'resnet50':
            self.extractor = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        elif name == 'resnet101':
            self.extractor = models.resnet101(weights='ResNet101_Weights.DEFAULT')
        elif name == 'resnet152':
            self.extractor = models.resnet152(weights='ResNet152_Weights.DEFAULT')
        else:
            raise ValueError('Model not found')
        
        # Freeze weights
        for param in self.extractor.parameters():
            param.requires_grad = False
        
        # Get feature dimensionality
        self.feature_dim = self.extractor.fc.in_features
        
        # Remove fc layer
        self.extractor.fc = torch.nn.Identity()
    
    # Forward pass
    def __call__(self, x):
        return self.extractor(x)

class DenseNetExtractor(ExtractorTemplate):
    
    def __init__(self, name):
        
        # Inputted model name
        self.name = name
        
        # Get the model and its weights
        if name == 'densenet121':
            self.extractor = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
        elif name == 'densenet161':
            self.extractor = models.densenet161(weights='DenseNet161_Weights.DEFAULT')
        elif name == 'densenet169':
            self.extractor = models.densenet169(weights='DenseNet169_Weights.DEFAULT')
        elif name == 'densenet201':
            self.extractor = models.densenet201(weights='DenseNet201_Weights.DEFAULT')
        else:
            raise ValueError('Model not found')
        
        # Freeze weights
        for param in self.extractor.parameters():
            param.requires_grad = False
        
        # Get feature dimensionality
        self.feature_dim = self.extractor.classifier.in_features
        
        # Remove fc layer
        self.extractor.classifier = torch.nn.Identity()
    
    # Forward pass
    def __call__(self, x):
        return self.extractor(x)

class VGGExtractor(ExtractorTemplate):
    
    def __init__(self, name):
        
        # Inputted model name
        self.name = name
        
        # Get the model and its weights
        if name == 'vgg11':
            self.extractor = models.vgg11(weights='VGG11_Weights.DEFAULT')
        elif name == 'vgg13':
            self.extractor = models.vgg13(weights='VGG13_Weights.DEFAULT')
        elif name == 'vgg16':
            self.extractor = models.vgg16(weights='VGG16_Weights.DEFAULT')
        elif name == 'vgg19':
            self.extractor = models.vgg19(weights='VGG19_Weights.DEFAULT')
        else:
            raise ValueError('Model not found')
        
        # Freeze weights
        for param in self.extractor.parameters():
            param.requires_grad = False
        
        # Get feature dimensionality
        self.feature_dim = self.extractor.classifier[6].in_features
        
        # Remove fc layer
        self.extractor.classifier[6] = torch.nn.Identity()
    
    # Forward pass
    def __call__(self, x):
        return self.extractor(x)

class PredictionModel(nn.Module):
    
    def __init__(self, extractors, num_classes):
        
        super(PredictionModel, self).__init__()
        
        # Models in ensemble
        self.extractors = extractors
        
        # Input size of classifier head
        input_dim = sum([extractor.feature_dim for extractor in self.extractors])
        
        # Classifier layers
        self.fc1 = torch.nn.Linear(input_dim, 1024)
        self.act1 = torch.nn.SiLU()
        self.fc2 = torch.nn.Linear(1024, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)
    
    # Forward pass
    def __call__(self, x):
        
        # Extract features from each model
        extracted_features = [extractor(x) for extractor in self.extractors]
        x = torch.cat(extracted_features, dim=1)
        
        # Classifier head
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        
        return self.softmax(x)