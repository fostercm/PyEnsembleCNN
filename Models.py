from abc import ABC, abstractmethod
from torchvision import models
import torch
from torch import nn
from pytorch_grad_cam import GradCAM # type: ignore
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget # type: ignore

# Abstract class for models
class ExtractorTemplate(ABC):
    
    @abstractmethod
    def __call__(self, x):
        "Forward pass"
    
    @abstractmethod
    def getCAM(self, x, class_idx):
        "Get class activation map"


class ResNetExtractor(ExtractorTemplate):
    
    def __init__(self, name):
        
        # Inputted model name
        self.name = name
        
        # Get the model and its weights
        if name == 'resnet18':
            self.extractor = models.resnet18(weights='ResNet18_Weights.DEFAULT')
            self.cam = GradCAM(self.extractor, [self.extractor.layer4[-1].conv2])
        elif name == 'resnet34':
            self.extractor = models.resnet34(weights='ResNet34_Weights.DEFAULT')
            self.cam = GradCAM(self.extractor, [self.extractor.layer4[-1].conv2])
        elif name == 'resnet50':
            self.extractor = models.resnet50(weights='ResNet50_Weights.DEFAULT')
            self.cam = GradCAM(self.extractor, [self.extractor.layer4[-1].conv3])
        elif name == 'resnet101':
            self.extractor = models.resnet101(weights='ResNet101_Weights.DEFAULT')
            self.cam = GradCAM(self.extractor, [self.extractor.layer4[-1].conv3])
        elif name == 'resnet152':
            self.extractor = models.resnet152(weights='ResNet152_Weights.DEFAULT')
            self.cam = GradCAM(self.extractor, [self.extractor.layer4[-1].conv3])
        else:
            raise ValueError('Model not found')
        
        # Freeze weights
        for param in self.extractor.parameters():
            param.requires_grad = False
        
        # Output in 512 dimensionality
        self.extractor.fc = nn.Linear(self.extractor.fc.in_features, 512)
    
    # Forward pass
    def __call__(self, x):
        return self.extractor(x)

    def getCAM(self, x, class_idx):
        
        # Unfreeze weights (gradients required)
        for param in self.extractor.parameters():
            param.requires_grad = True
            
        # Get CAM
        CAM = self.cam(x, [ClassifierOutputTarget(class_idx)])
        
        # Freeze weights
        for param in self.extractor.parameters():
            param.requires_grad = False
        
        for param in self.extractor.fc.parameters():
            param.requires_grad = True
        
        # Remove batch dimension
        return torch.tensor(CAM[0])

class DenseNetExtractor(ExtractorTemplate):
    
    def __init__(self, name):
        
        # Inputted model name
        self.name = name
        
        # Get the model and its weights
        if name == 'densenet121':
            self.extractor = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
            self.cam = GradCAM(self.extractor, [self.extractor.features.denseblock4.denselayer16.conv2])
        elif name == 'densenet161':
            self.extractor = models.densenet161(weights='DenseNet161_Weights.DEFAULT')
            self.cam = GradCAM(self.extractor, [self.extractor.features.denseblock4.denselayer24.conv2])
        elif name == 'densenet169':
            self.extractor = models.densenet169(weights='DenseNet169_Weights.DEFAULT')
            self.cam = GradCAM(self.extractor, [self.extractor.features.denseblock4.denselayer32.conv2])
        elif name == 'densenet201':
            self.extractor = models.densenet201(weights='DenseNet201_Weights.DEFAULT')
            self.cam = GradCAM(self.extractor, [self.extractor.features.denseblock4.denselayer32.conv2])
        else:
            raise ValueError('Model not found')
        
        # Freeze weights
        for param in self.extractor.parameters():
            param.requires_grad = False
        
        # Output in 512 dimensionality
        self.extractor.classifier = nn.Linear(self.extractor.classifier.in_features, 512)
    
    # Forward pass
    def __call__(self, x):
        return self.extractor(x)
    
    def getCAM(self, x, class_idx):
        
        # Unfreeze weights (gradients required)
        for param in self.extractor.parameters():
            param.requires_grad = True
        
        CAM = self.cam(x, [ClassifierOutputTarget(class_idx)])
        
        # Freeze weights
        for param in self.extractor.parameters():
            param.requires_grad = False
        
        for param in self.extractor.classifier.parameters():
            param.requires_grad = True
        
        # Remove batch dimension
        return torch.tensor(CAM[0])

class VGGExtractor(ExtractorTemplate):
    
    def __init__(self, name):
        
        # Inputted model name
        self.name = name
        
        # Get the model and its weights
        if name == 'vgg11':
            self.extractor = models.vgg11(weights='VGG11_Weights.DEFAULT')
            self.cam = GradCAM(self.extractor, [self.extractor.features[18]])
        elif name == 'vgg13':
            self.extractor = models.vgg13(weights='VGG13_Weights.DEFAULT')
            self.cam = GradCAM(self.extractor, [self.extractor.features[22]])
        elif name == 'vgg16':
            self.extractor = models.vgg16(weights='VGG16_Weights.DEFAULT')
            self.cam = GradCAM(self.extractor, [self.extractor.features[28]])
        elif name == 'vgg19':
            self.extractor = models.vgg19(weights='VGG19_Weights.DEFAULT')
            self.cam = GradCAM(self.extractor, [self.extractor.features[34]])
        else:
            raise ValueError('Model not found')
        
        # Freeze weights
        for param in self.extractor.parameters():
            param.requires_grad = False
        
        # Output in 512 dimensionality
        self.extractor.classifier[6] = nn.Linear(self.extractor.classifier[6].in_features, 512)
    
    # Forward pass
    def __call__(self, x):
        return self.extractor(x)
    
    def getCAM(self, x, class_idx):
        
        # Unfreeze weights (gradients required)
        for param in self.extractor.parameters():
            param.requires_grad = True
        
        CAM = self.cam(x, [ClassifierOutputTarget(class_idx)])
        
        # Freeze weights
        for param in self.extractor.parameters():
            param.requires_grad = False
        
        for param in self.extractor.classifier[6].parameters():
            param.requires_grad = True
        
        # Remove batch dimension
        return torch.tensor(CAM[0])

class PredictionModel(nn.Module):
    
    def __init__(self, extractors, num_classes):
        
        super(PredictionModel, self).__init__()
        
        # Models in ensemble
        self.extractors = extractors
        
        # Extractor proportions
        self.proportions = nn.Parameter(torch.randn(len(extractors)))
        
        # Classifier layers
        self.fc1 = torch.nn.Linear(512, 256)
        self.act1 = torch.nn.SiLU()
        self.fc2 = torch.nn.Linear(256, num_classes)
        
        # Softmax
        self.proportion_softmax = torch.nn.Softmax(dim=0)
        self.output_softmax = torch.nn.Softmax(dim=1)
    
    # Forward pass
    def __call__(self, x):
        
        # Extract features from each model
        extracted_features = torch.stack([extractor(x) for extractor in self.extractors])
        
        # Turn proportions into a probability distribution
        proportions = self.proportion_softmax(self.proportions)
        
        # Weighted average of extractor outputs
        x = torch.sum(extracted_features * proportions[:, None, None], dim=0)
        
        # Classifier head
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        
        return self.output_softmax(x)
    
    def getCAM(self, x, class_idx):
        
        # Extract features from each model
        CAMs = torch.stack([extractor.getCAM(x, class_idx) for extractor in self.extractors])
        
        # Turn proportions into a probability distribution
        proportions = self.proportion_softmax(self.proportions)
        
        # Weighted average of CAMs
        return torch.sum(CAMs * proportions[:, None, None], dim=0).detach()