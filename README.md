# PyEnsembleCNN: A dynamic CNN ensembling framework for PyTorch
A PyTorch Implementation of Self-Learning CNN Ensembles with Integrated Visualization

## Key Features

### Flexible ensemble architecture supporting any CNN backbone (ResNet, DenseNet, VGG, etc.)
* Novel trainable weighted averaging system that automatically learns optimal ensemble proportions
* Built-in Class Activation Mapping (CAM) visualization support for model interpretability

### Two distinct ensemble approaches:
* **AverageEnsemble:** Features learned weight distribution across models
* **StackEnsemble:** Concatenates features for enhanced representation power 

### Technical Innovation
The framework introduces a unique approach to ensemble weighting by making the model weights themselves trainable parameters. Instead of using static weights based on individual model performance, this implementation allows the ensemble to dynamically learn the optimal contribution of each model during training.

## Technical Details

### Usage
```python
# Imports
from torchvision.models import resnet50, densenet121, vgg16
from PyEnsembleCNN.ensembles import AverageEnsemble
from PyEnsembleCNN.utils.modify import replace_classifier
from PyEnsembleCNN.utils.train import train

# Example usage with common CNN architectures
extractors = [resnet50(weights='IMAGENET1K_V1'),
              densenet121(weights='IMAGENET1K_V1'),
              vgg16(weights='IMAGENET1K_V1')]  # Pre-trained models

# Replace the classification heads of each extractor with pooling or interpolation (downscaling vs upscaling)
for i in range(len(extractors)):
  replace_classifier(extractors[i], 2048, pooling='avg') # Supports avg and max pooling

# Simple MLP for classification
classifier = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Linear(512, num_classes)
)

# Initialize ensemble with automatic weight learning and train
model = AverageEnsemble(extractors, classifier, CAM=True)
best_epoch, best_model = train(model, train_loader, val_loader, optimizer, criterion, epochs, device='cuda', verbose=True)
```

### Advanced Features

* **Automatic Architecture Validation:** Built-in dimension checking ensures compatibility between extractors and classifiers
* **GPU Support:** Seamless device transition with comprehensive to(device) implementation
* **Flexible Feature Handling:** Supports both weighted averaging and feature stacking approaches
* **Integrated Visualization:** Native support for Class Activation Mapping in AverageEnsemble
* **Memory Efficient:** Automatic freezing of extractor weights to optimize memory usage

### Implementation Highlights
The core innovation lies in the AverageEnsemble class, which implements a weighted average of the ensembles through a learnable parameter:
```python
self.proportions = nn.Parameter(torch.randn(len(extractors)))
proportions = self.proportion_softmax(self.proportions)
```

## Applications
Originally developed for medical imaging applications, but designed to be domain-agnostic and applicable to any computer vision task requiring ensemble methods, including:

* Medical image analysis
* Object detection
* Image classification
* Visual reasoning tasks

## Future Development
* Benchmark comparisons against traditional ensemble methods
* Integration of additional visualization techniques
* Support for non-CNN architectures
* Performance optimization for large-scale deployments

## Installation and Dependencies
```bash
pip install PyEnsembleCNN
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
