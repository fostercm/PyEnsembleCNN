import torch
from torch import nn

# Get the last convolutional layer of the model for CAM purposes
def get_last_conv_layer(model):
    
    last_conv_layer = None
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer
            
    if last_conv_layer is None:
        raise Exception("No 2D convolutional layers found in model")
    
    return last_conv_layer  
    
# Replace the classification head of a CNN with a layer to output at a certain size
def replace_classifier(model, output_size, pooling='avg'):
    
    # Remove the classifier layer
    classifier_name = list(model.named_children())[-1][0]
    setattr(model, classifier_name, nn.Identity())
    
    # Get the input size of the classifier
    input_size = model(torch.randn(1, 3, 224, 224)).shape[1]
    
    # If we are downsizing, pool
    if input_size > output_size:
        if pooling == 'avg':
            setattr(model, classifier_name, nn.AdaptiveAvgPool1d(output_size))
        elif pooling == 'max':
            setattr(model, classifier_name, nn.AdaptiveMaxPool1d(output_size))
        else:
            raise ValueError('Pooling must be either "avg" or "max"')
    
    # If we are upsizing, interpolate
    elif input_size < output_size:
        setattr(model, classifier_name, Interpolate(size=output_size, mode='linear'))
    
# Implementation of the interpolate layer for upsizing
class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        x = x.squeeze(1)
        return x