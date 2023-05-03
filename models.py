import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import maxvit_t, MaxVit_T_Weights

def get_model(name, num_classes):
    
    if name == 'vgg16':
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        fc_inputs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(fc_inputs, num_classes)

    if name == 'vgg16_bn':
        model = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        
        fc_inputs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(fc_inputs, num_classes)
        
    elif name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        fc_inputs = model.fc.in_features
        model.fc = nn.Linear(fc_inputs, num_classes)

    if name == 'maxvit':
        model = maxvit_t(weights=MaxVit_T_Weights.DEFAULT)
        
        fc_inputs = model.classifier[5].in_features
        model.classifier[5] = nn.Linear(fc_inputs, num_classes)
    
    return model
