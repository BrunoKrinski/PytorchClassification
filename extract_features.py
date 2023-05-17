import os
import json
import time
import glob
import argparse
import numpy as np
import torch.nn as nn
import torch, torchvision
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from models import get_model
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int)
    parser.add_argument('--folder', type=str)
    parser.add_argument('--model', type=str)
    return parser.parse_args()    

def predict(model_path, test_image_name, size = 224):

    image_transforms = { 
        'test': transforms.Compose([
            transforms.Resize(size=size),
            transforms.CenterCrop(size=size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }
       
    transform = image_transforms['test']

    test_image = Image.open(test_image_name).convert("RGB")
        
    test_image_tensor = transform(test_image)
        
    #model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    #summary(model, input_size=(3, 224, 224), batch_size=1, device='cpu')
    #print(model)
    
    #device = torch.device('cpu')
    #model = torch.load(model_path, map_location=device)
    #print(model)
    #summary(model, input_size=(3, 224, 224), batch_size=1, device='cpu')
    
    layer = model._modules.get('avgpool')
    
    my_embedding = torch.zeros(2048)
    
    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten()) 
    
    h = layer.register_forward_hook(copy_data)
    
    with torch.no_grad():
        model(test_image_tensor.unsqueeze(0))
    h.remove()
    
    return my_embedding.tolist()
    
    
def main():
    args = get_args()
    
    images_path = glob.glob(f'{args.folder}/*/*')
    for image in images_path:
        if '.json' not in image:
            print(image)    
            features = predict(args.model, image, args.size)
            
            json_path = image.replace('.jpg','.json')
            with open(json_path, 'w') as outfile:
                json.dump(features, outfile)
        

if __name__ == '__main__':
    main()