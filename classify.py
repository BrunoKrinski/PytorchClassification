import os
import time
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int)
    parser.add_argument('--image', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    return parser.parse_args()

def predict(model_path, test_image_name, idx_to_class, size = 224):

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

    test_image = Image.open(test_image_name)
    #plt.imshow(test_image)
    
    test_image_tensor = transform(test_image)
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    
    model = torch.load(model_path)

    with torch.no_grad():
        model.eval()

        out = model(test_image_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(3, dim=1)
        cls = idx_to_class[topclass.cpu().numpy()[0][0]]
        score = topk.cpu().numpy()[0][0]

        for i in range(3):
            print("Prediction", i+1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])

def main():
    args = get_args()

    dataset = args.dataset
    train_directory = os.path.join(dataset, 'train')

    data = {
        'train': datasets.ImageFolder(root=train_directory),
    }

    idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}

    predict(args.model, args.image, idx_to_class, args.size)

if __name__ == '__main__':
    main()