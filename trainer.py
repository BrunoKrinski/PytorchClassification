import os
import time
import argparse
import numpy as np
import torch.nn as nn
import torch, torchvision
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from models import get_model
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model', type=str, default='vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    return parser.parse_args()

def train_and_validate(model, 
                       loss_criterion, 
                       optimizer, 
                       train_data_loader, 
                       valid_data_loader, 
                       train_data_size,
                       valid_data_size,
                       device, 
                       output,
                       epochs=25):
        
    start = time.time()
    history = []
    best_loss = 100000.0
    best_epoch = None

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        model.train()
        
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        for i, (inputs, labels) in enumerate(tqdm(train_data_loader)):

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            train_acc += acc.item() * inputs.size(0)
            
            #print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        with torch.no_grad():

            model.eval()

            for j, (inputs, labels) in enumerate(tqdm(valid_data_loader)):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch

        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        avg_valid_loss = valid_loss/valid_data_size 
        avg_valid_acc = valid_acc/valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                
        epoch_end = time.time()
    
        print("Epoch : {:03d}, Training: Loss - {:.4f}, Accuracy - {:.4f}%, Validation : Loss - {:.4f}, Accuracy - {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        
        torch.save(model, f"{output}/model_{epoch}.pt")
            
    return model, history, best_epoch

def main():
    args = get_args()

    size = args.size

    image_transforms = { 
        'train': transforms.Compose([
            transforms.Resize(size=size),
            transforms.CenterCrop(size=size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=size),
            transforms.CenterCrop(size=size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=size),
            transforms.CenterCrop(size=size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    dataset = args.dataset

    train_directory = os.path.join(dataset, 'train')
    valid_directory = os.path.join(dataset, 'valid')
    test_directory = os.path.join(dataset, 'test')

    bs = args.batch_size

    num_classes = len(os.listdir(valid_directory))
    print(num_classes)

    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
        'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
    }

    idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
    print(idx_to_class)

    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])
    test_data_size = len(data['test'])

    train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
    valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True)
    test_data_loader = DataLoader(data['test'], batch_size=bs, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(train_data_size, valid_data_size, test_data_size)
    
    model = get_model(args.model, num_classes=num_classes)
    #summary(model, input_size=(3, 224, 224), batch_size=1, device='cpu')
    
    model = model.to(device)

    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    trained_model, history, best_epoch = train_and_validate(model, 
                       loss_func, 
                       optimizer, 
                       train_data_loader, 
                       valid_data_loader, 
                       train_data_size,
                       valid_data_size,
                       device, 
                       args.output_dir,
                       epochs=args.epochs)

    torch.save(history, f"{args.output_dir}/{args.model}_history.pt")

if __name__ == '__main__':
    main()