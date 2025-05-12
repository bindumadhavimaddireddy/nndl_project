#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision

from torch.utils.data import Dataset, DataLoader, BatchSampler, random_split
from torchvision import transforms
from PIL import Image


# In[11]:


# Create Dataset class for multilabel classification
class MultiClassImageDataset(Dataset):
    def __init__(self, ann_df, super_map_df, sub_map_df, img_dir, transform=None):
        self.ann_df = ann_df 
        self.super_map_df = super_map_df
        self.sub_map_df = sub_map_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ann_df)

    def __getitem__(self, idx):
        img_name = self.ann_df['image'][idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        super_idx = self.ann_df['superclass_index'][idx]
        super_label = self.super_map_df['class'][super_idx]
        
        sub_idx = self.ann_df['subclass_index'][idx]
        sub_label = self.sub_map_df['class'][sub_idx]
        
        if self.transform:
            image = self.transform(image)  
            
        return image, super_idx, super_label, sub_idx, sub_label

class MultiClassImageTestDataset(Dataset):
    def __init__(self, super_map_df, sub_map_df, img_dir, transform=None):
        self.super_map_df = super_map_df
        self.sub_map_df = sub_map_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): # Count files in img_dir
        return len([fname for fname in os.listdir(self.img_dir)])

    def __getitem__(self, idx):
        img_name = str(idx) + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)  
            
        return image, img_name


# In[12]:


train_ann_df = pd.read_csv('train_data.csv')
# test_ann_df = pd.read_csv('test_data.csv')
super_map_df = pd.read_csv('superclass_mapping.csv')
sub_map_df = pd.read_csv('subclass_mapping.csv')

train_img_dir = 'train_images'
test_img_dir = 'test_images'

image_preprocessing = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0), std=(1)),
])

# Create train and val split
train_dataset = MultiClassImageDataset(train_ann_df, super_map_df, sub_map_df, train_img_dir, transform=image_preprocessing)
train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1]) 

# Create test dataset
test_dataset = MultiClassImageTestDataset(super_map_df, sub_map_df, test_img_dir, transform=image_preprocessing)

# Create dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

val_loader = DataLoader(val_dataset, 
                        batch_size=batch_size, 
                        shuffle=True)

test_loader = DataLoader(test_dataset, 
                         batch_size=1, 
                         shuffle=False)


# In[13]:


class CNN(nn.Module):
    def __init__(self, input_size=64):
        super().__init__()
        
        self.feature_size = input_size // (2**3)
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding='same'), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding='same'), 
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc1 = nn.Linear(self.feature_size * self.feature_size * 128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3a = nn.Linear(128, 4)
        self.fc3b = nn.Linear(128, 88)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        super_out = self.fc3a(x)
        sub_out = self.fc3b(x)
        return super_out, sub_out

class Trainer():
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, test_loader=None, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train_epoch(self):
        running_loss = 0.0
        for i, data in enumerate(self.train_loader):
            inputs, super_labels, sub_labels = data[0].to(device), data[1].to(device), data[3].to(device)

            self.optimizer.zero_grad()
            super_outputs, sub_outputs = self.model(inputs)
            loss = self.criterion(super_outputs, super_labels) + self.criterion(sub_outputs, sub_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Training loss: {running_loss/i:.3f}')

    def validate_epoch(self):
        super_correct = 0
        sub_correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                inputs, super_labels, sub_labels = data[0].to(device), data[1].to(device), data[3].to(device)

                super_outputs, sub_outputs = self.model(inputs)
                loss = self.criterion(super_outputs, super_labels) + self.criterion(sub_outputs, sub_labels)
                _, super_predicted = torch.max(super_outputs.data, 1)
                _, sub_predicted = torch.max(sub_outputs.data, 1)
                
                total += super_labels.size(0)
                super_correct += (super_predicted == super_labels).sum().item()
                sub_correct += (sub_predicted == sub_labels).sum().item()
                running_loss += loss.item()            

        print(f'Validation loss: {running_loss/i:.3f}')
        print(f'Validation superclass acc: {100 * super_correct / total:.2f} %')
        print(f'Validation subclass acc: {100 * sub_correct / total:.2f} %') 

    def test(self, save_to_csv=False, return_predictions=False):
        if not self.test_loader:
            raise NotImplementedError('test_loader not specified')

        # Evaluate on test set, in this simple demo no special care is taken for novel/unseen classes
        test_predictions = {'image': [], 'superclass_index': [], 'subclass_index': []}
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, img_name = data[0].to(device), data[1]
        
                super_outputs, sub_outputs = self.model(inputs)
                _, super_predicted = torch.max(super_outputs.data, 1)
                _, sub_predicted = torch.max(sub_outputs.data, 1)
                
                test_predictions['image'].append(img_name[0])
                test_predictions['superclass_index'].append(super_predicted.item())
                test_predictions['subclass_index'].append(sub_predicted.item())
                
        test_predictions = pd.DataFrame(data=test_predictions)
        
        if save_to_csv:
            test_predictions.to_csv('example_test_predictions.csv', index=False)

        if return_predictions:
            return test_predictions


# In[14]:


# Init model and trainer
device = 'cpu'
model = CNN(input_size=64).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, test_loader)


# In[15]:


# Training loop
for epoch in range(20):
    print(f'Epoch {epoch+1}')
    trainer.train_epoch()
    trainer.validate_epoch()
    print('')

print('Finished Training')


# In[16]:


test_predictions = trainer.test(save_to_csv=True, return_predictions=True)


# In[17]:


# Quick script for evaluating generated csv files with ground truth

# super_correct = 0
# sub_correct = 0
# seen_super_correct = 0
# seen_sub_correct = 0
# unseen_super_correct = 0
# unseen_sub_correct = 0

# total = 0
# seen_super_total = 0
# unseen_super_total = 0
# seen_sub_total = 0
# unseen_sub_total = 0

# for i in range(len(test_predictions)):
#     super_pred = test_predictions['superclass_index'][i]
#     sub_pred = test_predictions['subclass_index'][i]

#     super_gt = test_ann_df['superclass_index'][i]
#     sub_gt = test_ann_df['subclass_index'][i]

#     # Total setting
#     if super_pred == super_gt:
#         super_correct += 1
#     if sub_pred == sub_gt:
#         sub_correct += 1
#     total += 1

#     # Unseen superclass setting
#     if super_gt == 3:
#         if super_pred == super_gt:
#             unseen_super_correct += 1
#         if sub_pred == sub_gt:
#             unseen_sub_correct += 1
#         unseen_super_total += 1
#         unseen_sub_total += 1
    
#     # Seen superclass, unseen subclass setting
#     if super_gt != 3 and sub_gt == 87:
#         if super_pred == super_gt:
#             seen_super_correct += 1
#         if sub_pred == sub_gt:
#             unseen_sub_correct += 1
#         seen_super_total += 1
#         unseen_sub_total += 1

#     # Seen superclass and subclass setting
#     if super_gt != 3 and sub_gt != 87:
#         if super_pred == super_gt:
#             seen_super_correct += 1
#         if sub_pred == sub_gt:
#             seen_sub_correct += 1
#         seen_super_total += 1
#         seen_sub_total += 1

# print('Superclass Accuracy')
# print(f'Overall: {100*super_correct/total:.2f} %')
# print(f'Seen: {100*seen_super_correct/seen_super_total:.2f} %')
# print(f'Unseen: {100*unseen_super_correct/unseen_super_total:.2f} %')

# print('\nSubclass Accuracy')
# print(f'Overall: {100*sub_correct/total:.2f} %')
# print(f'Seen: {100*seen_sub_correct/seen_sub_total:.2f} %')
# print(f'Unseen: {100*unseen_sub_correct/unseen_sub_total:.2f} %')

